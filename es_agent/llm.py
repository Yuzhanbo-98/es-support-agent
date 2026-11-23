"""手写的模型适配层：只用 httpx 发 HTTP 请求，不用任何厂商 SDK。

内部统一的消息格式（与厂商无关）：
  {"role": "system", "content": str}
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

工具定义统一为：{"name": str, "description": str, "parameters": <JSON Schema>}
每个适配器负责把统一格式转换成各家接口的格式，再把返回结果转换回来。
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def post_with_retry(
    client: httpx.Client,
    url: str,
    headers: dict,
    payload: dict,
    max_retries: int = 3,
    base_delay: float = 1.0,
    sleep=time.sleep,
) -> dict:
    """POST，遇到限流 / 5xx / 超时按指数退避重试。"""
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.post(url, headers=headers, json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = e
        else:
            if resp.status_code < 400:
                return resp.json()
            if resp.status_code not in RETRYABLE_STATUS:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            last_err = LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            retry_after = resp.headers.get("retry-after")
            if retry_after and retry_after.replace(".", "", 1).isdigit() and attempt < max_retries:
                sleep(min(float(retry_after), 30))
                continue
        if attempt < max_retries:
            sleep(base_delay * (2**attempt) + random.uniform(0, 0.25))
    raise LLMError(f"重试 {max_retries} 次后仍失败：{last_err}")


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"_value": val}
    except json.JSONDecodeError:
        return {"_raw": raw}


class OpenAICompatLLM:
    """OpenAI 兼容的 /chat/completions 接口：OpenAI、DeepSeek、通义千问、Ollama、vLLM 等。"""

    def __init__(self, base_url: str, api_key: str, model: str, *, temperature: float = 0.2,
                 timeout: float = 60, max_retries: int = 3, client: httpx.Client | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = client or httpx.Client(timeout=timeout)

    @staticmethod
    def to_wire_messages(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                        for tc in m["tool_calls"]
                    ]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    @staticmethod
    def to_wire_tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["parameters"]}} for t in tools]

    @staticmethod
    def parse(data: dict) -> LLMResponse:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"无法解析返回：{str(data)[:300]}") from e
        calls = [
            ToolCall(id=tc.get("id") or f"call_{i}", name=tc["function"]["name"],
                     arguments=_parse_args(tc["function"].get("arguments")))
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        u = data.get("usage") or {}
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls,
                           usage={"input_tokens": u.get("prompt_tokens", 0),
                                  "output_tokens": u.get("completion_tokens", 0)})

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": self.to_wire_messages(messages),
                                   "temperature": self.temperature}
        if tools:
            payload["tools"] = self.to_wire_tools(tools)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return self.parse(post_with_retry(self.client, self.url, headers, payload, self.max_retries))


class AnthropicLLM:
    """Anthropic Messages 接口（/v1/messages），直接用 HTTP 调用。"""

    def __init__(self, base_url: str, api_key: str, model: str, *, max_tokens: int = 1024,
                 temperature: float = 0.2, timeout: float = 60, max_retries: int = 3,
                 client: httpx.Client | None = None):
        base = base_url.rstrip("/")
        self.url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = client or httpx.Client(timeout=timeout)

    @staticmethod
    def to_wire(messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_parts.append(m["content"])
            elif role == "user":
                out.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
            elif role == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                # 连续的工具结果要合并进同一条 user 消息
                if out and out[-1]["role"] == "user" and out[-1]["content"] and \
                        out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return "\n\n".join(system_parts), out

    @staticmethod
    def parse(data: dict) -> LLMResponse:
        if "content" not in data:
            raise LLMError(f"无法解析返回：{str(data)[:300]}")
        texts, calls = [], []
        for b in data["content"]:
            if b.get("type") == "text":
                texts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                calls.append(ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {}))
        u = data.get("usage") or {}
        return LLMResponse(text="".join(texts), tool_calls=calls,
                           usage={"input_tokens": u.get("input_tokens", 0),
                                  "output_tokens": u.get("output_tokens", 0)})

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        system, wire = self.to_wire(messages)
        payload: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                   "temperature": self.temperature, "messages": wire}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [{"name": t["name"], "description": t["description"],
                                 "input_schema": t["parameters"]} for t in tools]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        return self.parse(post_with_retry(self.client, self.url, headers, payload, self.max_retries))


def make_llm(cfg) -> LLM:
    if cfg.llm_provider == "anthropic":
        return AnthropicLLM(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)
    if cfg.llm_provider == "openai":
        return OpenAICompatLLM(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)
    raise ValueError(f"未知的 LLM_PROVIDER：{cfg.llm_provider}")
