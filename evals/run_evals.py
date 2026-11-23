"""评测：用固定的模拟店铺数据，跑 evals/cases.jsonl 里的每个用例。

规则检查（不需要额外模型）：
  - must_call / must_not_call：有没有调用（或避免调用）某个工具
  - include_any：每一组里至少出现一个关键词（忽略大小写和重音）
  - must_not_include：一个都不能出现（防编造、防泄露）
  - spanish：回复必须是西语
可选 --judge：再让模型按评分标准打分（有没有编造、语气和称谓是否合适）。

用法：
  python -m evals.run_evals                 # 全部用例
  python -m evals.run_evals --only size_dress_es
  python -m evals.run_evals --judge --out report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from es_agent.agent import Agent
from es_agent.config import Config
from es_agent.llm import LLM, make_llm
from es_agent.shop import make_shop
from es_agent.store import Store
from es_agent.tools import ToolContext, ToolRegistry

CASES = Path(__file__).parent / "cases.jsonl"

ES_WORDS = {"de", "la", "el", "que", "y", "en", "los", "las", "por", "para", "con", "tu", "su", "es", "una",
            "un", "te", "se", "no", "del", "al", "puedes", "esta", "pedido", "talla", "envio", "hola", "gracias",
            "tenemos", "nuestro", "nuestra", "dias", "le", "usted", "si", "muy", "pero", "como"}
EN_WORDS = {"the", "and", "you", "your", "is", "are", "to", "of", "for", "with", "it", "this", "that", "we",
            "can", "our", "please", "thanks", "order", "days"}


def norm(s: str) -> str:
    s = re.sub(r"[*_`]", "", s)  # 去掉 Markdown 加粗/斜体，否则 "todavía **no**" 匹配不到 "todavia no"
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def is_spanish(text: str) -> bool:
    words = re.findall(r"[a-z]+", norm(text))
    es = sum(w in ES_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return es >= 2 and es > en


def check_case(case: dict, reply: str, called: list[str]) -> list[str]:
    exp = case["expect"]
    fails = []
    r = norm(reply)
    for t in exp.get("must_call", []):
        if t not in called:
            fails.append(f"没有调用 {t}")
    for t in exp.get("must_not_call", []):
        if t in called:
            fails.append(f"不该调用 {t}")
    for group in exp.get("include_any", []):
        if not any(norm(k) in r for k in group):
            fails.append(f"回复缺少其一：{group}")
    for k in exp.get("must_not_include", []):
        if norm(k) in r:
            fails.append(f"回复不该出现：{k}")
    if exp.get("spanish", True) and not is_spanish(reply):
        fails.append("回复不是西语")
    return fails


JUDGE_PROMPT = """You are grading a customer-service reply from a Spanish-language e-commerce assistant.
Return ONLY a JSON object: {{"grounded": 1-5, "tone": 1-5, "comment": "<one short sentence>"}}

grounded: 5 = every factual claim is supported by the TOOL RESULTS below (or the assistant honestly says it
does not know / hands off); 1 = invents facts (sizes, stock, dates, policies) not in the tool results.
Store rules the assistant was given (these are NOT invented facts): requests to the seller are reviewed within
24 hours and answered by email; refunds/cancellations are never done by the assistant itself.
Store rule: the assistant ALWAYS answers in Spanish, even to customers writing in English; that is correct, not a tone problem.
tone: 5 = natural Spanish suited to the customer's variety and register (tú/usted/vos as the customer used{register}),
concise and polite; 1 = unnatural, wrong register, or rude.

CUSTOMER MESSAGES:
{turns}

TOOL RESULTS:
{tools}

ASSISTANT FINAL REPLY:
{reply}
"""


def judge(llm: LLM, case: dict, reply: str, trace: list[dict]) -> dict:
    reg = case["expect"].get("register")
    prompt = JUDGE_PROMPT.format(
        register=f"; expected register: {reg}" if reg else "",
        turns="\n".join(case["turns"]),
        tools="\n".join(f"- {t.get('name')}({json.dumps(t.get('arguments'), ensure_ascii=False)}) -> {t.get('result')}"
                        for t in trace) or "(none)",
        reply=reply)
    resp = llm.chat([{"role": "user", "content": prompt}], [])
    m = re.search(r"\{.*\}", resp.text, re.S)
    try:
        return json.loads(m.group(0)) if m else {"error": resp.text[:200]}
    except json.JSONDecodeError:
        return {"error": resp.text[:200]}


_SHOPS: dict = {}


def _shop_for(cfg: Config):
    """SHOP_BACKEND=mock 用模拟数据，=shopify 用真实开发者商店。同一后端只建一次（复用 token）。"""
    key = (cfg.shop_backend, cfg.shopify_store_domain, cfg.mock_store_path)
    if key not in _SHOPS:
        _SHOPS[key] = make_shop(cfg)
    return _SHOPS[key]


def run_case(llm: LLM, case: dict, cfg: Config) -> dict:
    shop = _shop_for(cfg)
    store = Store(":memory:")
    agent = Agent(llm, ToolRegistry(), shop_name=shop.name, currency=shop.currency,
                  max_steps=cfg.max_agent_steps)
    cid = store.new_conversation()
    history: list[dict] = []
    trace: list[dict] = []
    reply = ""
    usage = {"input_tokens": 0, "output_tokens": 0}
    start = time.perf_counter()
    for turn in case["turns"]:
        ctx = ToolContext(shop=shop, store=store, conversation_id=cid)
        result, history = agent.run_turn(history, turn, ctx)
        trace += result.tool_trace
        reply = result.reply
        for k in usage:
            usage[k] += result.usage.get(k, 0)
    called = [t["name"] for t in trace if "name" in t]
    return {"id": case["id"], "variant": case["variant"], "reply": reply, "called": called, "trace": trace,
            "failures": check_case(case, reply, called), "usage": usage,
            "seconds": round(time.perf_counter() - start, 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="只跑这些用例 id")
    ap.add_argument("--judge", action="store_true", help="额外用模型打分")
    ap.add_argument("--out", default="eval_report.json")
    args = ap.parse_args(argv)

    cfg = Config.from_env()
    llm = make_llm(cfg)
    cases = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]

    results = []
    for c in cases:
        try:
            res = run_case(llm, c, cfg)
        except Exception as e:
            res = {"id": c["id"], "variant": c["variant"], "reply": "", "called": [], "trace": [],
                   "failures": [f"运行出错：{e}"], "usage": {}, "seconds": 0}
        if args.judge and res["reply"]:
            res["judge"] = judge(llm, c, res["reply"], res["trace"])
            j = res["judge"]
            if isinstance(j.get("grounded"), int) and j["grounded"] < 4:
                res["failures"].append(f"评审：可能编造（grounded={j['grounded']}）")
            if isinstance(j.get("tone"), int) and j["tone"] < 4:
                res["failures"].append(f"评审：语气/称谓不合适（tone={j['tone']}）")
        ok = not res["failures"]
        print(f"{'✓' if ok else '✗'} {res['id']:<28} {res['variant']:<9} {res['seconds']:>5}s  "
              + ("" if ok else "; ".join(res["failures"])))
        results.append(res)

    by_variant = defaultdict(lambda: [0, 0])
    for r in results:
        by_variant[r["variant"]][1] += 1
        by_variant[r["variant"]][0] += not r["failures"]
    passed = sum(not r["failures"] for r in results)
    print(f"\n通过 {passed}/{len(results)}（{passed / max(len(results), 1):.0%}）")
    for v, (p, n) in sorted(by_variant.items()):
        print(f"  {v:<10} {p}/{n}")
    tokens_in = sum(r["usage"].get("input_tokens", 0) for r in results)
    tokens_out = sum(r["usage"].get("output_tokens", 0) for r in results)
    print(f"token：输入 {tokens_in}，输出 {tokens_out}，模型 {cfg.llm_model}")

    Path(args.out).write_text(json.dumps({"model": cfg.llm_model, "passed": passed, "total": len(results),
                                          "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
