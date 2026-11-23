"""测试用的假模型：按脚本依次返回预设的回复，不发网络请求。"""
from es_agent.llm import LLMResponse, ToolCall


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        if not self.script:
            return LLMResponse(text="")
        item = self.script.pop(0)
        if isinstance(item, str):
            return LLMResponse(text=item, usage={"input_tokens": 10, "output_tokens": 5})
        name, args = item
        return LLMResponse(text="", tool_calls=[ToolCall(id=f"c{len(self.calls)}", name=name, arguments=args)],
                           usage={"input_tokens": 10, "output_tokens": 5})
