"""工具注册：每个工具 = 一个 Python 函数 + 一份 JSON Schema。

原则：
- 顾客侧 Agent 只有只读工具，加两个“提交给卖家”的工具（退货/退款申请、转人工）。
- 任何会改变订单或退钱的动作，都只会生成待审批工单，不会直接执行。
- 订单信息必须同时提供订单号和下单邮箱才会返回，防止被人套取别人的订单。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .shop import Shop
from .store import Store

MAX_RESULT_CHARS = 6000


@dataclass
class ToolContext:
    shop: Shop
    store: Store
    conversation_id: str
    events: list[dict] = field(default_factory=list)  # 本轮产生的工单等，给评测和前端用


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[[ToolContext, dict], Any]

    @property
    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def search_products(ctx: ToolContext, a: dict):
    results = ctx.shop.search_products(a.get("query", ""), limit=int(a.get("limit", 5)))
    return {"results": results, "note": "Use get_product_details for sizes, stock, materials." if results
            else "No matching products in this store."}


def get_product_details(ctx: ToolContext, a: dict):
    p = ctx.shop.get_product(a.get("product_id", ""))
    return p or {"error": "product_not_found"}


def get_store_policies(ctx: ToolContext, a: dict):
    return ctx.shop.get_policies()


def lookup_order(ctx: ToolContext, a: dict):
    o = ctx.shop.find_order(a.get("order_number", ""), a.get("email", ""))
    if not o:
        return {"error": "order_not_found_or_email_mismatch",
                "note": "Ask the customer to double-check the order number and the email used for the order."}
    o.pop("email", None)
    return o


def request_return_or_refund(ctx: ToolContext, a: dict):
    order = ctx.shop.find_order(a.get("order_number", ""), a.get("email", ""))
    if not order:
        return {"error": "order_not_found_or_email_mismatch",
                "note": "Cannot open a request without a valid order number + email."}
    tid = ctx.store.create_ticket(
        ctx.conversation_id, kind=a.get("request_type", "return"), summary_zh=a.get("summary_zh", ""),
        payload={"reason": a.get("reason", ""), "items": a.get("items", []), "order": order},
        order_number=order["name"], email=a.get("email", ""))
    ctx.events.append({"type": "ticket", "id": tid, "kind": a.get("request_type", "return")})
    return {"status": "pending_seller_approval", "ticket_id": tid,
            "note": "The request was sent to the seller. Nothing has been refunded or changed yet. "
                    "Tell the customer the team will review it within 24 hours and reply by email. "
                    "Do NOT promise approval or a refund."}


def handoff_to_human(ctx: ToolContext, a: dict):
    tid = ctx.store.create_ticket(ctx.conversation_id, kind="handoff", summary_zh=a.get("summary_zh", ""),
                                  payload={"reason": a.get("reason", "")})
    ctx.events.append({"type": "ticket", "id": tid, "kind": "handoff"})
    return {"status": "handed_off", "ticket_id": tid,
            "note": "Tell the customer a team member will reply by email within 24 hours."}


SUMMARY_ZH = {"type": "string",
              "description": "给中国卖家看的中文摘要：顾客是谁、要什么、关键信息（订单号、商品、原因）。1-3 句。"}

TOOLS: list[Tool] = [
    Tool("search_products",
         "Search the store catalog. The catalog is in English, so translate the customer's need into "
         "short English keywords (e.g. 'linen dress', 'running shoes').",
         _obj({"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
         search_products),
    Tool("get_product_details",
         "Get full live data for one product: description, material, care, size chart, fit notes, "
         "shipping origin, every variant with price and stock.",
         _obj({"product_id": {"type": "string"}}, ["product_id"]), get_product_details),
    Tool("get_store_policies", "Get the store's return, exchange, shipping and payment policies.",
         _obj({}, []), get_store_policies),
    Tool("lookup_order",
         "Look up an order's status and tracking. Requires BOTH the order number and the email used to order.",
         _obj({"order_number": {"type": "string"}, "email": {"type": "string"}}, ["order_number", "email"]),
         lookup_order),
    Tool("request_return_or_refund",
         "Open a return / refund / exchange / order-change request for the SELLER TO APPROVE. "
         "Use only after the customer gave order number + email and a reason. Does not refund anything itself.",
         _obj({"order_number": {"type": "string"}, "email": {"type": "string"},
               "request_type": {"type": "string", "enum": ["return", "refund", "exchange", "order_change", "cancel"]},
               "reason": {"type": "string", "description": "Customer's reason, in Spanish as they said it."},
               "items": {"type": "array", "items": {"type": "string"}},
               "summary_zh": SUMMARY_ZH},
              ["order_number", "email", "request_type", "reason", "summary_zh"]),
         request_return_or_refund),
    Tool("handoff_to_human",
         "Hand the conversation to the seller's team when the store data does not contain the answer, "
         "the customer is angry or asks for a human, or the request is outside your tools.",
         _obj({"reason": {"type": "string"}, "summary_zh": SUMMARY_ZH}, ["reason", "summary_zh"]),
         handoff_to_human),
]


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self.tools = {t.name: t for t in (tools or TOOLS)}

    @property
    def specs(self) -> list[dict]:
        return [t.spec for t in self.tools.values()]

    def execute(self, ctx: ToolContext, name: str, args: dict) -> str:
        start = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            result: Any = {"error": f"unknown_tool: {name}"}
        else:
            try:
                result = tool.fn(ctx, args or {})
            except Exception as e:  # 工具出错要回传给模型，而不是让整个对话崩掉
                result = {"error": f"{type(e).__name__}: {e}"}
        text = json.dumps(result, ensure_ascii=False)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "...(truncated)"
        ctx.store.log_tool(ctx.conversation_id, name, args or {}, text, int((time.perf_counter() - start) * 1000))
        return text
