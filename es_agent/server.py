"""FastAPI 服务：顾客聊天接口 + 演示店铺页 + 卖家中文后台。

启动：uvicorn es_agent.server:app --reload
  演示店铺（顾客视角）：http://localhost:8000/
  卖家后台（中文）：   http://localhost:8000/admin   （用户名任意，密码 = ADMIN_PASSWORD）
"""
from __future__ import annotations

import html
import json
import os
import secrets
import time
from pathlib import Path

from collections import defaultdict, deque

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from .agent import Agent
from .config import Config
from .llm import LLM, make_llm
from .shop import Shop, make_shop
from .store import Store
from .tools import ToolContext, ToolRegistry

STATIC = Path(__file__).parent / "static"

KIND_ZH = {"return": "退货", "refund": "退款", "exchange": "换货", "order_change": "改订单",
           "cancel": "取消订单", "handoff": "转人工"}
STATUS_ZH = {"pending": "待处理", "approved": "已同意", "rejected": "已拒绝", "done": "已处理"}

# 卖家点“同意/拒绝”后，自动发给顾客的西语通知（MVP 用模板，后续可让 Agent 生成）
DECISION_MSG_ES = {
    "approved": "Buenas noticias: hemos aprobado tu solicitud ({kind_es}) del pedido {order}. "
                "Te enviaremos las instrucciones por email.",
    "rejected": "Hemos revisado tu solicitud ({kind_es}) del pedido {order} y lamentablemente no podemos "
                "aprobarla según nuestra política. Te escribiremos por email con más detalles.",
    "done": "Nuestro equipo ha revisado tu consulta y te ha respondido por email.",
}
KIND_ES = {"return": "devolución", "refund": "reembolso", "exchange": "cambio", "order_change": "cambio de pedido",
           "cancel": "cancelación", "handoff": "consulta"}


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


class ChatOut(BaseModel):
    conversation_id: str
    reply: str
    handed_off: bool = False


def create_app(cfg: Config | None = None, *, llm: LLM | None = None, shop: Shop | None = None,
               store: Store | None = None) -> FastAPI:
    cfg = cfg or Config.from_env()
    if os.environ.get("RENDER") and cfg.admin_password == "change-me":
        raise RuntimeError("部署到公网前请设置 ADMIN_PASSWORD（卖家后台密码）")
    llm = llm or make_llm(cfg)
    shop = shop or make_shop(cfg)
    store = store or Store(cfg.db_path)
    registry = ToolRegistry()
    shop_name = getattr(shop, "name", None) or "Tienda"
    agent = Agent(llm, registry, shop_name=shop_name, currency=getattr(shop, "currency", "EUR"),
                  max_steps=cfg.max_agent_steps)

    app = FastAPI(title="西语客服 Agent MVP")
    security = HTTPBasic()

    # 商店前台（*.myshopify.com 或 WIDGET_ALLOWED_ORIGINS 里的自定义域名）可以跨域调用聊天接口
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"https://[a-z0-9-]+\.myshopify\.com",
                       allow_origins=cfg.widget_allowed_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    # 公开接口会花模型费用：每个 IP 每 10 分钟最多 RATE_LIMIT_PER_10MIN 条消息
    hits: dict[str, deque] = defaultdict(deque)

    def rate_limited(ip: str) -> bool:
        now, q = time.time(), hits[ip]
        while q and now - q[0] > 600:
            q.popleft()
        if len(q) >= cfg.rate_limit_per_10min:
            return True
        q.append(now)
        return False

    def admin_auth(creds: HTTPBasicCredentials = Depends(security)):
        if not secrets.compare_digest(creds.password, cfg.admin_password):
            raise HTTPException(401, "密码错误", headers={"WWW-Authenticate": "Basic"})

    @app.post("/api/chat", response_model=ChatOut)
    def chat(body: ChatIn, request: Request):
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "")).split(",")[0]
        if rate_limited(ip.strip()):
            raise HTTPException(429, "Demasiados mensajes")
        cid = body.conversation_id
        history = store.get_history(cid) if cid else None
        if history is None:
            cid = store.new_conversation()
            history = []
        ctx = ToolContext(shop=shop, store=store, conversation_id=cid)
        result, history = agent.run_turn(history, body.message, ctx)
        store.save_history(cid, history)
        store.add_message(cid, "customer", body.message)
        store.add_message(cid, "agent", result.reply)
        return ChatOut(conversation_id=cid, reply=result.reply,
                       handed_off=any(e.get("kind") == "handoff" for e in result.events))

    @app.get("/api/conversations/{cid}/messages")
    def conversation_messages(cid: str):
        return store.messages(cid)

    @app.get("/widget.js")
    def widget():
        return Response((STATIC / "widget.js").read_text(encoding="utf-8"), media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=300"})

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def demo_store():
        return (STATIC / "demo.html").read_text(encoding="utf-8")

    @app.get("/admin", response_class=HTMLResponse, dependencies=[Depends(admin_auth)])
    def admin():
        return render_admin(store)

    @app.post("/admin/tickets/{tid}/{decision}", dependencies=[Depends(admin_auth)])
    def decide(tid: int, decision: str):
        if decision not in ("approved", "rejected", "done"):
            raise HTTPException(400, "无效操作")
        t = store.get_ticket(tid)
        if not t:
            raise HTTPException(404, "工单不存在")
        store.decide_ticket(tid, decision)
        # 把卖家的决定用西语告诉顾客（写进对话，顾客下次刷新聊天窗口能看到）
        msg = DECISION_MSG_ES[decision].format(kind_es=KIND_ES.get(t["kind"], "solicitud"),
                                               order=t["order_number"] or "")
        store.add_message(t["conversation_id"], "seller", msg)
        hist = store.get_history(t["conversation_id"]) or []
        hist.append({"role": "assistant", "content": msg, "tool_calls": []})
        store.save_history(t["conversation_id"], hist)
        return RedirectResponse("/admin", status_code=303)

    app.state.store = store
    app.state.agent = agent
    return app


def _fmt_time(ts: float | None) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts)) if ts else ""


def render_admin(store: Store) -> str:
    s = store.stats()
    e = html.escape
    rows = []
    for t in store.tickets():
        payload = json.loads(t["payload"] or "{}")
        transcript = "".join(
            f"<div class='m {e(m['role'])}'><b>{ {'customer': '顾客', 'agent': 'Agent', 'seller': '卖家'}.get(m['role'], m['role']) }：</b>{e(m['content'])}</div>"
            for m in store.messages(t["conversation_id"]))
        actions = ""
        if t["status"] == "pending":
            if t["kind"] == "handoff":
                actions = f"<form method='post' action='/admin/tickets/{t['id']}/done'><button>标记已处理</button></form>"
            else:
                actions = (f"<form method='post' action='/admin/tickets/{t['id']}/approved'><button class='ok'>同意</button></form>"
                           f"<form method='post' action='/admin/tickets/{t['id']}/rejected'><button class='no'>拒绝</button></form>")
        rows.append(f"""
        <div class='card {e(t['status'])}'>
          <div class='head'><span class='tag'>{e(KIND_ZH.get(t['kind'], t['kind']))}</span>
            <span>#{t['id']} · {e(STATUS_ZH.get(t['status'], t['status']))} · {_fmt_time(t['created_at'])}</span>
            {f"<span>订单 {e(t['order_number'])}</span>" if t['order_number'] else ''}</div>
          <p class='sum'>{e(t['summary_zh'] or '（无摘要）')}</p>
          {f"<p class='reason'>顾客原话：{e(payload.get('reason', ''))}</p>" if payload.get('reason') else ''}
          <details><summary>查看对话原文</summary>{transcript}</details>
          <div class='actions'>{actions}</div>
        </div>""")
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>西语客服后台</title>
<style>
body{{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#f6f6f4;color:#222}}
header{{background:#1f2937;color:#fff;padding:16px 24px}} main{{max-width:900px;margin:0 auto;padding:16px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}}
.stat{{background:#fff;border-radius:10px;padding:14px}} .stat b{{display:block;font-size:26px}}
.card{{background:#fff;border-radius:10px;padding:14px 16px;margin-bottom:12px;border-left:4px solid #f59e0b}}
.card.approved,.card.done{{border-color:#10b981;opacity:.75}} .card.rejected{{border-color:#ef4444;opacity:.75}}
.head{{display:flex;gap:12px;flex-wrap:wrap;font-size:13px;color:#555;align-items:center}}
.tag{{background:#eef2ff;color:#3730a3;border-radius:6px;padding:2px 8px;font-weight:600}}
.sum{{font-size:15px;margin:10px 0 6px}} .reason{{color:#555;font-size:13px;margin:0 0 8px}}
.m{{font-size:13px;padding:4px 0;border-bottom:1px dashed #eee}} .actions{{display:flex;gap:8px;margin-top:10px}}
button{{border:0;border-radius:6px;padding:6px 14px;cursor:pointer;background:#e5e7eb}}
button.ok{{background:#10b981;color:#fff}} button.no{{background:#ef4444;color:#fff}}
</style></head><body><header><b>西语客服 Agent · 卖家后台</b></header><main>
<div class='stats'>
 <div class='stat'>会话总数<b>{s['conversations']}</b></div>
 <div class='stat'>Agent 自动解决<b>{s['auto_resolved']}</b></div>
 <div class='stat'>自动解决率<b>{s['auto_rate']*100:.0f}%</b></div>
 <div class='stat'>待处理工单<b>{s['pending_tickets']}</b></div>
</div>
<h3>工单（退货 / 退款 / 转人工）</h3>
{''.join(rows) or '<p>暂无工单。</p>'}
</main></body></html>"""


def _lazy_app():
    return create_app()


# uvicorn es_agent.server:app 时才真正创建（避免 import 时就要求配置）
class _LazyApp:
    _app = None

    async def __call__(self, scope, receive, send):
        if _LazyApp._app is None:
            _LazyApp._app = _lazy_app()
        await _LazyApp._app(scope, receive, send)


app = _LazyApp()
