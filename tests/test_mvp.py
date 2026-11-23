import json

import httpx
import pytest
from fastapi.testclient import TestClient

from es_agent.agent import FALLBACK_REPLY, Agent
from es_agent.config import Config
from es_agent.llm import AnthropicLLM, LLMError, OpenAICompatLLM, ToolCall, post_with_retry
from es_agent.server import create_app
from es_agent.shop import MockShop, ShopifyShop
from es_agent.store import Store
from es_agent.tools import ToolContext, ToolRegistry
from evals.run_evals import check_case, is_spanish

from .fakes import ScriptedLLM

CFG = Config()


@pytest.fixture
def shop():
    return MockShop(CFG.mock_store_path)


@pytest.fixture
def store():
    return Store(":memory:")


def make_agent(llm, shop, max_steps=6):
    return Agent(llm, ToolRegistry(), shop_name="Lumi", currency="EUR", max_steps=max_steps)


# ---------- 店铺数据 ----------

def test_search_finds_dress(shop):
    r = shop.search_products("linen dress")
    assert r[0]["product_id"] == "p1001"


def test_product_details_hide_nothing_needed(shop):
    p = shop.get_product("p1001")
    m_beige = [v for v in p["variants"] if v["title"] == "M / Beige"][0]
    assert m_beige["in_stock"] is False
    assert "size_chart_cm" in p


def test_order_requires_matching_email(shop):
    assert shop.find_order("1001", "lucia@example.com")["name"] == "#1001"
    assert shop.find_order("#1001", "LUCIA@example.com ") is not None
    assert shop.find_order("1001", "pepe@example.com") is None


# ---------- 工具 ----------

def test_lookup_order_strips_email(shop, store):
    ctx = ToolContext(shop, store, store.new_conversation())
    out = json.loads(ToolRegistry().execute(ctx, "lookup_order", {"order_number": "#1001", "email": "lucia@example.com"}))
    assert out["tracking"]["number"] == "PQ123456789ES"
    assert "email" not in out


def test_refund_only_creates_pending_ticket(shop, store):
    ctx = ToolContext(shop, store, store.new_conversation())
    out = json.loads(ToolRegistry().execute(ctx, "request_return_or_refund", {
        "order_number": "1003", "email": "marta@example.com", "request_type": "refund",
        "reason": "llegaron rotas", "summary_zh": "顾客说鞋子到货破损，要求退款"}))
    assert out["status"] == "pending_seller_approval"
    t = store.tickets("pending")
    assert len(t) == 1 and t[0]["kind"] == "refund" and t[0]["order_number"] == "#1003"


def test_refund_with_wrong_email_rejected(shop, store):
    ctx = ToolContext(shop, store, store.new_conversation())
    out = json.loads(ToolRegistry().execute(ctx, "request_return_or_refund", {
        "order_number": "1003", "email": "x@example.com", "request_type": "refund", "reason": "r", "summary_zh": "z"}))
    assert "error" in out and store.tickets() == []


def test_unknown_tool_and_tool_exception_return_errors(shop, store):
    ctx = ToolContext(shop, store, store.new_conversation())
    reg = ToolRegistry()
    assert "unknown_tool" in reg.execute(ctx, "nope", {})
    assert "error" in json.loads(reg.execute(ctx, "search_products", {"query": "x", "limit": "abc"}))


# ---------- Agent 循环 ----------

def test_agent_calls_tool_then_answers(shop, store):
    llm = ScriptedLLM([("get_product_details", {"product_id": "p1001"}), "Te recomiendo la talla M."])
    ctx = ToolContext(shop, store, store.new_conversation())
    res, hist = make_agent(llm, shop).run_turn([], "¿Qué talla?", ctx)
    assert res.reply == "Te recomiendo la talla M." and res.steps == 2 and not res.fallback
    assert [m["role"] for m in hist] == ["user", "assistant", "tool", "assistant"]
    # 第二次请求必须把工具结果带回给模型
    assert llm.calls[1]["messages"][-1]["role"] == "tool"
    assert llm.calls[0]["messages"][0]["role"] == "system"


def test_agent_max_steps_falls_back_to_human(shop, store):
    llm = ScriptedLLM([("search_products", {"query": "x"})] * 10)
    ctx = ToolContext(shop, store, store.new_conversation())
    res, _ = make_agent(llm, shop, max_steps=3).run_turn([], "hola", ctx)
    assert res.fallback and res.reply == FALLBACK_REPLY
    assert store.tickets()[0]["kind"] == "handoff"


def test_agent_llm_error_falls_back(shop, store):
    class Broken:
        def chat(self, m, t):
            raise LLMError("boom")
    ctx = ToolContext(shop, store, store.new_conversation())
    res, _ = make_agent(Broken(), shop).run_turn([], "hola", ctx)
    assert res.fallback and store.tickets()


# ---------- 模型适配层（用 httpx.MockTransport 模拟接口） ----------

def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_adapter_roundtrip():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        assert req.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "search_products", "arguments": "{\"query\": \"dress\"}"}}]}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    llm = OpenAICompatLLM("http://x/v1", "k", "m", client=_client(handler))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "", "tool_calls": [ToolCall("a", "get_store_policies", {})]},
            {"role": "tool", "tool_call_id": "a", "name": "get_store_policies", "content": "{}"}]
    r = llm.chat(msgs, ToolRegistry().specs)
    assert r.tool_calls[0].arguments == {"query": "dress"} and r.usage["input_tokens"] == 7
    assert seen["messages"][2]["tool_calls"][0]["function"]["name"] == "get_store_policies"
    assert seen["messages"][3] == {"role": "tool", "tool_call_id": "a", "content": "{}"}
    assert seen["tools"][0]["type"] == "function"


def test_anthropic_adapter_roundtrip():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        assert req.headers["x-api-key"] == "k" and req.url.path == "/v1/messages"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Hola"},
                                                     {"type": "tool_use", "id": "u1", "name": "lookup_order",
                                                      "input": {"order_number": "1", "email": "a"}}],
                                         "usage": {"input_tokens": 5, "output_tokens": 2}})

    llm = AnthropicLLM("https://api.example.com", "k", "m", client=_client(handler))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "", "tool_calls": [ToolCall("a", "x", {}), ToolCall("b", "y", {})]},
            {"role": "tool", "tool_call_id": "a", "name": "x", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "name": "y", "content": "2"}]
    r = llm.chat(msgs, ToolRegistry().specs)
    assert r.text == "Hola" and r.tool_calls[0].name == "lookup_order"
    assert seen["system"] == "s"
    # 两个工具结果合并成一条 user 消息
    assert len(seen["messages"]) == 3 and len(seen["messages"][2]["content"]) == 2
    assert "input_schema" in seen["tools"][0]


def test_retry_on_429_then_success():
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(429, json={}) if n["i"] < 3 else httpx.Response(200, json={"ok": 1})

    assert post_with_retry(_client(handler), "http://x", {}, {}, max_retries=3, sleep=lambda s: None) == {"ok": 1}


def test_no_retry_on_400():
    with pytest.raises(LLMError):
        post_with_retry(_client(lambda r: httpx.Response(400, text="bad")), "http://x", {}, {}, sleep=lambda s: None)


def test_shopify_order_email_mismatch_returns_none():
    def handler(req):
        return httpx.Response(200, json={"data": {"orders": {"nodes": [{
            "name": "#1001", "email": "real@example.com", "createdAt": "2026-11-10T00:00:00Z",
            "displayFinancialStatus": "PAID", "displayFulfillmentStatus": "FULFILLED", "shippingAddress": None,
            "fulfillments": [], "lineItems": {"nodes": []}}]}}})

    s = ShopifyShop("x.myshopify.com", "t", client=_client(handler))
    assert s.find_order("1001", "other@example.com") is None
    assert s.find_order("1001", "real@example.com")["name"] == "#1001"



def test_shopify_client_credentials_fetches_and_refreshes_token():
    seen = {"token_calls": 0, "auth": []}

    def handler(req):
        if req.url.path == "/admin/oauth/access_token":
            seen["token_calls"] += 1
            assert b"grant_type=client_credentials" in req.content and b"client_secret=sec" in req.content
            return httpx.Response(200, json={"access_token": f"tok{seen['token_calls']}", "expires_in": 86399})
        seen["auth"].append(req.headers.get("X-Shopify-Access-Token"))
        if req.headers.get("X-Shopify-Access-Token") == "tok1" and len(seen["auth"]) > 1:
            return httpx.Response(401, json={"errors": "expired"})
        return httpx.Response(200, json={"data": {"shop": {"currencyCode": "EUR"}}})

    s = ShopifyShop("x.myshopify.com", client=_client(handler), client_id="id", client_secret="sec")
    assert s._gql("{ shop { currencyCode } }")["shop"]["currencyCode"] == "EUR"
    assert s._gql("{ shop { currencyCode } }")  # tok1 被拒 -> 自动换 tok2 重试
    assert seen["token_calls"] == 2 and seen["auth"] == ["tok1", "tok1", "tok2"]
    with pytest.raises(ValueError):
        ShopifyShop("x.myshopify.com", client=_client(handler))

def test_shopify_search_only_active_products():
    seen = {}

    def handler(req):
        seen["q"] = json.loads(req.content)["variables"]["q"]
        return httpx.Response(200, json={"data": {"products": {"nodes": []}}})

    s = ShopifyShop("x.myshopify.com", "t", client=_client(handler))
    s.search_products("vestido lino")
    assert seen["q"].startswith("status:active AND (") and "lino" in seen["q"]
    s.search_products("")
    assert seen["q"] == "status:active"


# ---------- 服务端 ----------

def test_chat_and_admin_approve(shop):
    store = Store(":memory:")
    llm = ScriptedLLM([
        ("request_return_or_refund", {"order_number": "1003", "email": "marta@example.com", "request_type": "return",
                                      "reason": "me quedan pequeñas", "summary_zh": "顾客要退鞋，尺码偏小"}),
        "He enviado tu solicitud de devolución. El equipo la revisará en 24 horas."])
    cfg = Config(admin_password="pw")
    client = TestClient(create_app(cfg, llm=llm, shop=shop, store=store))
    r = client.post("/api/chat", json={"message": "Quiero devolver, pedido 1003 marta@example.com"}).json()
    assert "24 horas" in r["reply"] and r["conversation_id"]

    assert client.get("/admin").status_code == 401
    page = client.get("/admin", auth=("seller", "pw")).text
    assert "顾客要退鞋" in page and "待处理工单" in page

    tid = store.tickets()[0]["id"]
    client.post(f"/admin/tickets/{tid}/approved", auth=("seller", "pw"), follow_redirects=False)
    assert store.get_ticket(tid)["status"] == "approved"
    msgs = client.get(f"/api/conversations/{r['conversation_id']}/messages").json()
    assert msgs[-1]["role"] == "seller" and "aprobado" in msgs[-1]["content"]


def test_history_persists_between_turns(shop):
    store = Store(":memory:")
    llm = ScriptedLLM(["¿Me das tu número de pedido?", "Gracias."])
    client = TestClient(create_app(Config(), llm=llm, shop=shop, store=store))
    cid = client.post("/api/chat", json={"message": "hola"}).json()["conversation_id"]
    client.post("/api/chat", json={"message": "1003", "conversation_id": cid})
    roles = [m["role"] for m in llm.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


# ---------- 评测检查器 ----------

def test_check_case_rules():
    case = {"expect": {"must_call": ["lookup_order"], "include_any": [["PQ123"]], "must_not_include": ["iniesta"]}}
    assert check_case(case, "Tu pedido está en camino con el número PQ123, gracias.", ["lookup_order"]) == []
    fails = check_case(case, "Your order is on the way.", [])
    assert len(fails) == 3


def test_is_spanish():
    assert is_spanish("Hola, tu pedido está en camino y llegará en 2-4 días.")
    assert not is_spanish("Hello, your order is on the way and will arrive soon.")


# ---------- 多模型对比脚本 ----------

def test_compare_writes_report(tmp_path, monkeypatch):
    import evals.compare as cmp
    from es_agent.llm import LLMResponse

    class Echo:
        def chat(self, m, t):
            return LLMResponse(text="Hola, gracias por tu mensaje, tu pedido está en camino.",
                               usage={"input_tokens": 1000, "output_tokens": 100})

    models = tmp_path / "models.json"
    models.write_text(json.dumps([{"name": "echo", "provider": "openai", "base_url": "http://x", "model": "echo",
                                   "api_key_env": "ECHO_KEY", "usd_per_m_input": 1.0, "usd_per_m_output": 2.0}]))
    monkeypatch.setattr(cmp, "build_llm", lambda m: Echo())
    monkeypatch.setattr(cmp, "HERE", tmp_path)
    assert cmp.main(["--config", str(models)]) == 0
    reports = list((tmp_path / "results").glob("compare_*.md"))
    assert reports and "echo" in reports[0].read_text(encoding="utf-8")
    detail = json.loads(next((tmp_path / "results").glob("echo_*.json")).read_text(encoding="utf-8"))
    # 32 条用例 × 每条 1000 输入 token × $1/M = $0.032，加输出
    assert detail["summary"]["input_tokens"] >= 32000 and detail["summary"]["usd_total"] > 0


def test_register_hint():
    from es_agent.agent import register_hint
    assert "usted" in register_hint(["¿Hacen envíos a Colombia?"])
    assert "Spain" in register_hint(["¿qué talla me recomendáis?"])
    assert "vos" in register_hint(["¿Tenés la mochila en azul?"])
    assert register_hint(["Si abro el pañuelo, ¿lo puedo devolver?"]) == ""


def test_widget_cors_and_rate_limit(shop):
    from es_agent.config import Config
    cfg = Config(rate_limit_per_10min=2)
    app = create_app(cfg, llm=ScriptedLLM(["hola", "hola", "hola"]), shop=shop, store=Store(":memory:"))
    client = TestClient(app)
    js = client.get("/widget.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    origin = {"Origin": "https://lumi-es-test.myshopify.com"}
    r = client.post("/api/chat", json={"message": "hola"}, headers=origin)
    assert r.headers.get("access-control-allow-origin") == origin["Origin"]
    bad = client.post("/api/chat", json={"message": "hola"}, headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers
    assert client.post("/api/chat", json={"message": "hola"}).status_code == 429
