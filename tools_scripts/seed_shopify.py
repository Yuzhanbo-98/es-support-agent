"""把评测需要的店铺政策和 3 张测试订单写进 Shopify 开发者商店。

商品请先用 test_store/shopify_products.csv 在后台导入。本脚本只补政策和订单，
订单内容和 data/mock_store.json 里的 #1001-#1003 一致，评测用例才能在真实店铺上通过。

用法（.env 或环境变量里要有 SHOPIFY_STORE_DOMAIN 和 token 或 client id/secret）：
    python -m tools_scripts.seed_shopify            # 写入
    python -m tools_scripts.seed_shopify --dry-run  # 只检查连接和现有数据

需要的权限：write_legal_policies, write_orders, write_draft_orders,
write_merchant_managed_fulfillment_orders（只在测试店用，客服 Agent 本身只读）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from es_agent.config import Config  # noqa: E402
from es_agent.shop import ShopifyShop  # noqa: E402

POLICY_TYPES = {  # store_policies.txt 的段落 -> Shopify 政策类型
    "RETURNS": "REFUND_POLICY", "EXCHANGES": "REFUND_POLICY",
    "SHIPPING": "SHIPPING_POLICY", "PAYMENT": "TERMS_OF_SALE",
}


def load_policies() -> dict[str, str]:
    text = (ROOT / "test_store" / "store_policies.txt").read_text(encoding="utf-8")
    out: dict[str, list[str]] = {}
    for title, body in re.findall(r"== (\w+) ==\n(.*?)(?=\n== |\Z)", text, re.S):
        out.setdefault(POLICY_TYPES[title], []).append(f"<h3>{title.title()}</h3><p>{body.strip()}</p>")
    return {k: "".join(v) for k, v in out.items()}


def check(errors: list, what: str) -> None:
    if errors:
        raise SystemExit(f"{what} 失败：{errors}")


def seed_policies(shop: ShopifyShop) -> None:
    m = """mutation($p: ShopPolicyInput!) { shopPolicyUpdate(shopPolicy: $p) {
             shopPolicy { type } userErrors { field message } } }"""
    for ptype, body in load_policies().items():
        r = shop._gql(m, {"p": {"type": ptype, "body": body}})["shopPolicyUpdate"]
        check(r["userErrors"], f"政策 {ptype}")
        print(f"  政策已写入：{ptype}")


def variant_ids(shop: ShopifyShop) -> dict[tuple[str, str], str]:
    q = "{ productVariants(first: 100) { nodes { id title product { title } } } }"
    return {(v["product"]["title"], v["title"]): v["id"] for v in shop._gql(q)["productVariants"]["nodes"]}


def existing_orders(shop: ShopifyShop) -> set[str]:
    q = "{ orders(first: 50, sortKey: CREATED_AT) { nodes { name email } } }"
    return {o["name"] for o in shop._gql(q)["orders"]["nodes"]}


def create_order(shop: ShopifyShop, o: dict, variants: dict) -> None:
    lines = []
    for it in o["items"]:
        vid = variants.get((it["title"], it["variant"]))
        if not vid:
            raise SystemExit(f"店里找不到变体 {it['title']} / {it['variant']}，请先导入 shopify_products.csv")
        lines.append({"variantId": vid, "quantity": it["quantity"]})
    first = o["email"].split("@")[0].title()
    m = """mutation($o: OrderCreateOrderInput!, $opt: OrderCreateOptionsInput) {
             orderCreate(order: $o, options: $opt) {
               order { id name fulfillmentOrders(first: 5) { nodes { id } } } userErrors { field message } } }"""
    order_input = {
        "email": o["email"], "lineItems": lines, "financialStatus": "PAID",
        "processedAt": f"{o['created_at']}T10:00:00Z",
        "shippingAddress": {"firstName": first, "lastName": "Test", "address1": "Calle Mayor 1",
                            "city": "Madrid" if o["shipping_country"] == "ES" else "Ciudad de México",
                            "zip": "28013" if o["shipping_country"] == "ES" else "06000",
                            "countryCode": o["shipping_country"]},
    }
    opts = {"inventoryBehaviour": "BYPASS", "sendReceipt": False, "sendFulfillmentReceipt": False}
    r = shop._gql(m, {"o": order_input, "opt": opts})["orderCreate"]
    check(r["userErrors"], f"订单 {o['name']}")
    order = r["order"]
    print(f"  订单已创建：{order['name']}（期望 {o['name']}）")
    if order["name"] != o["name"]:
        print("  注意：订单号和评测用例不一致，商店里可能已有别的订单。")

    if o["fulfillment_status"] != "FULFILLED":
        return
    t = o.get("tracking") or {}
    fm = """mutation($f: FulfillmentInput!) { fulfillmentCreate(fulfillment: $f) {
              fulfillment { id } userErrors { field message } } }"""
    f = shop._gql(fm, {"f": {
        "lineItemsByFulfillmentOrder": [{"fulfillmentOrderId": n["id"]} for n in order["fulfillmentOrders"]["nodes"]],
        "trackingInfo": {"company": t.get("company"), "number": t.get("number"), "url": t.get("url")},
        "notifyCustomer": False}})["fulfillmentCreate"]
    check(f["userErrors"], f"发货 {o['name']}")
    print(f"  已发货：{o['name']} {t.get('company')} {t.get('number')}")

    if o.get("delivered"):
        em = """mutation($e: FulfillmentEventInput!) { fulfillmentEventCreate(fulfillmentEvent: $e) {
                  fulfillmentEvent { status } userErrors { field message } } }"""
        try:
            e = shop._gql(em, {"e": {"fulfillmentId": f["fulfillment"]["id"], "status": "DELIVERED",
                                     "happenedAt": f"{o['delivered_at']}T12:00:00Z"}})["fulfillmentEventCreate"]
            check(e["userErrors"], f"签收 {o['name']}")
            print(f"  已标记签收：{o['name']} {o['delivered_at']}")
        except (RuntimeError, SystemExit) as exc:
            print(f"  标记签收失败（可在后台手动标记）：{exc}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = Config.from_env()
    shop = ShopifyShop(cfg.shopify_store_domain, cfg.shopify_admin_token, cfg.shopify_api_version,
                       client_id=cfg.shopify_client_id, client_secret=cfg.shopify_client_secret)
    print(f"已连接：{shop.name}（{cfg.shopify_store_domain}），货币 {shop.currency}")
    variants = variant_ids(shop)
    have = existing_orders(shop)
    print(f"  商品变体 {len(variants)} 个，已有订单 {sorted(have) or '无'}")
    if shop.currency != "EUR":
        print("  注意：店铺货币不是 EUR，评测和政策按欧元写。请在后台「设置 > 常规 > 商店货币」改成 EUR。")
    if args.dry_run:
        return 0
    seed_policies(shop)
    mock = json.loads((ROOT / "data" / "mock_store.json").read_text(encoding="utf-8"))
    for o in mock["orders"]:
        if o["name"] in have:
            print(f"  跳过：{o['name']} 已存在")
            continue
        create_order(shop, o, variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
