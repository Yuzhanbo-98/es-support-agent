"""店铺数据层：同一套接口，两种实现。

- MockShop：读 data/mock_store.json，本地开发和评测用。
- ShopifyShop：用 httpx 直接调 Shopify Admin GraphQL API，读真实店铺或开发者商店。

Agent 只通过这四个方法拿数据，所以回答永远以店铺实时数据为准。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Protocol

import httpx


class Shop(Protocol):
    def search_products(self, query: str, limit: int = 5) -> list[dict]: ...
    def get_product(self, product_id: str) -> dict | None: ...
    def get_policies(self) -> dict: ...
    def find_order(self, order_number: str, email: str) -> dict | None: ...


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def _normalize_order_number(n: str) -> str:
    digits = re.sub(r"\D", "", n or "")
    return f"#{digits}" if digits else ""


def _summary(p: dict) -> dict:
    in_stock = [v for v in p["variants"] if v["inventory"] > 0]
    prices = sorted({float(v["price"]) for v in p["variants"]})
    return {
        "product_id": p["id"],
        "title": p["title"],
        "type": p.get("product_type", ""),
        "price_range": f"{prices[0]:.2f}-{prices[-1]:.2f}" if prices else "",
        "available_variants": len(in_stock),
        "total_variants": len(p["variants"]),
    }


def _stock_hint(qty: int) -> dict:
    """不把精确库存交给模型：只有剩 1-3 件时才标“少量”，避免回复里泄露库存数。"""
    return {"low_stock": True} if 0 < qty <= 3 else {}


class MockShop:
    def __init__(self, path: str):
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.currency = self.data["shop"]["currency"]
        self.name = self.data["shop"]["name"]

    def search_products(self, query: str, limit: int = 5) -> list[dict]:
        q = _tokens(query)
        scored = []
        for p in self.data["products"]:
            hay = _tokens(" ".join([p["title"], p.get("product_type", ""), " ".join(p.get("tags", [])),
                                    p.get("description", "")]))
            title = _tokens(p["title"] + " " + " ".join(p.get("tags", [])))
            score = len(q & hay) + 2 * len(q & title)
            if score or not q:
                scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        return [_summary(p) for _, p in scored[:limit]]

    def get_product(self, product_id: str) -> dict | None:
        for p in self.data["products"]:
            if p["id"] == product_id:
                d = {k: v for k, v in p.items() if k != "variants"}
                d["currency"] = self.currency
                d["variants"] = [
                    {"title": v["title"], "price": v["price"], "options": v["options"],
                     "in_stock": v["inventory"] > 0, **_stock_hint(v["inventory"])}
                    for v in p["variants"]
                ]
                return d
        return None

    def get_policies(self) -> dict:
        return dict(self.data["policies"])

    def find_order(self, order_number: str, email: str) -> dict | None:
        num = _normalize_order_number(order_number)
        for o in self.data["orders"]:
            if o["name"] == num and o["email"].lower() == (email or "").strip().lower():
                return dict(o)
        return None


class ShopifyShop:
    """直接 POST 到 https://{shop}/admin/api/{version}/graphql.json。

    需要的权限：read_products, read_inventory, read_orders, read_legal_policies。
    """

    PRODUCT_FIELDS = """
      id title productType tags description
      options { name values }
      variants(first: 50) { nodes { title price inventoryQuantity selectedOptions { name value } } }
      metafields(first: 20) { nodes { namespace key value } }
    """

    def __init__(self, domain: str, token: str = "", api_version: str = "2025-10",
                 client: httpx.Client | None = None, client_id: str = "", client_secret: str = ""):
        """两种认证方式：
        - token：旧版自定义应用的永久 Admin API token（shpat_...）。
        - client_id + client_secret：Dev Dashboard 新版应用，用 client credentials grant
          换取 24 小时有效的 token，过期前自动续期。应用和商店必须属于同一个组织。
        """
        self.domain = domain
        self.url = f"https://{domain}/admin/api/{api_version}/graphql.json"
        self.client = client or httpx.Client(timeout=20)
        self.client_id, self.client_secret = client_id, client_secret
        self._token = token
        self._token_expires_at = float("inf") if token else 0.0
        if not token and not (client_id and client_secret):
            raise ValueError("ShopifyShop 需要 token，或者 client_id + client_secret")
        self._currency: str | None = None
        self._name: str | None = None

    def _access_token(self, force: bool = False) -> str:
        if self.client_id and self.client_secret and (force or time.time() >= self._token_expires_at - 60):
            r = self.client.post(f"https://{self.domain}/admin/oauth/access_token",
                                 data={"grant_type": "client_credentials", "client_id": self.client_id,
                                       "client_secret": self.client_secret})
            r.raise_for_status()
            body = r.json()
            self._token = body["access_token"]
            self._token_expires_at = time.time() + int(body.get("expires_in", 86399))
        return self._token

    def _gql(self, query: str, variables: dict | None = None) -> dict:
        payload = {"query": query, "variables": variables or {}}
        r = self.client.post(self.url, headers=self._headers(), json=payload)
        if r.status_code == 401 and self.client_id:
            # token 被提前作废（比如卖家重装了应用），换一个新的再试一次
            r = self.client.post(self.url, headers=self._headers(force=True), json=payload)
        r.raise_for_status()
        body = r.json()
        if body.get("errors"):
            raise RuntimeError(f"Shopify GraphQL 错误：{body['errors']}")
        return body["data"]

    def _headers(self, force: bool = False) -> dict:
        headers = {"Content-Type": "application/json"}
        token = self._access_token(force)
        if token:
            headers["X-Shopify-Access-Token"] = token
        return headers

    def _load_shop_info(self) -> None:
        info = self._gql("{ shop { name currencyCode } }")["shop"]
        self._name, self._currency = info["name"], info["currencyCode"]

    @property
    def currency(self) -> str:
        if self._currency is None:
            self._load_shop_info()
        return self._currency

    @property
    def name(self) -> str:
        if self._name is None:
            self._load_shop_info()
        return self._name

    @staticmethod
    def _to_internal(n: dict) -> dict:
        variants = [
            {"title": v["title"], "price": v["price"], "inventory": v.get("inventoryQuantity") or 0,
             "options": {o["name"]: o["value"] for o in v.get("selectedOptions", [])}}
            for v in n["variants"]["nodes"]
        ]
        extra = {f"{m['namespace']}.{m['key']}": m["value"] for m in (n.get("metafields") or {}).get("nodes", [])}
        return {"id": n["id"], "title": n["title"], "product_type": n.get("productType", ""),
                "tags": n.get("tags", []), "description": n.get("description", ""),
                "options": n.get("options", []), "metafields": extra, "variants": variants}

    def search_products(self, query: str, limit: int = 5) -> list[dict]:
        q = """query($q: String, $n: Int!) { products(first: $n, query: $q) { nodes { %s } } }""" % self.PRODUCT_FIELDS
        words = " OR ".join(sorted(_tokens(query))) or None
        nodes = self._gql(q, {"q": words, "n": limit})["products"]["nodes"]
        return [_summary(self._to_internal(n)) for n in nodes]

    def get_product(self, product_id: str) -> dict | None:
        q = """query($id: ID!) { product(id: $id) { %s } }""" % self.PRODUCT_FIELDS
        n = self._gql(q, {"id": product_id})["product"]
        if not n:
            return None
        p = self._to_internal(n)
        p["currency"] = self.currency
        p["variants"] = [{"title": v["title"], "price": v["price"], "options": v["options"],
                          "in_stock": v["inventory"] > 0, **_stock_hint(v["inventory"])} for v in p["variants"]]
        return p

    def get_policies(self) -> dict:
        data = self._gql("{ shop { shopPolicies { type body } } }")
        # 隐私政策是 Shopify 自动生成的长文，和客服问题无关，只会浪费 token
        return {pol["type"].lower(): re.sub(r"<[^>]+>", " ", pol["body"]).strip()
                for pol in data["shop"]["shopPolicies"] if pol["type"] != "PRIVACY_POLICY"}

    def find_order(self, order_number: str, email: str) -> dict | None:
        num = _normalize_order_number(order_number)
        if not num:
            return None
        q = """query($q: String!) { orders(first: 1, query: $q) { nodes {
              name email createdAt processedAt displayFinancialStatus displayFulfillmentStatus
              shippingAddress { countryCodeV2 }
              fulfillments { status deliveredAt trackingInfo { company number url } }
              lineItems(first: 20) { nodes { title variantTitle quantity } } } } }"""
        nodes = self._gql(q, {"q": f"name:{num}"})["orders"]["nodes"]
        if not nodes or (nodes[0].get("email") or "").lower() != (email or "").strip().lower():
            return None
        o = nodes[0]
        f = (o.get("fulfillments") or [{}])[0] if o.get("fulfillments") else {}
        t = (f.get("trackingInfo") or [None])[0] if f else None
        return {
            "name": o["name"], "email": o["email"], "created_at": (o.get("processedAt") or o["createdAt"])[:10],
            "financial_status": o["displayFinancialStatus"], "fulfillment_status": o["displayFulfillmentStatus"],
            "delivered": bool(f.get("deliveredAt")), "delivered_at": (f.get("deliveredAt") or "")[:10] or None,
            "tracking": t, "shipping_country": (o.get("shippingAddress") or {}).get("countryCodeV2"),
            "items": [{"title": li["title"], "variant": li.get("variantTitle"), "quantity": li["quantity"]}
                      for li in o["lineItems"]["nodes"]],
        }


def make_shop(cfg) -> Shop:
    if cfg.shop_backend == "shopify":
        return ShopifyShop(cfg.shopify_store_domain, cfg.shopify_admin_token, cfg.shopify_api_version,
                           client_id=cfg.shopify_client_id, client_secret=cfg.shopify_client_secret)
    return MockShop(cfg.mock_store_path)
