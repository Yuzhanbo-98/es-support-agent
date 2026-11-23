"""从环境变量读取配置。不依赖任何第三方配置库。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """极简 .env 读取：KEY=VALUE，忽略注释和空行，不覆盖已有环境变量。"""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Config:
    llm_provider: str = "openai"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    shop_backend: str = "mock"
    shopify_store_domain: str = ""
    shopify_admin_token: str = ""
    shopify_client_id: str = ""
    shopify_client_secret: str = ""
    shopify_api_version: str = "2025-10"
    mock_store_path: str = str(Path(__file__).resolve().parent.parent / "data" / "mock_store.json")
    db_path: str = "agent.db"
    admin_password: str = "change-me"
    max_agent_steps: int = 6
    widget_allowed_origins: list[str] = field(default_factory=list)
    rate_limit_per_10min: int = 30

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv()
        c = cls()
        env = os.environ.get
        c.llm_provider = env("LLM_PROVIDER", c.llm_provider).lower()
        c.llm_base_url = env("LLM_BASE_URL", c.llm_base_url).rstrip("/")
        c.llm_api_key = env("LLM_API_KEY", c.llm_api_key)
        c.llm_model = env("LLM_MODEL", c.llm_model)
        c.shop_backend = env("SHOP_BACKEND", c.shop_backend).lower()
        c.shopify_store_domain = env("SHOPIFY_STORE_DOMAIN", c.shopify_store_domain)
        c.shopify_admin_token = env("SHOPIFY_ADMIN_TOKEN", c.shopify_admin_token)
        c.shopify_client_id = env("SHOPIFY_CLIENT_ID", c.shopify_client_id)
        c.shopify_client_secret = env("SHOPIFY_CLIENT_SECRET", c.shopify_client_secret)
        c.shopify_api_version = env("SHOPIFY_API_VERSION", c.shopify_api_version)
        c.mock_store_path = env("MOCK_STORE_PATH", c.mock_store_path)
        c.db_path = env("DB_PATH", c.db_path)
        c.admin_password = env("ADMIN_PASSWORD", c.admin_password)
        c.max_agent_steps = int(env("MAX_AGENT_STEPS", str(c.max_agent_steps)))
        c.widget_allowed_origins = [o.strip() for o in env("WIDGET_ALLOWED_ORIGINS", "").split(",") if o.strip()]
        c.rate_limit_per_10min = int(env("RATE_LIMIT_PER_10MIN", str(c.rate_limit_per_10min)))
        return c
