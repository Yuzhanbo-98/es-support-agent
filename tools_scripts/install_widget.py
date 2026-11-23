"""把聊天插件装进商店当前主题：在 layout/theme.liquid 的 </body> 前插入一行 <script>。

需要应用有 read_themes, write_themes 权限。重复运行不会重复插入。
用法：python -m tools_scripts.install_widget https://es-support-agent-xxxx.onrender.com
      python -m tools_scripts.install_widget --remove
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from es_agent.config import Config  # noqa: E402
from es_agent.shop import ShopifyShop  # noqa: E402

MARK_START, MARK_END = "<!-- es-agent widget -->", "<!-- /es-agent widget -->"
LAYOUT = "layout/theme.liquid"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("host", nargs="?", help="Render 服务地址")
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args(argv)
    if not args.remove and not args.host:
        ap.error("需要服务地址")

    cfg = Config.from_env()
    shop = ShopifyShop(cfg.shopify_store_domain, cfg.shopify_admin_token, cfg.shopify_api_version,
                       client_id=cfg.shopify_client_id, client_secret=cfg.shopify_client_secret)
    theme = shop._gql("{ themes(first: 1, roles: [MAIN]) { nodes { id name } } }")["themes"]["nodes"][0]
    q = """query($id: ID!, $f: [String!]) { theme(id: $id) { files(filenames: $f) { nodes {
             filename body { ... on OnlineStoreThemeFileBodyText { content } } } } } }"""
    content = shop._gql(q, {"id": theme["id"], "f": [LAYOUT]})["theme"]["files"]["nodes"][0]["body"]["content"]

    content = re.sub(re.escape(MARK_START) + ".*?" + re.escape(MARK_END) + r"\n?", "", content, flags=re.S)
    if not args.remove:
        host = args.host.rstrip("/")
        if not host.startswith("https://"):
            host = "https://" + host.removeprefix("http://")
        tag = f'{MARK_START}<script src="{host}/widget.js" defer></script>{MARK_END}\n'
        if "</body>" not in content:
            raise SystemExit("theme.liquid 里找不到 </body>")
        content = content.replace("</body>", tag + "</body>", 1)

    m = """mutation($id: ID!, $files: [OnlineStoreThemeFilesUpsertFileInput!]!) {
             themeFilesUpsert(themeId: $id, files: $files) { upsertedThemeFiles { filename } userErrors { field message } } }"""
    r = shop._gql(m, {"id": theme["id"], "files": [{"filename": LAYOUT, "body": {"type": "TEXT", "value": content}}]})
    errs = r["themeFilesUpsert"]["userErrors"]
    if errs:
        raise SystemExit(f"写入失败：{errs}")
    print(f"{'已移除' if args.remove else '已安装'}聊天插件：主题「{theme['name']}」")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
