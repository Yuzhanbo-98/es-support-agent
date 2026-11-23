"""从 data/mock_store.json 生成两样东西，方便在 Shopify 开发者商店里做对比测试：

1. shopify_products.csv —— 用 Shopify 后台「产品 > 导入」导入，得到和模拟数据一样的 5 个商品
2. inbox_test_sheet.csv —— 32 条西语测试问题 + 正确答案要点，
   用来手动测试 Shopify 自带的 Inbox「AI sales associate」，和我们的 Agent 对比

用法：python tools_scripts/make_test_store_files.py
"""
from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test_store"


def body_html(p: dict) -> str:
    parts = [f"<p>{html.escape(p['description'])}</p>",
             f"<p><strong>Material:</strong> {html.escape(p.get('material', ''))}</p>",
             f"<p><strong>Care:</strong> {html.escape(p.get('care', ''))}</p>",
             f"<p><strong>Ships from:</strong> {html.escape(p.get('shipping_from', ''))}</p>"]
    if p.get("fit_note"):
        parts.append(f"<p><strong>Fit:</strong> {html.escape(p['fit_note'])}</p>")
    chart = p.get("size_chart_cm")
    if chart:
        cols = sorted({k for row in chart.values() for k in row})
        head = "".join(f"<th>{c} (cm)</th>" for c in cols)
        rows = "".join("<tr><td>" + s + "</td>" + "".join(f"<td>{chart[s].get(c, '')}</td>" for c in cols) + "</tr>"
                       for s in chart)
        parts.append(f"<table><tr><th>Size</th>{head}</tr>{rows}</table>")
    return "".join(parts)


def handle(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def products_csv(data: dict) -> None:
    cols = ["Handle", "Title", "Body (HTML)", "Vendor", "Type", "Tags", "Published",
            "Option1 Name", "Option1 Value", "Option2 Name", "Option2 Value",
            "Variant SKU", "Variant Inventory Tracker", "Variant Inventory Qty", "Variant Inventory Policy",
            "Variant Fulfillment Service", "Variant Price", "Variant Requires Shipping", "Variant Taxable", "Status"]
    with open(OUT / "shopify_products.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in data["products"]:
            opt_names = list(p["variants"][0]["options"].keys())
            for i, v in enumerate(p["variants"]):
                row = {"Handle": handle(p["title"]), "Variant SKU": f"{p['id']}-{v['id']}",
                       "Variant Inventory Tracker": "shopify", "Variant Inventory Qty": v["inventory"],
                       "Variant Inventory Policy": "deny", "Variant Fulfillment Service": "manual",
                       "Variant Price": v["price"], "Variant Requires Shipping": "TRUE", "Variant Taxable": "TRUE"}
                for n, name in enumerate(opt_names[:2], start=1):
                    row[f"Option{n} Value"] = v["options"][name]
                    if i == 0:
                        row[f"Option{n} Name"] = name
                if i == 0:
                    row.update({"Title": p["title"], "Body (HTML)": body_html(p), "Vendor": data["shop"]["name"],
                                "Type": p.get("product_type", ""), "Tags": ", ".join(p.get("tags", [])),
                                "Published": "TRUE", "Status": "active"})
                w.writerow(row)


def policies_txt(data: dict) -> None:
    text = "\n\n".join(f"== {k.upper()} ==\n{v}" for k, v in data["policies"].items())
    (OUT / "store_policies.txt").write_text(
        "把下面的内容分别粘贴到 Shopify 后台「设置 > 政策」（退款政策、配送政策等）：\n\n" + text, encoding="utf-8")


def inbox_sheet() -> None:
    cases = [json.loads(l) for l in (ROOT / "evals" / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(OUT / "inbox_test_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "西语变体", "依次发送的消息", "正确回答必须包含（任一）", "不能出现",
                    "必须做的动作", "我们 Agent 的回答", "我们是否通过", "Shopify Inbox AI 的回答", "Inbox 是否通过",
                    "备注（称谓 tú/usted/vos 是否对、是否编造）"])
        for c in cases:
            e = c["expect"]
            w.writerow([c["id"], c["variant"], " ⏎ ".join(c["turns"]),
                        " 且 ".join("/".join(g) for g in e.get("include_any", [])),
                        "/".join(e.get("must_not_include", [])),
                        ", ".join(e.get("must_call", [])), "", "", "", "", ""])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = json.loads((ROOT / "data" / "mock_store.json").read_text(encoding="utf-8"))
    products_csv(data)
    policies_txt(data)
    inbox_sheet()
    print(f"已生成：{OUT}/shopify_products.csv, store_policies.txt, inbox_test_sheet.csv")


if __name__ == "__main__":
    main()
