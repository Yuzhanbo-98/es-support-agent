"""多模型对比：用同一套 32 条评测跑几个模型，输出通过率、按西语变体的通过率、token 和估算成本。

用法：
  1. 复制 evals/models.example.json 为 evals/models.json，按需删改模型
  2. 在 .env 里填对应的 key（变量名见 models.json 的 api_key_env）
  3. python -m evals.compare                 # 跑 models.json 里的全部模型
     python -m evals.compare --models deepseek-chat qwen-plus
     python -m evals.compare --judge         # 再加模型评审（更慢、更贵）

结果：终端打印对比表，并写出 evals/results/compare_<时间>.md 和每个模型的 JSON 明细。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from es_agent.config import Config
from es_agent.llm import AnthropicLLM, OpenAICompatLLM

from .run_evals import CASES, judge, run_case

HERE = Path(__file__).parent


def build_llm(m: dict):
    key = os.environ.get(m.get("api_key_env", ""), "")
    if not key:
        raise SystemExit(f"缺少环境变量 {m.get('api_key_env')}（模型 {m['name']}）。请在 .env 里填好。")
    if m["provider"] == "anthropic":
        return AnthropicLLM(m["base_url"], key, m["model"])
    return OpenAICompatLLM(m["base_url"], key, m["model"])


def cost_usd(m: dict, tin: int, tout: int) -> float:
    return tin / 1e6 * m.get("usd_per_m_input", 0) + tout / 1e6 * m.get("usd_per_m_output", 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "models.json"))
    ap.add_argument("--models", nargs="*", help="只跑这些 name")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.from_env()  # 也会读取 .env
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"找不到 {cfg_path}。先复制 evals/models.example.json 为 evals/models.json。")
    models = json.loads(cfg_path.read_text(encoding="utf-8"))
    if args.models:
        models = [m for m in models if m["name"] in set(args.models)]
    cases = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    summary = []
    for m in models:
        llm = build_llm(m)
        print(f"\n=== {m['name']} ({m['model']}) ===")
        results = []
        for c in cases:
            try:
                r = run_case(llm, c, cfg)
            except Exception as e:  # 单条失败不影响其他用例
                r = {"id": c["id"], "variant": c["variant"], "reply": "", "called": [], "trace": [],
                     "failures": [f"运行出错：{e}"], "usage": {}, "seconds": 0}
            if args.judge and r["reply"]:
                r["judge"] = judge(llm, c, r["reply"], r["trace"])
                j = r["judge"]
                if isinstance(j.get("grounded"), int) and j["grounded"] < 4:
                    r["failures"].append(f"评审：可能编造（grounded={j['grounded']}）")
                if isinstance(j.get("tone"), int) and j["tone"] < 4:
                    r["failures"].append(f"评审：语气/称谓不合适（tone={j['tone']}）")
            ok = not r["failures"]
            print(f"  {'✓' if ok else '✗'} {r['id']:<28} {'' if ok else '; '.join(r['failures'])}")
            results.append(r)

        by_var = defaultdict(lambda: [0, 0])
        for r in results:
            by_var[r["variant"]][1] += 1
            by_var[r["variant"]][0] += not r["failures"]
        tin = sum(r["usage"].get("input_tokens", 0) for r in results)
        tout = sum(r["usage"].get("output_tokens", 0) for r in results)
        conversations = len(results)
        total_cost = cost_usd(m, tin, tout)
        row = {
            "name": m["name"], "model": m["model"],
            "passed": sum(not r["failures"] for r in results), "total": len(results),
            "by_variant": {k: f"{p}/{n}" for k, (p, n) in sorted(by_var.items())},
            "input_tokens": tin, "output_tokens": tout,
            "usd_total": round(total_cost, 4),
            "usd_per_conversation": round(total_cost / max(conversations, 1), 5),
            "avg_seconds": round(sum(r["seconds"] for r in results) / max(conversations, 1), 2),
        }
        summary.append(row)
        (out_dir / f"{m['name']}_{stamp}.json").write_text(
            json.dumps({"summary": row, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    variants = sorted({v for s in summary for v in s["by_variant"]})
    lines = ["| 模型 | 通过率 | " + " | ".join(variants) + " | 每次对话成本 | 平均耗时 |",
             "|---|---|" + "---|" * len(variants) + "---|---|"]
    for s in summary:
        lines.append(f"| {s['name']} | {s['passed']}/{s['total']} ({s['passed'] / max(s['total'], 1):.0%}) | "
                     + " | ".join(s["by_variant"].get(v, "-") for v in variants)
                     + f" | ${s['usd_per_conversation']:.4f} | {s['avg_seconds']}s |")
    table = "\n".join(lines)
    print("\n" + table)
    report = out_dir / f"compare_{stamp}.md"
    report.write_text(f"# 模型对比 {stamp}\n\n{table}\n\n成本按 models.json 里填的单价估算。\n", encoding="utf-8")
    print(f"\n已写出 {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
