# 西语客服 Agent · MVP

这是给 Shopify 上中国卖家用的 AI 客服。顾客用西班牙语（包括西班牙、拉美的说法和英西混用）提问时，Agent 会实时读取店铺数据来回答，内容覆盖商品细节、尺码、库存、物流、退换货政策和订单状态。退款、退货、取消订单这类请求不会自动执行，而是生成工单，卖家在中文后台审批后才处理。

整个项目是手写的：不用 LangChain 这类框架，也不用任何厂商的 SDK。调模型只用 httpx 发 HTTP 请求。

## 第一步：在你自己电脑上用真实模型跑评测（约 20 分钟）

1. 装 Python 3.11 或更高版本，把这个文件夹解压到任意位置，在文件夹里打开终端：
   ```bash
   python -m venv .venv
   # Mac / Linux：
   source .venv/bin/activate
   # Windows（PowerShell）：
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m pytest -q          # 应该显示 20 passed
   ```
2. 申请至少一个模型的 API key（DeepSeek 最便宜；Gemini 有免费额度），在文件夹里新建 `.env`，写一行，例如：
   ```
   DEEPSEEK_API_KEY=你的key
   ```
   key 只放在你自己电脑的 `.env` 里，不要发到聊天或别处。
3. 复制 `evals/models.example.json` 为 `evals/models.json`，删掉你没有 key 的模型。**单价是示例，跑之前按官网当前价格改**，否则成本数字不准。
4. 运行：
   ```bash
   python -m evals.compare
   ```
   终端会打印每条用例的通过情况和一张对比表，同时在 `evals/results/` 里写出 `compare_时间.md` 和每个模型的详细 JSON。32 条用例用便宜模型跑一遍，费用一般不到 1 美元。
5. 把 `evals/results/` 文件夹发回给我，我来分析失败用例、修 Agent，再一起复测。

**顺便和 Shopify 自带 AI 对比**（可选）：运行 `python tools_scripts/make_test_store_files.py`，会在 `test_store/` 生成
`shopify_products.csv`（在开发者商店「产品 > 导入」）、`store_policies.txt`（粘贴到「设置 > 政策」）和 `inbox_test_sheet.csv`（32 条西语问题，逐条发给 Shopify Inbox 的 AI sales associate，把回答填进表里）。

## 目录

```
es_agent/
  llm.py      模型适配层：OpenAI 兼容接口 + Anthropic 接口，含工具调用格式互转、重试
  agent.py    Agent 循环 + 系统提示（只按店铺数据回答、称谓跟随顾客、不承诺退款）
  tools.py    6 个工具：搜商品、商品详情、店铺政策、查订单、提交退换货申请、转人工
  shop.py     店铺数据：MockShop（模拟数据）/ ShopifyShop（Admin GraphQL API）
  store.py    SQLite：会话、消息、工具调用日志、工单
  server.py   FastAPI：聊天接口、演示店铺页、中文卖家后台
data/mock_store.json   模拟的中国卖家店铺（5 个商品、政策、3 个订单）
evals/cases.jsonl      32 条评测用例（es-ES / es-MX / es-AR / es-CO / Spanglish / 攻击类）
evals/run_evals.py     评测脚本（规则检查 + 可选的模型评审）
tests/                 19 个单元测试（不需要 API key）
```

## 跑起来

```bash
pip install -r requirements.txt
cp .env.example .env        # 填模型地址、key、模型名
python -m pytest -q         # 单元测试，不调模型
uvicorn es_agent.server:app --reload
```

- 顾客视角的演示店铺：http://localhost:8000/ ，点右下角聊天
- 卖家中文后台：http://localhost:8000/admin （用户名随便填，密码是 `.env` 里的 `ADMIN_PASSWORD`）

可以试这几句：
- `Hola, mido 92 de pecho, ¿qué talla del vestido de lino me recomendáis?`
- `¿Dónde está mi pedido #1001? Mi email es lucia@example.com`
- `Las zapatillas del pedido #1003 llegaron rotas, quiero mi dinero. Email: marta@example.com`，发完去后台看工单，点“同意”

## 换模型

只改 `.env`：

| 想用 | LLM_PROVIDER | LLM_BASE_URL | LLM_MODEL 举例 |
|---|---|---|---|
| DeepSeek | openai | https://api.deepseek.com/v1 | deepseek-chat |
| OpenAI | openai | https://api.openai.com/v1 | gpt-4o-mini |
| 通义千问 | openai | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus |
| 本地 Ollama | openai | http://localhost:11434/v1 | qwen2.5:14b |
| Anthropic | anthropic | https://api.anthropic.com | （填你要用的模型 ID） |

模型必须支持工具调用（function calling）。

## 跑评测

```bash
python -m evals.run_evals                  # 全部 32 条，规则检查
python -m evals.run_evals --judge          # 再让模型评审：有没有编造、语气和称谓
python -m evals.run_evals --only size_dress_es order_status_ok
```

输出会按西语变体分组统计通过率，并写出 `eval_report.json`，里面有每条用例的回复和工具调用记录。换一个模型再跑一遍，就能对比效果和 token 成本。

## 接真实的 Shopify 开发者商店

1. 在 dev.shopify.com 的「商店」里创建开发商店（必须在这里建，应用和商店要属于同一个组织）。
2. 商店后台「产品 > 导入」导入 `test_store/shopify_products.csv`，「设置 > 常规」把商店货币改成 EUR。
3. dev.shopify.com 的「应用」里创建应用，取消“嵌入后台”，权限范围填：
   `read_products,read_inventory,read_orders,read_legal_policies,write_legal_policies,write_draft_orders,write_orders,write_merchant_managed_fulfillment_orders`
   （write 权限只给下面的初始化脚本用，客服 Agent 只读）。发布版本，再把应用装到商店。
4. 应用「设置」里复制 Client ID 和 Client secret，填进 `.env`：
   `SHOP_BACKEND=shopify`、`SHOPIFY_STORE_DOMAIN`、`SHOPIFY_CLIENT_ID`、`SHOPIFY_CLIENT_SECRET`。
   新版应用没有永久 token，程序用 client credentials 自动换 24 小时有效的 token 并自动续期。旧版自定义应用的永久 token 填 `SHOPIFY_ADMIN_TOKEN` 也行。
5. `python -m tools_scripts.seed_shopify` 写入店铺政策和测试订单 #1001-#1003（和模拟数据一致）。
6. 评测：`SHOP_BACKEND=shopify python -m evals.compare --judge` 就会对着真实商店跑；`SHOP_BACKEND=mock` 用模拟数据，结果可复现。

## 挂到 Shopify 商店前台（部署到 Render）

1. 把代码推到 GitHub。
2. render.com 用 GitHub 登录 → **New → Blueprint** → 选这个仓库。Render 会读 `render.yaml`，
   让你填 4 个密钥：`LLM_API_KEY`、`SHOPIFY_CLIENT_ID`、`SHOPIFY_CLIENT_SECRET`、`ADMIN_PASSWORD`。
3. 部署好后得到 `https://xxx.onrender.com`。打开 `/healthz` 应显示 `{"ok":true}`，`/admin` 是卖家后台。
4. Shopify 后台 **在线商店 → 主题 → 自定义** → 页脚里 **添加区块 → 自定义 Liquid**，粘贴：
   `<script src="https://xxx.onrender.com/widget.js" defer></script>`，保存。前台右下角就会出现聊天按钮。

说明：聊天接口只接受 `*.myshopify.com` 的跨域请求（自定义域名加到 `WIDGET_ALLOWED_ORIGINS`），
每个 IP 每 10 分钟最多 30 条消息（`RATE_LIMIT_PER_10MIN`）。免费套餐闲置 15 分钟会休眠，SQLite 数据重启后清空，只适合演示。

## MVP 还没做的（见产品文档里的路线图）

- 做成正式的 Shopify App（OAuth 安装、主题 App 扩展挂聊天窗口、App 计费）
- 卖家同意退款后，自动调 Shopify 的退款接口（现在只改工单状态，并通知顾客）
- 邮件 / WhatsApp 渠道、流式输出、商品数据向量检索、多店铺
