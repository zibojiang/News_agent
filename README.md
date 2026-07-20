# 文旅行业情报 Agent

一个本地运行的 Streamlit 网页，按研究主题抓取真实新闻，
默认使用 OpenAI 提炼带证据的量化案例，也可切换到 Gemini，
分析结果写入 SQLite 新闻/案例数据库。

## Streamlit Community Cloud 快速展示

如果需要一个别人点开即可查看的 `*.streamlit.app` 网址，请按
[STREAMLIT_CLOUD.md](STREAMLIT_CLOUD.md) 操作。

Cloud 展示版默认公开只读：所有人可以查看新闻、原文链接和案例，
只有输入管理员口令后才能抓取、审核和修改主题。

## 第一版功能

- 首次启动时从项目内的《复星旅文 DeepResearch 研究主题目录》导入 28 个主题。
- 按主题手动搜索 Google News / Bing News RSS，输出真实原文链接。
- 提取新闻正文，记录媒体来源、发布时间和正文指纹。
- OpenAI/Gemini 输出摘要、量化案例、证据原文、企业、地区、指标标签和相关性。
- 低分结果保留在「新闻池」；达到门槛且有量化内容的结果进入「案例库」。
- 同一主题按 URL 和正文 SHA-256 指纹去重。
- 案例可标记为待审核、已确认或已忽略，并可导出 Excel。
- 主题的搜索词、启用状态、分数门槛和采集间隔可在网页修改。
- 独立 worker 可按主题周期采集，网页中可查看心跳和任务日志。

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

打开 `.env`，填写 OpenAI 配置：

```env
AI_PROVIDER=openai
OPENAI_API_KEY=你的_OpenAI_API_Key
OPENAI_MODEL=gpt-5.6-luna
```

如需继续使用 Gemini，把 `AI_PROVIDER` 改为 `gemini`，并填写
`GEMINI_API_KEY` 和 `GEMINI_MODEL`。
不要将 `.env` 提交或发送给其他人。

## 启动网页

```bash
source .venv/bin/activate
streamlit run app.py
```

默认数据库为 `data/industry_news.db`。原始研究主题 Excel 只会读取，
网页中的主题调整只写入 SQLite，不会改动 Excel。

## 启动周期采集

在第二个终端中执行：

```bash
cd "/Users/mr.jiang/Desktop/新闻数据爬取网页"
source .venv/bin/activate
python3 worker.py
```

worker 默认每 30 分钟检查一次到期主题：

- 每轮默认最多处理 2 个主题，避免集中消耗 AI API 配额。
- worker 启动后不立即抓取，会在第一个扫描周期到达时开始。
- 「实时」和「事件驱动」主题：默认每 4 小时采集。
- 其他主题：默认每 24 小时采集。

报告频率仍保留 Excel 中的实时、周报、月报、季度等业务定义，
与底层新闻采集间隔分开管理。

> worker 是本地进程。终端关闭或电脑关机后不会继续采集。

## 页面说明

- **仪表盘**：新闻数、达标案例数、待审核数和主题覆盖。
- **新闻池**：所有已完成 Agent 分析的新闻，包含低分结果。
- **案例库**：达到入库门槛的量化案例，支持审核和 Excel 导出。
- **主题管理**：管理 28 个研究主题的采集参数。
- **任务中心**：查看 worker 状态和最近 100 次运行日志。

## 本地测试

```bash
python3 -m unittest discover -s tests -v
```

测试使用临时 SQLite 数据库，不请求新闻站点，也不调用真实 AI API。

## 当前边界

- 仅处理公开可访问页面，不绕过登录、付费墙或反爬验证。
- JavaScript 渲染页面可能无法提取正文。
- Agent 输出必须经人工审核，特别是金额、比例和人事信息。
- 第一版不自动生成周报/月报正文，只完成连续采集与案例积累。
