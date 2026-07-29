# Streamlit Community Cloud 上线指南

目标：获得一个可分享的固定网址：

```text
https://你的项目名.streamlit.app
```

访客点开网址即可查看页面，无需在自己电脑安装 Python。

## 展示版的权限

- 所有访客都可以搜索新闻、查看新闻池和案例库、打开原文及导出结果。
- 点击右上角 `⚙️` 可直接进入管理台，审核案例或修改研究主题，无需密码。

> 这种开放模式适合内部使用或快速演示。公开部署后，任何访客都可能消耗
> AI API 配额或修改配置；正式对外使用时，请在部署层限制访问。

## 1. 创建 GitHub 仓库

建议在 GitHub 创建一个 **Private repository**，避免研究主题 Excel 在
GitHub 中公开。Streamlit 部署后仍可将网页设置为公开。

在 GitHub 中创建空仓库，不要勾选自动创建 README。保留仓库地址：

```text
https://github.com/<your-name>/<your-repository>.git
```

## 2. 上传当前文件夹

打开终端，逐条执行：

```bash
cd "/Users/mr.jiang/Desktop/新闻数据爬取网页"
git init
git add .
git status --short
```

检查 `git status --short`：

应该包含：

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `.streamlit/secrets.toml.example`
- `复星旅文DeepResearch研究主题目录_更新版_V4.xlsx`

绝对不应包含：

- `.env`
- `.streamlit/secrets.toml`
- `.venv/`
- `data/industry_news.db`
- 任何真实 API Key 或口令

确认后继续：

```bash
git commit -m "Prepare Streamlit Community Cloud demo"
git branch -M main
git remote add origin https://github.com/<your-name>/<your-repository>.git
git push -u origin main
```

将上面的 GitHub 地址替换为你自己的仓库地址。

## 3. 创建 Streamlit Cloud 应用

1. 打开 [share.streamlit.io](https://share.streamlit.io/)。
2. 使用 GitHub 账号登录并授权访问刚创建的仓库。
3. 点击 **Create app**。
4. 选择仓库和 `main` 分支。
5. **Main file path** 填写 `app.py`。
6. 点击 **Advanced settings**。
7. Python 版本选择 **3.12**。

## 4. 配置 Secrets

在 Advanced settings 的 **Secrets** 输入框中粘贴：

```toml
AI_PROVIDER = "openai"
OPENAI_API_KEY = "你自己的_OpenAI_API_Key"
OPENAI_MODEL = "gpt-5.6-luna"
DEPLOYMENT_MODE = "cloud_demo"
DATABASE_PATH = "data/industry_news.db"
TOPICS_XLSX_PATH = "复星旅文DeepResearch研究主题目录_更新版_V4.xlsx"
SCHEDULER_POLL_MINUTES = "30"
MAX_TOPICS_PER_CYCLE = "2"
TZ = "Asia/Shanghai"
```

请直接在 Streamlit Cloud 页面填写真实值，不要把真实值发到聊天、
写入代码或提交到 GitHub。

## 5. 部署和公开网址

1. 保存 Advanced settings。
2. 点击 **Deploy**。
3. 等待 Streamlit 安装 `requirements.txt` 中的依赖。
4. 部署完成后，复制页面的 `*.streamlit.app` 网址。
5. 打开应用的 **Share / App settings**，将查看权限设置为公开。

此后别人只需打开这个网址，就能直接看到网页。

## 6. 首次使用

Cloud 的数据库初始为空：

1. 打开应用网址。
2. 在搜索框输入一个关键词。
3. 展开「搜索设置」，将文章数量先设为 2–3 篇。
4. 点击「搜索并分析」。
5. 确认新闻池中的原文链接可点击。
6. 点击右上角 `⚙️`，确认管理台可以打开。

数据产生后，公开访客即可查看已有结果。

## 7. 更新网页

本地修改代码后：

```bash
git add .
git commit -m "Update Streamlit app"
git push
```

Community Cloud 会检测 GitHub 变更并重新部署。

## 展示版限制

- Community Cloud 不运行项目的独立 `worker.py`，只支持访客手动搜索。
- SQLite 和导出数据是云端临时文件，应用重启或重新部署后可能重置。
- 有价值的案例应及时通过「导出当前结果」保存到本地。
- 这一版不会在 Cloud 中持久化修改原始主题 Excel。
- 后续正式版需要独立服务器或外部数据库/云盘。

## 常见问题

### 页面能打开，但抓取提示 API Key 错误

进入 Streamlit App settings，检查 Secrets 中的 `AI_PROVIDER`、
`OPENAI_API_KEY` 和 `OPENAI_MODEL`。如果选择 Gemini，则检查对应的
`GEMINI_API_KEY` 和 `GEMINI_MODEL`。

### 点击搜索后提示 API 配额或权限错误

检查 Secrets 中的 API Key 是否有效、账号是否还有可用额度。开放管理台不代表
AI 服务免费，每次搜索仍会消耗所配置账号的 API 配额。

### 部署时提示缺少模块

检查 `requirements.txt` 是否位于仓库根目录，并在 Streamlit 日志中查看缺少的包名。
