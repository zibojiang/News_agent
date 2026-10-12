# Streamlit Community Cloud 上线指南

目标：获得一个可分享的固定网址：

```text
https://你的项目名.streamlit.app
```

访客点开网址即可查看页面，无需在自己电脑安装 Python。

## 展示版的权限

- 无需登录，每个浏览器会话独立保存搜索记录、新闻和案例；不同访客互相不可见。
- 点击右上角「我的记录 / My history」查看关键词和任务状态、审核自己的案例或导出结果。
- 网页不提供全站管理和主题修改入口；旧共享数据库不会向访客展示。
- 默认英文，右上角「中/ENG」可一键切换语言。

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

每个新浏览器会话的搜索记录初始为空：

1. 打开应用网址。
2. 在搜索框输入一个关键词。
3. 展开「搜索设置」，将文章数量先设为 2–3 篇。
4. 点击「下一步：确认意图」，确认搜索范围后开始搜索。
5. 确认新闻池中的原文链接可点击。
6. 点击右上角「我的记录」，确认能看到本次会话的搜索历史。

新开一个标签页或使用另一个浏览器访问，应得到独立的空白记录，
不会看到其他会话已产生的新闻。相同会话内切换语言和页面会保留当前结果。

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
- 搜索记录只保留在当前浏览器会话；刷新、新开标签页或重新打开页面可能开始新会话。
- 会话使用独立的临时 SQLite；会话释放且后台任务结束后清理，应用重启或重新部署也会丢失记录。
- 有价值的案例应及时通过「导出当前结果」保存到本地。
- 这一版不会在 Cloud 中持久化修改原始主题 Excel。
- 若需跨设备或长期保留历史，需增加登录及按账号隔离的持久存储。

## 常见问题

### 页面能打开，但抓取提示 API Key 错误

进入 Streamlit App settings，检查 Secrets 中的 `AI_PROVIDER`、
`OPENAI_API_KEY` 和 `OPENAI_MODEL`。如果选择 Gemini，则检查对应的
`GEMINI_API_KEY` 和 `GEMINI_MODEL`。

### 点击搜索后提示 API 配额或权限错误

检查 Secrets 中的 API Key 是否有效、账号是否还有可用额度。
无需登录不代表 AI 服务免费，每次搜索仍会消耗所配置账号的 API 配额。

### 部署时提示缺少模块

检查 `requirements.txt` 是否位于仓库根目录，并在 Streamlit 日志中查看缺少的包名。
