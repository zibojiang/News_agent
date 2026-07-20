"""用一次最小请求验证项目中的 Gemini API 配置。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai


def main() -> int:
    env_path = Path(__file__).resolve().with_name(".env")
    load_dotenv(env_path)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ 未配置有效的 GEMINI_API_KEY，请先检查项目中的 .env 文件。")
        return 1

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    print(f"正在测试 Gemini API,模型：{model}")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents="只回复：API测试成功",
        )
    except Exception as exc:
        print(f"❌ API 调用失败：{type(exc).__name__}: {exc}")
        return 1

    response_text = (response.text or "").strip()
    if not response_text:
        print("❌ API 已响应，但没有返回文本。")
        return 1

    print(f"✅ Gemini API 可用，返回内容：{response_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
