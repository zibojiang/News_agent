"""
scraper.py — 网页抓取模块

负责从行业资讯源获取最新新闻列表，并提取单篇文章的纯净正文文本。
当前实现基于 Google News RSS + Requests/BeautifulSoup，
后续可扩展 Playwright 以应对 JavaScript 渲染页面。
"""

from __future__ import annotations

import logging
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder

logger = logging.getLogger(__name__)

# 通用 HTTP 请求头，模拟浏览器访问
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 网络请求超时（秒）
REQUEST_TIMEOUT = 30

# 单次抓取最大文章数（避免 API 调用过多）
MAX_ARTICLES_PER_RUN = 8


def _safe_get(url: str, timeout: int = REQUEST_TIMEOUT) -> requests.Response | None:
    """
    带异常捕获的 GET 请求封装。

    Args:
        url: 目标 URL
        timeout: 超时秒数

    Returns:
        Response 对象；失败时返回 None
    """
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.Timeout:
        logger.warning("请求超时: %s", url)
    except requests.HTTPError as exc:
        logger.warning("HTTP 错误 %s: %s", exc.response.status_code if exc.response else "?", url)
    except requests.RequestException as exc:
        logger.warning("网络请求失败: %s — %s", url, exc)
    return None


def _parse_rss_items(xml_content: bytes, max_articles: int) -> list[dict[str, Any]]:
    """
    解析 RSS 2.0 XML，提取新闻条目。

    Args:
        xml_content: RSS 原始字节内容
        max_articles: 最多返回条数

    Returns:
        新闻列表，每项包含 title / url / published_at
    """
    articles: list[dict[str, Any]] = []
    root = ET.fromstring(xml_content)

    for item in root.findall(".//item")[:max_articles]:
        title_el = item.find("title")
        link_el = item.find("link")
        pub_el = item.find("pubDate")
        source_el = item.find("source")

        title = (title_el.text or "").strip() if title_el is not None else ""
        url = (link_el.text or "").strip() if link_el is not None else ""
        published_at = _parse_rss_pub_date(pub_el.text if pub_el is not None else "")
        source = (source_el.text or "").strip() if source_el is not None else ""

        if title and url:
            articles.append(
                {
                    "title": title,
                    "url": url,
                    "published_at": published_at,
                    "source": source,
                }
            )

    return articles


def _resolve_final_url(url: str) -> str:
    """
    解析跳转链接，获取媒体原文的真实 URL。

    - Google News RSS 链接：通过 googlenewsdecoder 解码为媒体原文地址
    - 其他链接：跟随 HTTP 重定向获取最终 URL

    Args:
        url: 原始链接（可能为 Google News 中转页）

    Returns:
        媒体原文 URL；解析失败时返回原始 URL
    """
    if "news.google.com" in url:
        try:
            result = gnewsdecoder(url)
            if result.get("status") and result.get("decoded_url"):
                decoded = result["decoded_url"]
                logger.debug("Google News 链接已解码: %s -> %s", url[:60], decoded)
                return decoded
            logger.warning("Google News 链接解码失败: %s", result)
        except Exception as exc:
            logger.warning("Google News 解码异常: %s — %s", url[:60], exc)

    # 非 Google News 链接，或解码失败时，尝试 HTTP 重定向
    try:
        response = requests.head(
            url,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        final_url = response.url
        if final_url and final_url != url:
            logger.debug("链接已重定向: %s -> %s", url[:60], final_url[:60])
        return final_url or url
    except requests.RequestException:
        try:
            response = requests.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                stream=True,
            )
            response.close()
            return response.url or url
        except requests.RequestException as exc:
            logger.warning("无法解析链接，使用原始 URL: %s — %s", url[:60], exc)
            return url


def _get_soup(html: str) -> BeautifulSoup:
    """优先使用 lxml 解析 HTML，不可用时回退到内置 html.parser。"""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _fetch_rss_news(rss_url: str, max_articles: int, source_name: str) -> list[dict[str, Any]]:
    """
    从指定 RSS 源抓取真实新闻列表。

    Args:
        rss_url: RSS 订阅地址
        max_articles: 最多返回条数
        source_name: 数据源名称（用于日志）

    Returns:
        新闻列表；抓取或解析失败时返回空列表
    """
    logger.info("尝试从 %s 抓取: %s", source_name, rss_url)
    response = _safe_get(rss_url)
    if response is None:
        return []

    try:
        articles = _parse_rss_items(response.content, max_articles)
        logger.info("%s 返回 %d 条新闻", source_name, len(articles))
        return articles
    except ET.ParseError as exc:
        logger.error("%s RSS 解析失败: %s", source_name, exc)
        return []


def _parse_rss_pub_date(pub_date: str) -> str:
    """
    尝试将 RSS 发布时间解析为统一格式字符串。

    Args:
        pub_date: RSS item 中的 pubDate 字段

    Returns:
        格式化后的时间字符串
    """
    if not pub_date:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 常见 RSS 日期格式示例: "Mon, 13 Jul 2026 08:00:00 GMT"
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(pub_date.strip(), fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    return pub_date.strip()


def fetch_latest_news(query: str, max_articles: int = MAX_ARTICLES_PER_RUN) -> list[dict[str, Any]]:
    """
    根据行业关键词抓取最新真实新闻列表。

    数据源（按优先级）：
    1. Google News RSS（中文）
    2. Bing News RSS（备用，同为真实新闻源）

    所有数据源均来自公开 RSS，不包含任何模拟/虚构数据。
    若全部源均不可用，返回空列表。

    Args:
        query: 行业关键词，如「新能源」「人工智能」
        max_articles: 最多返回的文章条数

    Returns:
        新闻列表，每项包含 title / url / published_at

    Raises:
        RuntimeError: 所有真实新闻源均无法获取数据时抛出
    """
    encoded_query = quote_plus(query)
    logger.info("开始抓取真实新闻列表，关键词: %s", query)

    # 真实新闻 RSS 源（按优先级排列）
    rss_sources = [
        (
            "Google News",
            f"https://news.google.com/rss/search?"
            f"q={encoded_query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        ),
        (
            "Bing News",
            f"https://www.bing.com/news/search?q={encoded_query}&format=rss",
        ),
    ]

    articles: list[dict[str, Any]] = []
    errors: list[str] = []

    for source_name, rss_url in rss_sources:
        fetched = _fetch_rss_news(rss_url, max_articles, source_name)
        if fetched:
            articles = fetched
            break
        errors.append(f"{source_name} 无有效结果")

    if not articles:
        error_msg = f"无法从任何真实新闻源获取数据（关键词: {query}）: {'; '.join(errors)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 解析 Google News 等跳转链接，替换为媒体原文 URL
    for article in articles:
        original_url = article["url"]
        resolved_url = _resolve_final_url(original_url)
        article["url"] = resolved_url

    logger.info("共获取 %d 条真实新闻", len(articles))
    return articles[:max_articles]


def _clean_text(text: str) -> str:
    """去除多余空白字符，合并连续换行。"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_article_text(url: str, max_chars: int = 12000) -> str:
    """
    抓取指定 URL 并提取去除 HTML 标签后的纯净正文。

    提取策略（按优先级）：
    1. <article> 标签内的段落
    2. 常见正文容器 class/id（article-content, post-content 等）
    3. 全页 <p> 标签拼接

    Args:
        url: 文章网页链接
        max_chars: 返回文本最大字符数（控制 Gemini Token 消耗）

    Returns:
        清洗后的正文文本；抓取或解析失败时返回空字符串
    """
    logger.info("开始提取正文: %s", url)

    # 先解析跳转链接，确保抓取的是媒体原文页面
    url = _resolve_final_url(url)

    response = _safe_get(url)
    if response is None:
        return ""

    # 尝试自动检测编码
    response.encoding = response.apparent_encoding or "utf-8"
    soup = _get_soup(response.text)

    # 移除脚本、样式、导航等噪声节点
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    paragraphs: list[str] = []

    # 策略 1：<article> 标签
    article_tag = soup.find("article")
    if article_tag:
        paragraphs = [
            _clean_text(p.get_text())
            for p in article_tag.find_all("p")
            if len(_clean_text(p.get_text())) > 20
        ]

    # 策略 2：常见正文容器
    if not paragraphs:
        content_selectors = [
            {"class_": re.compile(r"(article|content|post|news|detail|text)", re.I)},
            {"id": re.compile(r"(article|content|post|news|detail|text)", re.I)},
        ]
        for selector in content_selectors:
            container = soup.find(["div", "section"], **selector)
            if container:
                paragraphs = [
                    _clean_text(p.get_text())
                    for p in container.find_all("p")
                    if len(_clean_text(p.get_text())) > 20
                ]
                if paragraphs:
                    break

    # 策略 3：全页段落兜底
    if not paragraphs:
        paragraphs = [
            _clean_text(p.get_text())
            for p in soup.find_all("p")
            if len(_clean_text(p.get_text())) > 30
        ]

    full_text = "\n".join(paragraphs)

    if not full_text:
        # 最后兜底：取 body 纯文本
        body = soup.find("body")
        full_text = _clean_text(body.get_text()) if body else ""

    if not full_text:
        logger.warning("未能从页面提取有效正文: %s", url)
        return ""

    return full_text[:max_chars]


def fetch_and_extract_batch(
    query: str,
    max_articles: int = MAX_ARTICLES_PER_RUN,
    delay_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """
    组合函数：抓取新闻列表并逐篇提取正文。

    Args:
        query: 行业关键词
        max_articles: 最大文章数
        delay_seconds: 每篇文章之间的请求间隔（礼貌爬取）

    Returns:
        包含 title/url/source/published_at/content/content_hash 的字典列表
    """
    news_list = fetch_latest_news(query, max_articles=max_articles)
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(news_list):
        url = item["url"]
        content = extract_article_text(url)

        if not content:
            logger.warning("跳过无正文文章: %s", item.get("title"))
            continue

        results.append(
            {
                "title": item["title"],
                "url": url,
                "source": item.get("source", ""),
                "published_at": item.get("published_at", ""),
                "content": content,
                "content_hash": hashlib.sha256(
                    _clean_text(content).encode("utf-8")
                ).hexdigest(),
            }
        )

        # 最后一篇无需等待
        if idx < len(news_list) - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    logger.info("成功提取 %d/%d 篇文章正文", len(results), len(news_list))
    return results
