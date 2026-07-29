"""
quality_scorer.py — 新闻质量评分模块

独立于 AI 调用的纯算法评分，分两阶段：
1. 预筛阶段（提取正文后、送 AI 前）：来源权威度、关键词匹配、信息密度、数据含量、时效性
2. AI 后增强（AI 分析完成后）：标题正文一致性、原创性信号、报道完整性、透明度

评分规则版本化，支持缓存：相同 content_hash + 相同规则版本不重复评分。
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ============================================================
# 评分规则版本 — 变更时触发重新评分
# ============================================================
NEWS_QUALITY_RULE_VERSION = "v3"

# ============================================================
# 来源权威度配置（独立存放，便于扩展）
# ============================================================
# 一级：专业财经/官方媒体
TIER1_SOURCES: dict[str, int] = {
    "新华社": 25, "新华网": 25, "人民日报": 24, "央视新闻": 24, "央视网": 24,
    "证券时报": 23, "财联社": 23, "第一财经": 23, "界面新闻": 22, "界面": 22,
    "21世纪经济报道": 23, "经济观察报": 22, "每日经济新闻": 22,
    "36氪": 21, "澎湃新闻": 21, "中国证券报": 23, "上海证券报": 23,
    "证券日报": 22, "中国经营报": 21, "中新社": 24, "中国新闻网": 24,
    "光明日报": 23, "经济日报": 23, "科技日报": 22,
    "Reuters": 25, "Associated Press": 25, "AP News": 25, "BBC": 24,
    "Bloomberg": 24, "Financial Times": 24, "The Wall Street Journal": 24,
    "The New York Times": 23, "CNBC": 22,
}

# 二级：大型门户财经频道
TIER2_SOURCES: dict[str, int] = {
    "新浪财经": 15, "新浪": 15, "网易财经": 15, "网易": 15,
    "凤凰财经": 16, "凤凰网": 16, "腾讯财经": 15, "腾讯": 15,
    "东方财富": 15, "搜狐财经": 14, "搜狐": 14, "华尔街见闻": 17,
    "FT中文网": 18, "财新网": 18,
    "CNN": 18, "The Guardian": 18, "Forbes": 17, "TechCrunch": 17,
    "Skift": 18, "PhocusWire": 18,
}

# 三级/聚合器
TIER3_SOURCES: dict[str, int] = {
    "百度新闻": 5, "搜狗新闻": 5, "今日头条": 7, "一点资讯": 5,
}

# 权威研究、咨询和公共机构：对其官方报告给予一手来源评分。
RESEARCH_INSTITUTIONS: dict[str, int] = {
    "Deloitte": 24, "McKinsey": 24, "Boston Consulting Group": 23,
    "BCG": 23, "PwC": 23, "KPMG": 23, "Ernst & Young": 23,
    "Accenture": 22, "Gartner": 22, "IDC": 22, "Forrester": 21,
}

PUBLIC_INSTITUTIONS: dict[str, int] = {
    "OECD": 25, "World Bank": 25, "International Monetary Fund": 25,
    "IMF": 25, "United Nations": 25, "World Economic Forum": 23,
}

# 域名规则与来源名规则同时工作；即使 RSS 来源名缺失，也能识别官网。
SOURCE_DOMAIN_PROFILES: dict[str, tuple[int, str]] = {
    # 国际权威媒体
    "reuters.com": (25, "国际权威媒体官网"),
    "apnews.com": (25, "国际权威媒体官网"),
    "bbc.com": (24, "国际权威媒体官网"),
    "bbc.co.uk": (24, "国际权威媒体官网"),
    "bloomberg.com": (24, "国际权威媒体官网"),
    "ft.com": (24, "国际权威媒体官网"),
    "wsj.com": (24, "国际权威媒体官网"),
    "nytimes.com": (23, "国际权威媒体官网"),
    "cnbc.com": (22, "国际主流媒体官网"),
    # 权威研究与咨询机构
    "deloitte.com": (24, "权威研究/咨询机构官网"),
    "mckinsey.com": (24, "权威研究/咨询机构官网"),
    "bcg.com": (23, "权威研究/咨询机构官网"),
    "pwc.com": (23, "权威研究/咨询机构官网"),
    "kpmg.com": (23, "权威研究/咨询机构官网"),
    "ey.com": (23, "权威研究/咨询机构官网"),
    "accenture.com": (22, "权威研究/咨询机构官网"),
    "gartner.com": (22, "专业研究机构官网"),
    "idc.com": (22, "专业研究机构官网"),
    "forrester.com": (21, "专业研究机构官网"),
    # 多边与公共机构
    "oecd.org": (25, "国际公共机构官网"),
    "worldbank.org": (25, "国际公共机构官网"),
    "imf.org": (25, "国际公共机构官网"),
    "un.org": (25, "国际公共机构官网"),
    "weforum.org": (23, "国际研究/公共机构官网"),
}

# 通用域名评分和扣分
GOV_DOMAIN_SCORE = 25
EDU_DOMAIN_SCORE = 22
BLOG_DOMAIN_PENALTY = -3  # blog.* / *.wordpress.* 降分
AGGREGATOR_DOMAIN_PENALTY = -2  # rss.app 等聚合域名降分

# 默认未知来源基础分
DEFAULT_SOURCE_SCORE = 10


def compute_source_credibility(source: str, url: str) -> tuple[int, str]:
    """计算来源权威度分数（0-25），返回 (分数, 评级理由)。"""
    source_clean = source.strip()
    domain = _extract_domain(url).lower()
    candidates: list[tuple[int, str]] = []

    source_groups = (
        ("权威研究/咨询机构", RESEARCH_INSTITUTIONS),
        ("国际公共机构", PUBLIC_INSTITUTIONS),
        ("一级媒体", TIER1_SOURCES),
        ("二级媒体", TIER2_SOURCES),
        ("聚合/平台来源", TIER3_SOURCES),
    )
    for category, sources in source_groups:
        for name, candidate_score in sources.items():
            if _source_name_matches(source_clean, name):
                candidates.append((candidate_score, f"{category}：{name}"))

    for known_domain, (candidate_score, category) in SOURCE_DOMAIN_PROFILES.items():
        if _domain_matches(domain, known_domain):
            candidates.append((candidate_score, f"{category}：{known_domain}"))

    domain_parts = set(domain.split("."))
    if domain and domain_parts.intersection({"gov", "govt", "gouv"}):
        candidates.append((GOV_DOMAIN_SCORE, "政府域名/公共机构官网"))
    if domain and (
        "edu" in domain_parts or re.search(r"\.ac\.[a-z]{2}$", domain)
    ):
        candidates.append((EDU_DOMAIN_SCORE, "教育/学术机构域名"))

    if candidates:
        score, primary_reason = max(candidates, key=lambda item: item[0])
        reasons = [primary_reason]
    else:
        score = DEFAULT_SOURCE_SCORE
        reasons = ["未知来源，使用中性基础分"]

    if any(kw in domain for kw in ("blog", "wordpress", "zhihu.com", "weixin.qq.com")):
        score += BLOG_DOMAIN_PENALTY
        reasons.append("个人/自媒体域名 -3")
    if any(kw in domain for kw in ("rss.app", "news.google.com", "newsnow", "feed")):
        score += AGGREGATOR_DOMAIN_PENALTY
        reasons.append("聚合域名 -2")

    return max(0, min(25, score)), "; ".join(reasons)


def compute_keyword_relevance(title: str, keyword: str) -> tuple[int, str]:
    """计算标题-关键词匹配度（0-30），返回 (分数, 理由)。"""
    if not keyword.strip() or not title.strip():
        return 3, "关键词或标题为空"

    kw = keyword.strip()
    t = title.strip()
    reasons: list[str] = []

    # 完整关键词出现在标题
    if kw in t:
        return 30, f"关键词「{kw}」完整出现在标题"

    # 关键词核心部分匹配
    kw_chars = list(kw)
    # 提取2-4字片段
    kw_segments = set()
    for length in (4, 3, 2):
        for i in range(len(kw_chars) - length + 1):
            kw_segments.add("".join(kw_chars[i:i + length]))
    # 单字也加入
    for c in kw_chars:
        if len(c.strip()) >= 1:
            kw_segments.add(c)

    if not kw_segments:
        return 3, "无法提取关键词片段"

    hit_count = sum(1 for seg in kw_segments if seg in t)
    hit_ratio = hit_count / len(kw_segments)

    if hit_ratio >= 0.7:
        score = 24
        reasons.append(f"关键词片段命中率 {hit_ratio:.0%}")
    elif hit_ratio >= 0.5:
        score = 18
        reasons.append(f"关键词片段命中率 {hit_ratio:.0%}")
    elif hit_ratio >= 0.3:
        score = 12
        reasons.append(f"关键词片段命中率 {hit_ratio:.0%}")
    elif hit_ratio >= 0.1:
        score = 6
        reasons.append(f"关键词片段命中率 {hit_ratio:.0%}")
    else:
        score = 3
        reasons.append("标题与关键词几乎无匹配")

    return score, "; ".join(reasons)


def compute_content_density(content: str) -> tuple[int, str]:
    """计算正文信息密度（0-20），返回 (分数, 理由)。"""
    reasons: list[str] = []
    length = len(content)
    paragraphs = [p for p in content.split("\n") if len(p.strip()) > 20]

    if length < 200:
        return 0, f"正文仅 {length} 字，硬跳过"

    if length > 3000:
        score = 20
        reasons.append(f"正文 {length} 字")
    elif length > 2000:
        score = 18
        reasons.append(f"正文 {length} 字")
    elif length > 1000:
        score = 14
        reasons.append(f"正文 {length} 字")
    elif length > 500:
        score = 10
        reasons.append(f"正文 {length} 字")
    elif length > 200:
        score = 6
        reasons.append(f"正文 {length} 字")
    else:
        score = 0

    if len(paragraphs) >= 5:
        avg_len = sum(len(p) for p in paragraphs) / len(paragraphs)
        if avg_len > 80:
            score = min(20, score + 2)
            reasons.append(f"段落结构好（{len(paragraphs)} 段，均长 {avg_len:.0f} 字）+2")

    return score, "; ".join(reasons)


def compute_data_richness(content: str) -> tuple[int, str]:
    """计算数据含量（0-15），基于数字/百分比密度。"""
    reasons: list[str] = []
    if not content.strip():
        return 0, "正文为空"

    sentences = re.split(r"[。！？；\n]", content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return 0, "无法分句"

    has_number = re.compile(r"\d+([.,，]?\d+)*\s*(亿|万|千|百|%|％|倍|元|美元|人|家|个)")
    has_simple_number = re.compile(r"\d{2,}")

    data_sentences = 0
    for s in sentences:
        if has_number.search(s) or has_simple_number.search(s):
            data_sentences += 1

    ratio = data_sentences / len(sentences)

    if ratio > 0.08:
        score = 15
        reasons.append(f"含数据句子占比 {ratio:.0%}")
    elif ratio > 0.05:
        score = 12
        reasons.append(f"含数据句子占比 {ratio:.0%}")
    elif ratio > 0.03:
        score = 9
        reasons.append(f"含数据句子占比 {ratio:.0%}")
    elif ratio > 0.01:
        score = 5
        reasons.append(f"含数据句子占比 {ratio:.0%}")
    else:
        score = 2
        reasons.append("几乎不含量化数据")

    return score, "; ".join(reasons)


def compute_freshness(published_at: str) -> tuple[int, str]:
    """计算时效性（0-10），返回 (分数, 理由)。"""
    reasons: list[str] = []
    dt = _parse_datetime(published_at)
    if dt is None:
        return 3, "无法解析发布时间，使用基础分"

    now = datetime.now(dt.tzinfo if dt.tzinfo else None) or datetime.now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    diff = now - dt

    if diff < timedelta(hours=24):
        score = 10
        reasons.append("24 小时内发布")
    elif diff < timedelta(days=3):
        score = 8
        reasons.append("3 天内发布")
    elif diff < timedelta(days=7):
        score = 6
        reasons.append("7 天内发布")
    elif diff < timedelta(days=30):
        score = 4
        reasons.append("30 天内发布")
    else:
        score = 2
        reasons.append("超过 30 天")

    return score, "; ".join(reasons)


# ============================================================
# 扣分规则（独立计算，不计入维度分数）
# ============================================================
@dataclass
class QualityPenalty:
    reason: str
    deduction: int


def compute_penalties(
    title: str,
    content: str,
    ai_result: dict[str, Any] | None = None,
) -> list[QualityPenalty]:
    """计算独立扣分项，返回扣分列表。"""
    penalties: list[QualityPenalty] = []

    # 1. 标题严重煽动性
    sensational_words = ["震惊", "可怕", "炸裂", "崩溃", "崩了", "疯了", "简直", "逆天",
                          "重磅突发", "刚刚", "紧急", "绝密"]
    hit_count = sum(1 for w in sensational_words if w in title)
    if hit_count >= 2:
        penalties.append(QualityPenalty(
            reason=f"标题含 {hit_count} 个煽动性词汇",
            deduction=8,
        ))
    elif hit_count == 1:
        penalties.append(QualityPenalty(
            reason="标题含煽动性词汇",
            deduction=4,
        ))

    # 2. 正文极短
    if len(content) < 100:
        penalties.append(QualityPenalty(
            reason="正文不足 100 字",
            deduction=15,
        ))

    # 3. AI 分析后发现的问题（如果有 ai_result）
    if ai_result:
        unsupported = ai_result.get("unsupportedClaims", [])
        if isinstance(unsupported, list) and unsupported:
            deduction = min(15, len(unsupported) * 5)
            penalties.append(QualityPenalty(
                reason=f"AI 发现 {len(unsupported)} 处缺乏支持的断言",
                deduction=deduction,
            ))

        # AI 判断标题-正文不一致
        consistency = ai_result.get("headlineBodyConsistency", 1.0)
        if isinstance(consistency, (int, float)) and consistency < 0.6:
            penalties.append(QualityPenalty(
                reason=f"标题与正文一致性低（{consistency:.0%}）",
                deduction=10,
            ))
        elif isinstance(consistency, (int, float)) and consistency < 0.8:
            penalties.append(QualityPenalty(
                reason=f"标题与正文一致性偏低（{consistency:.0%}）",
                deduction=5,
            ))

    return penalties


# ============================================================
# 汇总结构
# ============================================================
@dataclass
class QualitySummary:
    """质量评分汇总。"""
    total_score: int = 0  # 0-100
    dimension_scores: dict[str, int] = field(default_factory=dict)
    dimension_reasons: dict[str, str] = field(default_factory=dict)
    penalties: list[QualityPenalty] = field(default_factory=list)
    rule_version: str = NEWS_QUALITY_RULE_VERSION
    label: str = ""
    label_description: str = ""
    ai_confidence: float | None = None

    @property
    def total_penalty(self) -> int:
        return sum(p.deduction for p in self.penalties)

    @property
    def adjusted_score(self) -> int:
        return max(0, self.total_score - self.total_penalty)


def compute_quality_label(score: int) -> tuple[str, str]:
    """根据分数返回质量等级标签和描述。"""
    if score >= 90:
        return "优秀", "报道质量优秀，信息密度高，来源可靠"
    elif score >= 75:
        return "良好", "报道质量良好，信息较完整"
    elif score >= 60:
        return "一般", "报道质量一般，部分维度存在不足"
    elif score >= 40:
        return "偏低", "报道质量偏低，建议交叉验证"
    else:
        return "较低", "报道质量较低，信息不足或来源不明"


def score_article_pre_ai(
    title: str,
    url: str,
    content: str,
    source: str,
    published_at: str,
    keyword: str,
    content_hash: str,
) -> QualitySummary:
    """预 AI 评分（提取正文后立即执行）— 纯算法，不调 AI。"""
    dim_scores: dict[str, int] = {}
    dim_reasons: dict[str, str] = {}

    dim_scores["source_credibility"], dim_reasons["source_credibility"] = \
        compute_source_credibility(source, url)
    dim_scores["keyword_relevance"], dim_reasons["keyword_relevance"] = \
        compute_keyword_relevance(title, keyword)
    dim_scores["content_density"], dim_reasons["content_density"] = \
        compute_content_density(content)
    dim_scores["data_richness"], dim_reasons["data_richness"] = \
        compute_data_richness(content)
    dim_scores["freshness"], dim_reasons["freshness"] = \
        compute_freshness(published_at)

    total = sum(dim_scores.values())
    penalties = compute_penalties(title, content, ai_result=None)
    label, label_desc = compute_quality_label(max(0, total - sum(p.deduction for p in penalties)))

    return QualitySummary(
        total_score=total,
        dimension_scores=dim_scores,
        dimension_reasons=dim_reasons,
        penalties=penalties,
        label=label,
        label_description=label_desc,
    )


def enrich_with_ai_result(
    summary: QualitySummary,
    ai_result: dict[str, Any] | None,
) -> QualitySummary:
    """用 AI 分析结果增强评分。"""
    if not ai_result or not isinstance(ai_result, dict):
        return summary

    # 标题-正文一致性（AI 评估）
    consistency = ai_result.get("headlineBodyConsistency")
    if isinstance(consistency, (int, float)):
        consistency_score = int(round(consistency * 15))
        summary.dimension_scores["headline_body_consistency"] = max(0, min(15, consistency_score))
        summary.dimension_reasons["headline_body_consistency"] = \
            f"AI 评估一致性 {consistency:.0%}"

    # 原创性信号
    original_signals = ai_result.get("originalReportingSignals", [])
    if isinstance(original_signals, list):
        signal_score = min(10, len(original_signals) * 3)
        summary.dimension_scores["originality"] = signal_score
        summary.dimension_reasons["originality"] = \
            f"发现 {len(original_signals)} 个原创性信号" if original_signals else "未发现原创性信号"

    # 报道完整性（基于 AI 提取的结构化信息）
    completeness_score = 0
    completeness_reasons: list[str] = []
    if ai_result.get("involved_companies") or ai_result.get("namedSourceCount", 0) > 0:
        completeness_score += 4
        completeness_reasons.append("含人物/机构信息")
    if ai_result.get("hasBackgroundContext"):
        completeness_score += 4
        completeness_reasons.append("含背景信息")
    if ai_result.get("primaryDocumentCount", 0) > 0:
        completeness_score += 4
        completeness_reasons.append("引用原始材料")
    if ai_result.get("containsCounterpartyResponse"):
        completeness_score += 3
        completeness_reasons.append("含多方回应")
    if not completeness_reasons:
        completeness_reasons.append("未提取到完整性信号")
    summary.dimension_scores["completeness"] = completeness_score
    summary.dimension_reasons["completeness"] = "; ".join(completeness_reasons)

    # 透明度
    transparency_score = 0
    transparency_reasons: list[str] = []
    if ai_result.get("namedSourceCount", 0) > 0:
        transparency_score += 5
        transparency_reasons.append(f"引用 {ai_result['namedSourceCount']} 个具名信源")
    if ai_result.get("containsDirectQuotes"):
        transparency_score += 5
        transparency_reasons.append("含直接引用")
    if ai_result.get("articleType"):
        transparency_score += 3
        transparency_reasons.append(f"文章类型：{ai_result['articleType']}")
    if not transparency_reasons:
        transparency_reasons.append("未提取到透明度信号")
    summary.dimension_scores["transparency"] = transparency_score
    summary.dimension_reasons["transparency"] = "; ".join(transparency_reasons)

    # AI 置信度
    summary.ai_confidence = ai_result.get("ai_confidence")

    # 重新计算总分
    summary.total_score = sum(summary.dimension_scores.values())

    # 重新计算扣分（含 AI 发现的问题）
    title = ai_result.get("title", "")
    content = ai_result.get("content", "")
    summary.penalties = compute_penalties(title, content, ai_result=ai_result)

    # 更新标签
    summary.label, summary.label_description = compute_quality_label(summary.adjusted_score)

    return summary


def make_quality_cache_key(content_hash: str) -> str:
    """生成质量评分缓存键。"""
    return hashlib.sha256(
        f"{content_hash}:{NEWS_QUALITY_RULE_VERSION}".encode()
    ).hexdigest()


# ============================================================
# 内部辅助函数
# ============================================================

def _extract_domain(url: str) -> str:
    """从 URL 提取域名。"""
    try:
        return urlparse(url).hostname or ""
    except ValueError:
        return ""


def _domain_matches(domain: str, known_domain: str) -> bool:
    """严格匹配主域名或其子域名，避免伪造域名误命中。"""
    return domain == known_domain or domain.endswith(f".{known_domain}")


def _source_name_matches(source: str, known_name: str) -> bool:
    """中文名称按包含匹配，英文缩写按完整单词匹配。"""
    if not source or not known_name:
        return False
    if re.search(r"[A-Za-z]", known_name):
        pattern = rf"(?<![a-z0-9]){re.escape(known_name.casefold())}(?![a-z0-9])"
        return re.search(pattern, source.casefold()) is not None
    return known_name in source


def _parse_datetime(date_str: str) -> datetime | None:
    """尝试多种格式解析日期时间。"""
    if not date_str or not date_str.strip():
        return None
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%d %b %Y %H:%M:%S %z",
        "%b %d, %Y %H:%M:%S",
        "%b %d, %Y",
        "%Y年%m月%d日 %H:%M:%S",
        "%Y年%m月%d日",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, OverflowError):
            continue
    return None
