#!/usr/bin/env python3
"""Collect candidate AI reading links from curated RSS/Atom feeds.

The script intentionally produces a draft list instead of editing repository
content. A maintainer should still decide which links deserve curated summaries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import email.utils
import html
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


KEYWORDS: dict[str, list[str]] = {
    "llm": [
        "llm",
        "large language model",
        "large language models",
        "language model",
        "foundation model",
        "gpt",
        "claude",
        "gemini",
        "deepseek",
        "qwen",
        "llama",
        "大模型",
        "语言模型",
        "基础模型",
        "通用模型",
    ],
    "agent": [
        "agent",
        "agents",
        "agentic",
        "tool use",
        "tools",
        "workflow",
        "workflows",
        "multi-agent",
        "mcp",
        "memory",
        "guardrail",
        "guardrails",
        "智能体",
        "代理",
        "工具调用",
        "函数调用",
        "多智能体",
        "记忆",
        "工作流",
        "护栏",
    ],
    "coding-agent": [
        "coding agent",
        "codex",
        "claude code",
        "software engineering",
        "developer",
        "code",
        "repository",
        "编程智能体",
        "代码智能体",
        "代码生成",
        "软件工程",
        "开发者",
        "仓库",
    ],
    "rag": [
        "rag",
        "retrieval",
        "retrieval-augmented",
        "embedding",
        "embeddings",
        "rerank",
        "context engineering",
        "检索增强",
        "检索",
        "向量",
        "嵌入",
        "重排",
        "上下文工程",
    ],
    "eval": [
        "eval",
        "evals",
        "evaluation",
        "benchmark",
        "observability",
        "tracing",
        "评测",
        "评估",
        "基准",
        "可观测",
        "观测",
        "追踪",
    ],
    "infra": [
        "inference",
        "serving",
        "deployment",
        "latency",
        "cost",
        "cache",
        "sandbox",
        "推理",
        "部署",
        "服务化",
        "延迟",
        "成本",
        "缓存",
        "沙箱",
        "工程化",
    ],
    "multimodal": [
        "multimodal",
        "vision",
        "video",
        "image",
        "document",
        "ocr",
        "多模态",
        "视觉",
        "视频",
        "图像",
        "文档",
        "语音",
    ],
    "safety": [
        "safety",
        "security",
        "control",
        "alignment",
        "risk",
        "安全",
        "对齐",
        "风险",
        "控制",
        "红队",
    ],
    "training": [
        "training",
        "fine-tuning",
        "finetuning",
        "post-training",
        "distillation",
        "rlhf",
        "lora",
        "训练",
        "微调",
        "后训练",
        "蒸馏",
    ],
}


def strip_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def parse_date(value: str) -> dt.datetime | None:
    if not value:
        return None

    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except (TypeError, ValueError):
        pass

    normalized = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        pass

    for date_format in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(value, date_format)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)
        except ValueError:
            continue

    return None


def parse_date_from_link(link: str) -> dt.datetime | None:
    match = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|-)", link)
    if not match:
        return None

    try:
        year, month, day = (int(part) for part in match.groups())
        return dt.datetime(year, month, day, tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def children(node: ET.Element, name: str) -> list[ET.Element]:
    expected = name.lower()
    return [child for child in list(node) if local_name(child.tag) == expected]


def descendants(node: ET.Element, name: str) -> list[ET.Element]:
    expected = name.lower()
    return [child for child in node.iter() if child is not node and local_name(child.tag) == expected]


def element_text(node: ET.Element, names: list[str]) -> str:
    expected = {name.lower() for name in names}
    for child in list(node):
        if local_name(child.tag) in expected and child.text:
            return child.text.strip()
    return ""


def element_link(node: ET.Element, atom: bool) -> str:
    if atom:
        for found in children(node, "link"):
            href = found.attrib.get("href", "").strip()
            rel = found.attrib.get("rel", "alternate")
            if href and rel == "alternate":
                return href
        return ""

    link = element_text(node, ["link"])
    if link:
        return link
    guid = element_text(node, ["guid"])
    return guid if guid.startswith("http") else ""


def parse_entries(xml_bytes: bytes, source: dict[str, Any]) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    root_name = local_name(root.tag)
    is_atom = root_name == "feed"
    entries: list[ET.Element]

    if is_atom:
        entries = children(root, "entry")
    else:
        channel = children(root, "channel")
        entries = children(channel[0], "item") if channel else children(root, "item")
        if not entries:
            entries = descendants(root, "item")

    parsed: list[dict[str, Any]] = []
    for item in entries:
        if is_atom:
            title = element_text(item, ["title"])
            summary = element_text(
                item,
                [
                    "summary",
                    "content",
                ],
            )
            date_text = element_text(
                item,
                [
                    "published",
                    "updated",
                ],
            )
        else:
            title = element_text(item, ["title"])
            summary = element_text(item, ["description", "encoded", "summary", "content"])
            date_text = element_text(item, ["pubDate", "published", "updated", "date"])

        link = element_link(item, is_atom)
        if not title or not link:
            continue
        published = parse_date(date_text) or parse_date_from_link(link)

        parsed.append(
            {
                "source": source["name"],
                "title": strip_html(title),
                "link": link,
                "summary": strip_html(summary),
                "published": published,
                "topics": source.get("topics", []),
                "weight": int(source.get("weight", 1)),
                "language": source.get("language", ""),
                "category": source.get("category", ""),
            }
        )

    return parsed


def fetch_feed(source: dict[str, Any], timeout: int) -> bytes:
    request = urllib.request.Request(
        source["feed_url"],
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; AIReadingCandidates/1.0)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
            "Accept-Encoding": "identity",
            "Connection": "close",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def keyword_matches(haystack: str, keyword: str) -> bool:
    keyword = keyword.lower()
    if keyword.isascii():
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        return re.search(pattern, haystack) is not None
    return keyword in haystack


def score_item(item: dict[str, Any]) -> tuple[int, list[str]]:
    summary = item.get("summary", "")
    haystack = f"{item['title']} {summary[:800]}".lower()
    matched_topics: list[str] = []
    score = item.get("weight", 1)

    for topic, keywords in KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword_matches(haystack, keyword))
        if hits:
            matched_topics.append(topic)
            score += min(hits, 3)

    return score, matched_topics


def is_excluded(item: dict[str, Any], source: dict[str, Any]) -> bool:
    exclude_keywords = source.get("exclude_keywords", [])
    if not exclude_keywords:
        return False

    summary = item.get("summary", "")
    haystack = f"{item['title']} {summary[:800]}".lower()
    return any(keyword_matches(haystack, keyword) for keyword in exclude_keywords)


def fetch_source_entries(source: dict[str, Any], timeout: int) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    try:
        feed = fetch_feed(source, timeout)
        entries = parse_entries(feed, source)
        return source, entries, ""
    except Exception as exc:  # noqa: BLE001 - keep scheduled collection resilient.
        return source, [], f"- {source['name']}: {type(exc).__name__}: {exc}"


def collect(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.since_days)
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        source_results = executor.map(lambda source: fetch_source_entries(source, args.timeout), sources)

        for source, entries, warning in source_results:
            if warning:
                warnings.append(warning)
                continue

            for item in entries:
                published = item.get("published")
                if not published and not args.include_undated:
                    continue
                if published and published < cutoff:
                    continue
                if is_excluded(item, source):
                    continue

                score, matched_topics = score_item(item)
                min_score = max(args.min_score, int(source.get("min_score", args.min_score)))
                if score < min_score or not matched_topics:
                    continue

                item["score"] = score
                item["matched_topics"] = matched_topics
                candidates.append(item)

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for item in sorted(
        candidates,
        key=lambda value: (value.get("score", 0), value.get("published") or dt.datetime.min.replace(tzinfo=dt.timezone.utc)),
        reverse=True,
    ):
        link = item["link"].split("?")[0].rstrip("/")
        if link in seen:
            continue
        source = item["source"]
        if args.max_per_source > 0 and source_counts.get(source, 0) >= args.max_per_source:
            continue
        seen.add(link)
        source_counts[source] = source_counts.get(source, 0) + 1
        unique.append(item)
        if len(unique) >= args.max_items:
            break

    return unique, warnings


def render_markdown(items: list[dict[str, Any]], warnings: list[str], args: argparse.Namespace) -> str:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    lines = [
        f"# Reading candidates {today}",
        "",
        "These links were collected automatically from curated RSS feeds.",
        "Please review them before adding anything to `reading/YYYY/MM.md`.",
        "",
        f"- Window: last {args.since_days} days",
        f"- Max items: {args.max_items}",
        f"- Max per source: {args.max_per_source if args.max_per_source > 0 else 'unlimited'}",
        "",
    ]

    if warnings:
        lines.extend(["## Source warnings", ""])
        lines.extend(warnings)
        lines.append("")

    if not items:
        lines.extend(["## Candidates", "", "No matching candidates found."])
        return "\n".join(lines) + "\n"

    lines.extend(["## Candidates", ""])
    for index, item in enumerate(items, start=1):
        published = item.get("published")
        published_text = published.date().isoformat() if published else "unknown"
        summary = item.get("summary", "")
        if len(summary) > 280:
            summary = summary[:277].rstrip() + "..."

        lines.extend(
            [
                f"### {index}. {item['title']}",
                "",
                f"- Link: {item['link']}",
                f"- Source: {item['source']}",
                f"- Language: {item.get('language') or 'unknown'}",
                f"- Published: {published_text}",
                f"- Matched topics: {', '.join(item.get('matched_topics', []))}",
                f"- Score: {item.get('score', 0)}",
                f"- Draft summary: {summary or 'No feed summary provided.'}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="reading/sources.json")
    parser.add_argument("--since-days", type=int, default=7)
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--max-per-source", type=int, default=3)
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-undated", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    items, warnings = collect(args)
    markdown = render_markdown(items, warnings, args)

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
