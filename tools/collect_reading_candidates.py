#!/usr/bin/env python3
"""Collect candidate AI reading links from curated RSS/Atom feeds.

The script intentionally produces a draft list instead of editing repository
content. A maintainer should still decide which links deserve curated summaries.
"""

from __future__ import annotations

import argparse
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
    "agent": [
        "agent",
        "agents",
        "tool use",
        "tools",
        "workflow",
        "workflows",
        "multi-agent",
        "mcp",
        "memory",
        "guardrail",
        "guardrails",
    ],
    "coding-agent": [
        "coding agent",
        "codex",
        "claude code",
        "software engineering",
        "developer",
        "code",
        "repository",
    ],
    "rag": [
        "rag",
        "retrieval",
        "retrieval-augmented",
        "embedding",
        "embeddings",
        "rerank",
        "context engineering",
    ],
    "eval": [
        "eval",
        "evals",
        "evaluation",
        "benchmark",
        "observability",
        "tracing",
    ],
    "infra": [
        "inference",
        "serving",
        "deployment",
        "latency",
        "cost",
        "cache",
        "sandbox",
    ],
    "multimodal": [
        "multimodal",
        "vision",
        "video",
        "image",
        "document",
        "ocr",
    ],
    "safety": [
        "safety",
        "security",
        "control",
        "alignment",
        "risk",
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
        return None


def element_text(node: ET.Element, names: list[str]) -> str:
    for name in names:
        found = node.find(name)
        if found is not None and found.text:
            return found.text.strip()
    return ""


def element_link(node: ET.Element, atom: bool) -> str:
    if atom:
        for found in node.findall("{http://www.w3.org/2005/Atom}link"):
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
    is_atom = root.tag.endswith("feed")
    entries: list[ET.Element]

    if is_atom:
        entries = root.findall("{http://www.w3.org/2005/Atom}entry")
    else:
        entries = root.findall("./channel/item")

    parsed: list[dict[str, Any]] = []
    for item in entries:
        if is_atom:
            title = element_text(item, ["{http://www.w3.org/2005/Atom}title"])
            summary = element_text(
                item,
                [
                    "{http://www.w3.org/2005/Atom}summary",
                    "{http://www.w3.org/2005/Atom}content",
                ],
            )
            date_text = element_text(
                item,
                [
                    "{http://www.w3.org/2005/Atom}published",
                    "{http://www.w3.org/2005/Atom}updated",
                ],
            )
        else:
            title = element_text(item, ["title"])
            summary = element_text(item, ["description", "{http://purl.org/rss/1.0/modules/content/}encoded"])
            date_text = element_text(item, ["pubDate", "published", "updated"])

        link = element_link(item, is_atom)
        if not title or not link:
            continue

        parsed.append(
            {
                "source": source["name"],
                "title": strip_html(title),
                "link": link,
                "summary": strip_html(summary),
                "published": parse_date(date_text),
                "topics": source.get("topics", []),
                "weight": int(source.get("weight", 1)),
            }
        )

    return parsed


def fetch_feed(source: dict[str, Any], timeout: int) -> bytes:
    request = urllib.request.Request(
        source["feed_url"],
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; AIReadingCandidates/1.0)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def score_item(item: dict[str, Any]) -> tuple[int, list[str]]:
    haystack = f"{item['title']} {item.get('summary', '')}".lower()
    matched_topics: list[str] = []
    score = item.get("weight", 1)

    for topic, keywords in KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in haystack)
        if hits:
            matched_topics.append(topic)
            score += min(hits, 3)

    return score, matched_topics


def collect(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.since_days)
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    for source in sources:
        try:
            feed = fetch_feed(source, args.timeout)
            entries = parse_entries(feed, source)
        except Exception as exc:  # noqa: BLE001 - keep scheduled collection resilient.
            warnings.append(f"- {source['name']}: {type(exc).__name__}: {exc}")
            continue

        for item in entries:
            published = item.get("published")
            if published and published < cutoff:
                continue

            score, matched_topics = score_item(item)
            if score < args.min_score or not matched_topics:
                continue

            item["score"] = score
            item["matched_topics"] = matched_topics
            candidates.append(item)

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in sorted(
        candidates,
        key=lambda value: (value.get("score", 0), value.get("published") or dt.datetime.min.replace(tzinfo=dt.timezone.utc)),
        reverse=True,
    ):
        link = item["link"].split("?")[0].rstrip("/")
        if link in seen:
            continue
        seen.add(link)
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
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=15)
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
