#!/usr/bin/env python3
"""Download public IMSDb scripts to local text files for analysis."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import mimetypes
import re
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from urllib.parse import urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "Missing dependency. Install with: pip install requests beautifulsoup4"
    ) from exc


BASE_URL = "https://imsdb.com"
INDEX_URL = f"{BASE_URL}/all-scripts.html"
USER_AGENT = "character-analysis/imsdb-downloader (+local research use)"


@dataclass
class ListingEntry:
    title: str
    detail_url: str


@dataclass
class DownloadRecord:
    title: str
    detail_url: str
    script_url: str | None
    writers: list[str]
    genres: list[str]
    date_label: str | None
    date_value: str | None
    text_path: str | None
    detail_html_path: str | None
    script_html_path: str | None
    raw_path: str | None
    status: str
    error: str | None = None


class RateLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self._last_request_at = 0.0

    def wait(self) -> None:
        if self.delay_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

    def mark(self) -> None:
        self._last_request_at = time.monotonic()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def safe_filename(title: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".")
    return cleaned[:180] or "untitled"


def collapse_blank_lines(text: str, max_run: int = 2) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        line = line.rstrip()
        if line.strip():
            blank_run = 0
            collapsed.append(line)
            continue
        blank_run += 1
        if blank_run <= max_run:
            collapsed.append("")

    return "\n".join(collapsed) + ("\n" if collapsed else "")


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def fetch(
    session: requests.Session,
    url: str,
    *,
    rate_limiter: RateLimiter,
    timeout: float,
    attempts: int = 3,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            rate_limiter.wait()
            response = session.get(url, timeout=timeout)
            rate_limiter.mark()
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def parse_listing_page(page_html: str) -> list[ListingEntry]:
    entries: list[ListingEntry] = []
    seen: set[str] = set()

    section_match = re.search(
        r"<h1>\s*All Movie Scripts on IMSDb \(A-Z\)\s*</h1>(.*?)<br><br>\s*</table>",
        page_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    listing_html = section_match.group(1) if section_match else page_html

    for href, title in re.findall(
        r'<a href="(/Movie Scripts/[^"]+ Script\.html)"[^>]*>(.*?)</a>',
        listing_html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        absolute_url = urljoin(BASE_URL, href)
        if absolute_url in seen:
            continue
        normalized_title = normalize_space(html_lib.unescape(title))
        if not normalized_title:
            continue
        seen.add(absolute_url)
        entries.append(ListingEntry(title=normalized_title, detail_url=absolute_url))

    return entries


def parse_detail_page(html: str, fallback_title: str, detail_url: str) -> DownloadRecord:
    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.select_one("table.script-details h1") or soup.find("h1")
    title = normalize_space(title_node.get_text(" ", strip=True)) if title_node else fallback_title
    if title.endswith(" Script"):
        title = title[: -len(" Script")].strip()

    writers = [
        normalize_space(node.get_text(" ", strip=True))
        for node in soup.select('table.script-details a[href^="/writer.php?w="]')
    ]
    genres = [
        normalize_space(node.get_text(" ", strip=True))
        for node in soup.select('table.script-details a[href^="/genre/"]')
    ]

    script_url = None
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        if "/scripts/" in href:
            script_url = urljoin(BASE_URL, href)
            break

    match = re.search(
        r"<b>\s*(Script Date|Movie Release Date)\s*</b>\s*:\s*([^<]+)",
        html,
        flags=re.IGNORECASE,
    )
    date_label = match.group(1) if match else None
    date_value = normalize_space(match.group(2)) if match else None

    return DownloadRecord(
        title=title,
        detail_url=detail_url,
        script_url=script_url,
        writers=writers,
        genres=genres,
        date_label=date_label,
        date_value=date_value,
        text_path=None,
        detail_html_path=None,
        script_html_path=None,
        raw_path=None,
        status="pending",
        error=None,
    )


def looks_like_html(url: str, content_type: str) -> bool:
    lowered = content_type.lower()
    return url.endswith(".html") or "text/html" in lowered or "application/xhtml" in lowered


def guess_suffix(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix
    if suffix:
        return suffix
    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
    return guessed or ".bin"


def extract_script_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("td.scrtext")
    if container is None:
        raise ValueError("Could not find script text container (td.scrtext).")

    for node in container.select("script, style"):
        node.decompose()

    pre_blocks = container.find_all("pre")
    if pre_blocks:
        text_source = max(pre_blocks, key=lambda node: len(node.get_text()))
        text = text_source.get_text(strip=False)
    else:
        text = container.get_text("\n", strip=False)

    return collapse_blank_lines(text)


def filter_entries(
    entries: Iterable[ListingEntry],
    *,
    title_contains: list[str],
    start_at: str | None,
    limit: int | None,
) -> list[ListingEntry]:
    filtered: list[ListingEntry] = []
    for entry in entries:
        lowered = entry.title.lower()
        if start_at and lowered < start_at.lower():
            continue
        if title_contains and not all(token.lower() in lowered for token in title_contains):
            continue
        filtered.append(entry)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def download_scripts(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    scripts_dir = output_dir / "scripts"
    detail_html_dir = output_dir / "detail_html"
    script_html_dir = output_dir / "script_html"
    raw_dir = output_dir / "raw"

    output_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if args.save_detail_html:
        detail_html_dir.mkdir(parents=True, exist_ok=True)
    if args.save_script_html:
        script_html_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    rate_limiter = RateLimiter(args.delay)

    print(f"Fetching IMSDb index: {INDEX_URL}")
    index_response = fetch(session, INDEX_URL, rate_limiter=rate_limiter, timeout=args.timeout)
    index_html = index_response.text
    entries = parse_listing_page(index_html)
    write_json(output_dir / "catalog.json", [asdict(entry) for entry in entries])

    selected = filter_entries(
        entries,
        title_contains=args.title_contains or [],
        start_at=args.start_at,
        limit=args.limit,
    )
    print(f"Found {len(entries)} scripts in catalog; selected {len(selected)} entries.")

    if args.index_only:
        return 0

    results: list[DownloadRecord] = []
    failures = 0
    downloaded = 0
    skipped = 0

    for idx, entry in enumerate(selected, start=1):
        slug = safe_filename(entry.title)
        text_path = scripts_dir / f"{slug}.txt"

        if text_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{idx}/{len(selected)}] Skip existing: {entry.title}")
            results.append(
                DownloadRecord(
                    title=entry.title,
                    detail_url=entry.detail_url,
                    script_url=None,
                    writers=[],
                    genres=[],
                    date_label=None,
                    date_value=None,
                    text_path=str(text_path),
                    detail_html_path=None,
                    script_html_path=None,
                    raw_path=None,
                    status="skipped_existing",
                    error=None,
                )
            )
            continue

        print(f"[{idx}/{len(selected)}] Downloading: {entry.title}")
        try:
            detail_response = fetch(
                session,
                entry.detail_url,
                rate_limiter=rate_limiter,
                timeout=args.timeout,
            )
            detail_html = detail_response.text
            record = parse_detail_page(detail_html, fallback_title=entry.title, detail_url=entry.detail_url)

            if args.save_detail_html:
                detail_path = detail_html_dir / f"{safe_filename(record.title)}.html"
                detail_path.write_text(detail_html, encoding="utf-8")
                record.detail_html_path = str(detail_path)

            if not record.script_url:
                raise ValueError("Could not find 'Read Script' URL on detail page.")

            script_response = fetch(
                session,
                record.script_url,
                rate_limiter=rate_limiter,
                timeout=args.timeout,
            )
            content_type = script_response.headers.get("Content-Type", "")

            if looks_like_html(record.script_url, content_type):
                script_html = script_response.text
                text = extract_script_text(script_html)
                text_path.write_text(text, encoding="utf-8")
                record.text_path = str(text_path)

                if args.save_script_html:
                    script_path = script_html_dir / f"{safe_filename(record.title)}.html"
                    script_path.write_text(script_html, encoding="utf-8")
                    record.script_html_path = str(script_path)
            else:
                raw_suffix = guess_suffix(record.script_url, content_type)
                raw_path = raw_dir / f"{safe_filename(record.title)}{raw_suffix}"
                raw_path.write_bytes(script_response.content)
                record.raw_path = str(raw_path)

            record.status = "downloaded"
            results.append(record)
            downloaded += 1
        except Exception as exc:
            failures += 1
            print(f"  Failed: {exc}", file=sys.stderr)
            results.append(
                DownloadRecord(
                    title=entry.title,
                    detail_url=entry.detail_url,
                    script_url=None,
                    writers=[],
                    genres=[],
                    date_label=None,
                    date_value=None,
                    text_path=None,
                    detail_html_path=None,
                    script_html_path=None,
                    raw_path=None,
                    status="failed",
                    error=str(exc),
                )
            )

    write_jsonl(output_dir / "downloads.jsonl", [asdict(record) for record in results])

    print(
        f"Done. downloaded={downloaded} skipped={skipped} failed={failures} "
        f"output={output_dir}"
    )
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch download public IMSDb scripts as local text files."
    )
    parser.add_argument(
        "--output-dir",
        default="data/imsdb",
        help="Directory for saved scripts and metadata. Default: %(default)s",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only download the first N matching entries.",
    )
    parser.add_argument(
        "--title-contains",
        action="append",
        default=[],
        help="Case-insensitive substring filter. Repeatable.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Skip titles alphabetically before this value.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Minimum seconds between requests. Default: %(default)s",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if the target text file already exists.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only fetch the catalog and write catalog.json.",
    )
    parser.add_argument(
        "--save-detail-html",
        action="store_true",
        help="Also save each movie detail page as HTML.",
    )
    parser.add_argument(
        "--save-script-html",
        action="store_true",
        help="Also save each raw script page as HTML.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return download_scripts(args)


if __name__ == "__main__":
    raise SystemExit(main())
