"""
Putter image scraper for PutterTrack Pro dataset collection.

Sources (no API key required):
  1. DuckDuckGo Images  (primary)
  2. Bing Images        (fallback)
  3. Direct golf retailer product pages (Golf Galaxy, Rock Bottom Golf, etc.)

Usage:
    # Install deps
    pip install duckduckgo-search requests beautifulsoup4 Pillow tqdm aiohttp

    # Scrape everything
    python training/scraper.py scrape --out data/raw_images --limit 200

    # Check what was downloaded
    python training/scraper.py stats --dir data/raw_images
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import requests
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Search queries — cover all putter styles and camera angles seen in analysis
# ---------------------------------------------------------------------------

SEARCH_QUERIES = [
    # Face-on / front view (most important for our use case)
    "odyssey putter head face on grass",
    "scotty cameron putter head front view green",
    "blade putter head face on golf green",
    "mallet putter head face on green grass",
    "ping putter head front angle golf",
    "taylormade spider putter head front view",
    "titleist putter head face on green",
    "callaway putter head face on",
    "cleveland putter head front green",
    "bettinardi putter head face golf",
    "evnroll putter head front grass",
    "srixon putter head golf green",
    "mizuno putter head face view",

    # Slightly elevated / top-side (product photo style)
    "odyssey two ball putter head top",
    "scotty cameron newport putter close up",
    "blade putter head close up golf",
    "mallet putter head top view green",
    "anser style putter head golf",
    "plumber neck putter head close up",
    "face balanced putter head golf",

    # In-use / on green (training diversity)
    "putter head on putting green close up",
    "golf putter head address position",
    "putter face impact golf ball",
    "golf club putter head grass close",
    "putter setup address ball position",

    # By style
    "blade putter golf face",
    "mallet putter golf face",
    "arm lock putter head",
    "counterbalance putter head",
    "flow neck putter head",
    "center shaft putter head",
    "high toe putter head",

    # Brand variations for diversity
    "yes c groove putter head",
    "seemore putter head",
    "tour only raw putter head",
    "bobby grace putter head",
    "rife putter head golf",
    "acushnet putter head",
    "winn putter head golf",
]

# ---------------------------------------------------------------------------
# Image filtering parameters
# ---------------------------------------------------------------------------

MIN_WIDTH    = 200
MIN_HEIGHT   = 150
MAX_WIDTH    = 4000
MAX_HEIGHT   = 4000
MIN_FILESIZE = 8_000      # bytes (avoid tiny icons)
MAX_FILESIZE = 8_000_000  # bytes (avoid huge RAW photos)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# HTTP headers to mimic a real browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# DuckDuckGo image search
# ---------------------------------------------------------------------------

def _ddg_search_images(query: str, max_results: int = 50) -> Iterator[str]:
    """
    Yield image URLs from DuckDuckGo image search.
    Uses the duckduckgo_search library if available,
    otherwise falls back to direct HTTP parsing.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = ddgs.images(
                query,
                max_results=max_results,
                size="Medium",          # avoid tiny thumbnails
                type_image="photo",
            )
            for r in results:
                url = r.get("image") or r.get("url")
                if url:
                    yield url
    except ImportError:
        # Fallback: direct DDG endpoint
        yield from _ddg_fallback(query, max_results)
    except Exception as e:
        print(f"  [DDG] Error for '{query}': {e}")


def _ddg_fallback(query: str, max_results: int) -> Iterator[str]:
    """Direct DDG image search without the library."""
    try:
        # Step 1: get vqd token
        vqd_url = "https://duckduckgo.com/"
        params  = {"q": query}
        r = requests.get(vqd_url, params=params, headers=HEADERS, timeout=10)
        vqd = None
        for part in r.text.split("vqd="):
            if len(part) > 5:
                vqd = part.split('"')[1] if '"' in part[:30] else part.split("'")[1]
                break
        if not vqd:
            return

        # Step 2: fetch images
        img_url = "https://duckduckgo.com/i.js"
        params = {
            "l": "us-en", "o": "json", "q": query,
            "vqd": vqd, "f": ",,,,,", "p": "1",
        }
        r = requests.get(img_url, params=params, headers=HEADERS, timeout=10)
        data = r.json()
        for item in data.get("results", [])[:max_results]:
            url = item.get("image")
            if url:
                yield url
    except Exception as e:
        print(f"  [DDG fallback] Error: {e}")


# ---------------------------------------------------------------------------
# Bing image search (fallback)
# ---------------------------------------------------------------------------

def _bing_search_images(query: str, max_results: int = 30) -> Iterator[str]:
    """Scrape image URLs from Bing image search."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return

    try:
        url = "https://www.bing.com/images/search"
        params = {"q": query, "form": "HDRSC2", "first": "1", "tsc": "ImageHoverTitle"}
        r = requests.get(url, params=params, headers=HEADERS, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")
        count = 0
        for tag in soup.find_all("a", class_="iusc"):
            try:
                m = json.loads(tag.get("m", "{}"))
                img_url = m.get("murl") or m.get("turl")
                if img_url:
                    yield img_url
                    count += 1
                    if count >= max_results:
                        break
            except Exception:
                continue
    except Exception as e:
        print(f"  [Bing] Error for '{query}': {e}")


# ---------------------------------------------------------------------------
# Golf retailer direct scrapers
# ---------------------------------------------------------------------------

def _scrape_golf_galaxy(session: requests.Session, max_pages: int = 5) -> Iterator[str]:
    """Scrape putter product images from Golf Galaxy."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return

    base = "https://www.golfgalaxy.com"
    url  = f"{base}/c/putters?pageNumber=1&sortOption=FEATURED&inStoreOnly=false"
    for page in range(1, max_pages + 1):
        try:
            page_url = url.replace("pageNumber=1", f"pageNumber={page}")
            r = session.get(page_url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
            imgs = soup.find_all("img", class_=lambda c: c and "product" in c.lower())
            if not imgs:
                imgs = soup.find_all("img", src=lambda s: s and "putter" in s.lower())
            for img in imgs:
                src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
                if src and src.startswith("http"):
                    yield src
            time.sleep(1.0)
        except Exception:
            continue


def _scrape_rockbottom(session: requests.Session, max_pages: int = 3) -> Iterator[str]:
    """Scrape Rock Bottom Golf putter images."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return

    for page in range(1, max_pages + 1):
        try:
            url = f"https://www.rockbottomgolf.com/putters/?page={page}"
            r = session.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src", "")
                if src and any(x in src.lower() for x in ["putter", "golf", "club"]):
                    if src.startswith("//"):
                        src = "https:" + src
                    if src.startswith("http"):
                        yield src
            time.sleep(1.5)
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Download + validate
# ---------------------------------------------------------------------------

def _download_image(
    url: str,
    out_dir: Path,
    session: requests.Session,
    seen_hashes: set,
) -> tuple[bool, str]:
    """
    Download a single image, validate it, deduplicate, and save.
    Returns (success, reason).
    """
    try:
        r = session.get(url, headers=HEADERS, timeout=12, stream=True)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"

        content = r.content
        if len(content) < MIN_FILESIZE:
            return False, "too small"
        if len(content) > MAX_FILESIZE:
            return False, "too large"

        # Deduplication via MD5
        h = hashlib.md5(content).hexdigest()
        if h in seen_hashes:
            return False, "duplicate"
        seen_hashes.add(h)

        # Validate image
        img = Image.open(io.BytesIO(content))
        if img.format not in ALLOWED_FORMATS:
            return False, f"bad format {img.format}"
        w, h_px = img.size
        if w < MIN_WIDTH or h_px < MIN_HEIGHT:
            return False, f"too small {w}x{h_px}"
        if w > MAX_WIDTH or h_px > MAX_HEIGHT:
            img = img.resize((MAX_WIDTH, int(MAX_WIDTH * h_px / w)), Image.LANCZOS)

        # Convert WEBP → JPEG for compatibility
        if img.format == "WEBP" or img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            fmt = "JPEG"
            ext = ".jpg"
        else:
            fmt = img.format
            ext = ".jpg" if fmt == "JPEG" else ".png"

        fname = out_dir / (h[:16] + ext)
        if fmt == "JPEG":
            img.save(fname, "JPEG", quality=92, optimize=True)
        else:
            img.save(fname, fmt)

        return True, str(fname)

    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main scrape orchestrator
# ---------------------------------------------------------------------------

def scrape(
    out_dir: str,
    limit: int = 300,
    workers: int = 6,
    queries: list[str] | None = None,
    include_retailers: bool = True,
    delay: float = 0.3,
) -> int:
    """
    Scrape putter images from all sources into out_dir.
    Returns total images saved.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    queries = queries or SEARCH_QUERIES
    seen_hashes: set[str] = set()
    session = requests.Session()
    session.headers.update(HEADERS)

    # Collect all candidate URLs
    print(f"[scrape] Collecting URLs from {len(queries)} search queries…")
    all_urls: list[str] = []

    for q in tqdm(queries, desc="Searching DDG", unit="query"):
        urls = list(_ddg_search_images(q, max_results=max(20, limit // len(queries) + 5)))
        all_urls.extend(urls)
        time.sleep(delay + random.uniform(0.1, 0.4))

    # Bing fallback for more variety
    print(f"[scrape] Adding Bing results for top queries…")
    for q in tqdm(queries[:10], desc="Searching Bing", unit="query"):
        urls = list(_bing_search_images(q, max_results=15))
        all_urls.extend(urls)
        time.sleep(delay + random.uniform(0.2, 0.5))

    # Direct retailer scraping
    if include_retailers:
        print("[scrape] Scraping Golf Galaxy product pages…")
        all_urls.extend(list(_scrape_golf_galaxy(session, max_pages=4)))
        print("[scrape] Scraping Rock Bottom Golf…")
        all_urls.extend(list(_scrape_rockbottom(session, max_pages=3)))

    # Deduplicate URLs, shuffle for variety
    all_urls = list(dict.fromkeys(all_urls))  # preserve order + dedup
    random.shuffle(all_urls)
    print(f"[scrape] {len(all_urls)} unique URLs collected. Downloading up to {limit}…")

    # Parallel download
    saved = 0
    failed = 0
    errors: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_image, url, out_path, session, seen_hashes): url
            for url in all_urls[:limit * 3]   # over-fetch to hit the limit
        }
        with tqdm(total=limit, desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                ok, reason = future.result()
                if ok:
                    saved += 1
                    pbar.update(1)
                    if saved >= limit:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                else:
                    failed += 1
                    errors[reason] = errors.get(reason, 0) + 1

    # Save manifest
    manifest = {
        "total_downloaded": saved,
        "total_failed": failed,
        "failure_reasons": errors,
        "queries_used": queries,
        "output_dir": str(out_path.resolve()),
    }
    with open(out_path / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[scrape] Done: {saved} images saved to {out_path}")
    print(f"         Failed: {failed}  |  Reasons: {errors}")
    return saved


# ---------------------------------------------------------------------------
# Stats utility
# ---------------------------------------------------------------------------

def stats(img_dir: str) -> None:
    """Print statistics about a downloaded image directory."""
    p = Path(img_dir)
    if not p.exists():
        print(f"Directory not found: {img_dir}")
        return

    images = list(p.glob("*.jpg")) + list(p.glob("*.png"))
    print(f"\n[stats] Directory: {p.resolve()}")
    print(f"        Images: {len(images)}")

    if not images:
        return

    widths, heights, sizes = [], [], []
    for img_path in images:
        try:
            img = Image.open(img_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            sizes.append(img_path.stat().st_size / 1024)
        except Exception:
            continue

    print(f"        Avg resolution: {int(sum(widths)/len(widths))} × {int(sum(heights)/len(heights))}")
    print(f"        Avg file size:  {sum(sizes)/len(sizes):.1f} KB")
    print(f"        Total size:     {sum(sizes)/1024:.1f} MB")

    manifest_path = p / "_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            m = json.load(f)
        print(f"        Queries used:  {len(m.get('queries_used', []))}")
        print(f"        Failed:        {m.get('total_failed', '?')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape putter images for PutterTrack Pro YOLO training"
    )
    sub = parser.add_subparsers(dest="cmd")

    sc = sub.add_parser("scrape", help="Download images")
    sc.add_argument("--out",      default="data/raw_images",
                    help="Output directory (default: data/raw_images)")
    sc.add_argument("--limit",    type=int, default=300,
                    help="Max images to download (default: 300)")
    sc.add_argument("--workers",  type=int, default=6,
                    help="Parallel download workers (default: 6)")
    sc.add_argument("--delay",    type=float, default=0.3,
                    help="Seconds between search requests (default: 0.3)")
    sc.add_argument("--no-retailers", action="store_true",
                    help="Skip direct retailer scraping")
    sc.add_argument("--query",    action="append", dest="queries",
                    help="Extra search query (repeatable)")

    st = sub.add_parser("stats", help="Show stats for a downloaded directory")
    st.add_argument("--dir", default="data/raw_images")

    args = parser.parse_args()

    if args.cmd == "scrape":
        extra_queries = args.queries or []
        scrape(
            out_dir=args.out,
            limit=args.limit,
            workers=args.workers,
            delay=args.delay,
            include_retailers=not args.no_retailers,
            queries=SEARCH_QUERIES + extra_queries,
        )
    elif args.cmd == "stats":
        stats(args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
