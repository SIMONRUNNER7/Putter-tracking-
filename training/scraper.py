"""
Putter image scraper — vue de dessus uniquement.

On cible exclusivement la vue "adresse" (shaft entrant par le bas de l'image),
typique des photos produit des marques golf.

Sources:
  1. Shopify product API  — fairwayjockey, runner.golf, bettinardi, lab golf, evnroll
  2. Brand direct pages   — Scotty Cameron, Odyssey, TaylorMade, PXG, Mizuno, Miura,
                            Wilson, Cobra, PGA Tour Superstore, Golf Galaxy
  3. DuckDuckGo / Bing   — requêtes très ciblées "top view / address view"

Usage:
    python training/scraper.py scrape --out data/raw_images --limit 400
    python training/scraper.py stats  --dir data/raw_images
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
# Search queries — VIEW FROM ABOVE ONLY
# "address view" = shaft enters from bottom of the photo
# ---------------------------------------------------------------------------

SEARCH_QUERIES: list[str] = [
    # ── Vue de dessus générique ────────────────────────────────────────────
    "putter head top view overhead address position product photo",
    "putter head address view from above white background",
    "putter overhead view shaft golf product photo",
    "putter top down view blade mallet product",
    "golf putter crown view overhead studio photo",

    # ── Scotty Cameron ────────────────────────────────────────────────────
    "scotty cameron putter top view address overhead",
    "scotty cameron newport putter overhead product photo",
    "scotty cameron phantom putter top view",
    "scotty cameron putter head from above address",

    # ── Odyssey ───────────────────────────────────────────────────────────
    "odyssey putter top view overhead address",
    "odyssey ai-one putter top view product",
    "odyssey white hot putter overhead address view",
    "odyssey tri-hot putter top view from above",

    # ── TaylorMade ────────────────────────────────────────────────────────
    "taylormade spider putter top view overhead product",
    "taylormade truss putter address view from above",
    "taylormade putter head overhead studio",

    # ── Bettinardi ────────────────────────────────────────────────────────
    "bettinardi putter top view overhead address",
    "bettinardi queen b putter overhead product photo",
    "bettinardi studio stock putter top view",

    # ── PXG ───────────────────────────────────────────────────────────────
    "pxg putter top view overhead address",
    "pxg 0211 putter overhead product photo",

    # ── LAB Golf ──────────────────────────────────────────────────────────
    "lab golf putter top view overhead address",
    "lab df3 putter overhead product photo from above",

    # ── Evnroll ───────────────────────────────────────────────────────────
    "evnroll putter top view overhead address",
    "evnroll er2 putter overhead product",

    # ── Mizuno ────────────────────────────────────────────────────────────
    "mizuno putter top view overhead address product",
    "mizuno m-craft putter overhead studio photo",

    # ── Miura ─────────────────────────────────────────────────────────────
    "miura putter top view overhead address product photo",
    "miura km-350 putter overhead",

    # ── Wilson ────────────────────────────────────────────────────────────
    "wilson harmonized putter top view overhead address",
    "wilson staff putter head overhead product photo",

    # ── Cobra ─────────────────────────────────────────────────────────────
    "cobra putter top view overhead address product",
    "cobra king putter overhead studio photo",

    # ── Fairway Jockey / custom ───────────────────────────────────────────
    "fairway jockey custom putter top view overhead",
    "custom putter milled top view overhead white background",
    "handcrafted putter head top view studio photo",
]

# ---------------------------------------------------------------------------
# URL keywords that suggest a top/address view image
# (many brands use predictable suffixes in CDN image filenames)
# ---------------------------------------------------------------------------

TOP_VIEW_URL_HINTS = [
    "top", "overhead", "address", "crown",
    "_2.", "_02.", "-2.", "-02.",   # typically 2nd product image = top view
    "_3.", "_03.", "-3.", "-03.",   # sometimes 3rd
]

# ---------------------------------------------------------------------------
# Image constraints
# ---------------------------------------------------------------------------

MIN_WIDTH    = 300
MIN_HEIGHT   = 300
MAX_WIDTH    = 4000
MAX_HEIGHT   = 4000
MIN_FILESIZE = 15_000     # bytes — avoid thumbnails
MAX_FILESIZE = 10_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ===========================================================================
# SHOPIFY STORES  — /products.json gives clean CDN image URLs
# ===========================================================================

SHOPIFY_STORES: list[dict] = [
    {"name": "fairwayjockey",   "url": "https://fairwayjockey.com"},
    {"name": "runner_golf",     "url": "https://runner.golf"},
    {"name": "bettinardi",      "url": "https://bettinardigolf.com"},
    {"name": "lab_golf",        "url": "https://labgolf.com"},
    {"name": "evnroll",         "url": "https://evnroll.com"},
    {"name": "miura_golf",      "url": "https://miuragolf.com"},
]


def _shopify_product_images(
    store_url: str,
    session: requests.Session,
    putter_keywords: tuple[str, ...] = ("putter",),
) -> Iterator[str]:
    """
    Fetch all product images from a Shopify store via /products.json.
    Filters products whose title contains one of putter_keywords.
    Tries to return the top-view image (index 1 or 2) rather than all.
    """
    page = 1
    while True:
        try:
            url = f"{store_url}/products.json?limit=250&page={page}"
            r = session.get(url, headers=HEADERS, timeout=20)
            if r.status_code != 200:
                break
            data = r.json()
            products = data.get("products", [])
            if not products:
                break

            for product in products:
                title = product.get("title", "").lower()
                if not any(kw in title for kw in putter_keywords):
                    continue

                images = product.get("images", [])
                # Prefer image index 1 or 2 (= 2nd or 3rd photo, often top view)
                # but yield all so annotate.py can filter
                for img in images:
                    src = img.get("src", "")
                    if src:
                        # Strip Shopify size suffix for full-res
                        src = src.split("?")[0]
                        # Add _1000x for a decent size
                        yield src

            page += 1
            time.sleep(0.5)

            if len(products) < 250:
                break

        except Exception as e:
            print(f"  [Shopify:{store_url}] {e}")
            break


# ===========================================================================
# BRAND DIRECT SCRAPERS
# ===========================================================================

def _scrape_brand_generic(
    session: requests.Session,
    listing_url: str,
    brand_name: str,
    img_filter=None,
) -> Iterator[str]:
    """
    Generic BeautifulSoup scraper for a product listing page.
    img_filter(src) -> bool to keep only relevant images.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  [bs4] pip install beautifulsoup4")
        return

    try:
        r = session.get(listing_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup.find_all("img"):
            src = (tag.get("src") or tag.get("data-src") or
                   tag.get("data-lazy-src") or tag.get("data-original") or "")
            if src.startswith("//"):
                src = "https:" + src
            if not src.startswith("http"):
                continue
            if img_filter and not img_filter(src):
                continue
            yield src

    except Exception as e:
        print(f"  [{brand_name}] {e}")


def _scrape_scotty_cameron(session: requests.Session) -> Iterator[str]:
    """Scotty Cameron product listing + detail pages."""
    from bs4 import BeautifulSoup

    base_urls = [
        "https://scottycameron.titleist.com/equipment/putters",
        "https://scottycameron.titleist.com/equipment/putters/newport",
        "https://scottycameron.titleist.com/equipment/putters/phantom",
        "https://scottycameron.titleist.com/equipment/putters/special-select",
    ]

    for url in base_urls:
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(r.text, "html.parser")
            product_links: list[str] = []

            # Try to find product detail links
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/equipment/putters/" in href and href != url:
                    full = href if href.startswith("http") else "https://scottycameron.titleist.com" + href
                    if full not in product_links:
                        product_links.append(full)

            # Also yield images from listing page
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or ""
                if "putter" in src.lower() or "putters" in src.lower():
                    if src.startswith("http"):
                        yield src

            # Visit each product detail page
            for plink in product_links[:20]:
                try:
                    pr = session.get(plink, headers=HEADERS, timeout=20)
                    psoup = BeautifulSoup(pr.text, "html.parser")
                    for img in psoup.find_all("img"):
                        src = img.get("src") or img.get("data-src") or ""
                        if src.startswith("http"):
                            yield src
                    time.sleep(0.7)
                except Exception:
                    pass

            time.sleep(1.0)
        except Exception as e:
            print(f"  [ScottyCameron:{url}] {e}")


def _scrape_odyssey(session: requests.Session) -> Iterator[str]:
    """Odyssey Golf product pages."""
    urls = [
        "https://www.odysseygolf.com/en-us/putters/blade.html",
        "https://www.odysseygolf.com/en-us/putters/mallet.html",
        "https://www.odysseygolf.com/en-us/putters",
    ]
    for url in urls:
        yield from _scrape_brand_generic(session, url, "Odyssey",
            img_filter=lambda s: any(x in s.lower() for x in ["putter", "odyssey", "golf"]))
        time.sleep(1.0)


def _scrape_taylormade(session: requests.Session) -> Iterator[str]:
    """TaylorMade Golf putter pages."""
    urls = [
        "https://www.taylormadegolf.com/putters",
        "https://www.taylormadegolf.com/en-US/putters/all-putters/",
    ]
    for url in urls:
        yield from _scrape_brand_generic(session, url, "TaylorMade",
            img_filter=lambda s: "putter" in s.lower() or "cdn" in s.lower())
        time.sleep(1.0)


def _scrape_pxg(session: requests.Session) -> Iterator[str]:
    """PXG putter pages."""
    urls = [
        "https://www.pxg.com/en-us/golf-clubs/putters",
    ]
    for url in urls:
        yield from _scrape_brand_generic(session, url, "PXG",
            img_filter=lambda s: any(x in s.lower() for x in ["putter", "pxg", "cdn"]))
        time.sleep(1.0)


def _scrape_mizuno(session: requests.Session) -> Iterator[str]:
    """Mizuno Golf putter pages."""
    urls = [
        "https://golf.mizunousa.com/collections/putters",
        "https://golf.mizunousa.com/collections/m-craft-putters",
    ]
    # Mizuno USA is likely Shopify
    for store in urls:
        base = store.split("/collections/")[0]
        yield from _shopify_product_images(base, session)
        time.sleep(1.0)


def _scrape_wilson(session: requests.Session) -> Iterator[str]:
    """Wilson Golf putter pages."""
    urls = [
        "https://www.wilson.com/en-us/golf/clubs/putters",
    ]
    for url in urls:
        yield from _scrape_brand_generic(session, url, "Wilson",
            img_filter=lambda s: any(x in s.lower() for x in ["putter", "wilson", "harmonized"]))
        time.sleep(1.0)


def _scrape_cobra(session: requests.Session) -> Iterator[str]:
    """Cobra Golf putter pages."""
    urls = [
        "https://www.cobragolf.com/putters",
    ]
    for url in urls:
        yield from _scrape_brand_generic(session, url, "Cobra",
            img_filter=lambda s: any(x in s.lower() for x in ["putter", "cobra", "king"]))
        time.sleep(1.0)


def _scrape_pga_tour_superstore(session: requests.Session, max_pages: int = 5) -> Iterator[str]:
    """PGA Tour Superstore putter listing pages."""
    from bs4 import BeautifulSoup

    for page in range(1, max_pages + 1):
        url = f"https://www.pgatoursuperstore.com/putters/?start={(page-1)*48}&sz=48"
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(r.text, "html.parser")
            for img in soup.find_all("img"):
                src = (img.get("src") or img.get("data-src") or
                       img.get("data-lazy-src") or "")
                if src.startswith("//"):
                    src = "https:" + src
                if src.startswith("http") and len(src) > 20:
                    yield src
            time.sleep(1.5)
        except Exception as e:
            print(f"  [PGASuperstore p{page}] {e}")
            break


def _scrape_golf_galaxy(session: requests.Session, max_pages: int = 5) -> Iterator[str]:
    """Golf Galaxy putter listing pages."""
    from bs4 import BeautifulSoup

    for page in range(1, max_pages + 1):
        url = (
            f"https://www.golfgalaxy.com/c/putters"
            f"?pageNumber={page}&sortOption=FEATURED&inStoreOnly=false"
        )
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(r.text, "html.parser")
            for img in soup.find_all("img"):
                src = (img.get("src") or img.get("data-src") or
                       img.get("data-lazy-src") or "")
                if src.startswith("//"):
                    src = "https:" + src
                if src.startswith("http"):
                    yield src
            time.sleep(1.5)
        except Exception as e:
            print(f"  [GolfGalaxy p{page}] {e}")
            break


# ===========================================================================
# SEARCH ENGINE SCRAPERS
# ===========================================================================

def _ddg_search(query: str, max_results: int = 40) -> Iterator[str]:
    """DuckDuckGo image search — tries library then HTTP fallback."""
    DDGS = None
    for mod_name, cls in [("ddgs", "DDGS"), ("duckduckgo_search", "DDGS")]:
        try:
            mod = __import__(mod_name, fromlist=[cls])
            DDGS = getattr(mod, cls)
            break
        except ImportError:
            pass

    if DDGS:
        try:
            with DDGS() as client:
                for r in client.images(query, max_results=max_results,
                                       size="Large", type_image="photo"):
                    url = r.get("image") or r.get("url")
                    if url:
                        yield url
            return
        except Exception as e:
            print(f"  [DDG] {e}")

    # HTTP fallback
    try:
        vqd_r = requests.get("https://duckduckgo.com/", params={"q": query},
                              headers=HEADERS, timeout=10)
        vqd = None
        for part in vqd_r.text.split("vqd="):
            if len(part) > 5:
                vqd = part.split('"')[1] if '"' in part[:30] else part.split("'")[1]
                break
        if not vqd:
            return
        r = requests.get("https://duckduckgo.com/i.js",
                         params={"l": "us-en", "o": "json", "q": query,
                                 "vqd": vqd, "f": ",,,,,", "p": "1"},
                         headers=HEADERS, timeout=10)
        for item in r.json().get("results", [])[:max_results]:
            url = item.get("image")
            if url:
                yield url
    except Exception as e:
        print(f"  [DDG fallback] {e}")


def _bing_search(query: str, max_results: int = 25) -> Iterator[str]:
    """Bing image search fallback."""
    try:
        from bs4 import BeautifulSoup
        r = requests.get(
            "https://www.bing.com/images/search",
            params={"q": query, "form": "HDRSC2", "first": "1"},
            headers=HEADERS, timeout=12,
        )
        soup = BeautifulSoup(r.text, "html.parser")
        count = 0
        for tag in soup.find_all("a", class_="iusc"):
            try:
                m = json.loads(tag.get("m", "{}"))
                url = m.get("murl") or m.get("turl")
                if url:
                    yield url
                    count += 1
                    if count >= max_results:
                        break
            except Exception:
                continue
    except Exception as e:
        print(f"  [Bing] {e}")


# ===========================================================================
# IMAGE SCORER — prefer top-view / address-view candidates
# ===========================================================================

def _score_url_for_top_view(url: str) -> int:
    """
    Heuristic: return a score >0 if the URL suggests a top/address view image.
    Higher = more likely to be the right angle.
    """
    lower = url.lower()
    score = 0
    for hint in ("top", "overhead", "address", "crown", "above"):
        if hint in lower:
            score += 3
    # 2nd or 3rd product image is often the top view on e-commerce sites
    for suffix in ("_2.", "_02.", "-2.", "-02.", "/2.", "_3.", "_03.", "-3.", "-03."):
        if suffix in lower:
            score += 2
    # Penalise images likely to be "face on" (front of putter)
    for bad in ("face", "front", "detail", "grip", "shaft", "back", "sole",
                "lifestyle", "hero", "model", "green", "grass"):
        if bad in lower:
            score -= 1
    return score


# ===========================================================================
# DOWNLOAD + VALIDATE
# ===========================================================================

def _download_image(
    url: str,
    out_dir: Path,
    session: requests.Session,
    seen_hashes: set,
) -> tuple[bool, str]:
    try:
        r = session.get(url, headers=HEADERS, timeout=15, stream=True)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"

        content = r.content
        if len(content) < MIN_FILESIZE:
            return False, "too_small"
        if len(content) > MAX_FILESIZE:
            return False, "too_large"

        h = hashlib.md5(content).hexdigest()
        if h in seen_hashes:
            return False, "duplicate"
        seen_hashes.add(h)

        img = Image.open(io.BytesIO(content))
        if img.format not in ALLOWED_FORMATS:
            return False, f"bad_format:{img.format}"
        w, hp = img.size
        if w < MIN_WIDTH or hp < MIN_HEIGHT:
            return False, f"tiny:{w}x{hp}"
        if w > MAX_WIDTH or hp > MAX_HEIGHT:
            ratio = MAX_WIDTH / max(w, hp)
            img = img.resize((int(w * ratio), int(hp * ratio)), Image.LANCZOS)

        if img.mode in ("RGBA", "P") or img.format == "WEBP":
            img = img.convert("RGB")
            ext = ".jpg"
            fmt = "JPEG"
        else:
            fmt = img.format
            ext = ".jpg" if fmt == "JPEG" else ".png"

        fname = out_dir / (h[:16] + ext)
        if fmt == "JPEG":
            img.save(fname, "JPEG", quality=93, optimize=True)
        else:
            img.save(fname, fmt)

        return True, str(fname)

    except Exception as e:
        return False, str(e)[:60]


# ===========================================================================
# MAIN ORCHESTRATOR
# ===========================================================================

def scrape(
    out_dir: str = "data/raw_images",
    limit: int = 400,
    workers: int = 6,
    delay: float = 0.4,
    skip_search: bool = False,
    skip_brands: bool = False,
) -> int:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    session = requests.Session()
    session.headers.update(HEADERS)

    all_urls: list[str] = []

    # ------------------------------------------------------------------
    # 1. Shopify stores
    # ------------------------------------------------------------------
    if not skip_brands:
        print("\n[scrape] ── Shopify stores ───────────────────────────────")
        for store in SHOPIFY_STORES:
            print(f"  → {store['name']} ({store['url']})")
            urls = list(_shopify_product_images(store["url"], session))
            print(f"    {len(urls)} images found")
            all_urls.extend(urls)

        # ------------------------------------------------------------------
        # 2. Brand direct scrapers
        # ------------------------------------------------------------------
        print("\n[scrape] ── Brand direct pages ──────────────────────────")
        brand_scrapers = [
            ("Scotty Cameron",      _scrape_scotty_cameron),
            ("Odyssey",             _scrape_odyssey),
            ("TaylorMade",          _scrape_taylormade),
            ("PXG",                 _scrape_pxg),
            ("Mizuno",              _scrape_mizuno),
            ("Wilson",              _scrape_wilson),
            ("Cobra",               _scrape_cobra),
            ("PGA Tour Superstore", _scrape_pga_tour_superstore),
            ("Golf Galaxy",         _scrape_golf_galaxy),
        ]
        for name, fn in brand_scrapers:
            try:
                print(f"  → {name}")
                urls = list(fn(session))
                print(f"    {len(urls)} images found")
                all_urls.extend(urls)
            except Exception as e:
                print(f"    ERROR: {e}")

    # ------------------------------------------------------------------
    # 3. Search engines
    # ------------------------------------------------------------------
    if not skip_search:
        print("\n[scrape] ── Search engines (top-view queries) ────────────")
        per_query = max(20, limit // len(SEARCH_QUERIES) + 5)
        for q in tqdm(SEARCH_QUERIES, desc="DDG", unit="query"):
            urls = list(_ddg_search(q, max_results=per_query))
            all_urls.extend(urls)
            time.sleep(delay + random.uniform(0.1, 0.3))

        print("[scrape] Bing fallback for top 12 queries…")
        for q in tqdm(SEARCH_QUERIES[:12], desc="Bing", unit="query"):
            urls = list(_bing_search(q, max_results=20))
            all_urls.extend(urls)
            time.sleep(delay + random.uniform(0.2, 0.4))

    # ------------------------------------------------------------------
    # 4. Prioritise + deduplicate URLs
    # ------------------------------------------------------------------
    print(f"\n[scrape] {len(all_urls)} total URLs before dedup")
    seen_urls: set[str] = set()
    unique_urls: list[tuple[int, str]] = []   # (score, url)
    for url in all_urls:
        if url not in seen_urls:
            seen_urls.add(url)
            unique_urls.append((_score_url_for_top_view(url), url))

    # Higher score first — more likely to be top-view
    unique_urls.sort(key=lambda x: -x[0])
    sorted_urls = [u for _, u in unique_urls]
    print(f"[scrape] {len(sorted_urls)} unique URLs. Downloading up to {limit}…")

    # ------------------------------------------------------------------
    # 5. Parallel download
    # ------------------------------------------------------------------
    saved = 0
    failed = 0
    errors: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_image, url, out_path, session, seen_hashes): url
            for url in sorted_urls[: limit * 4]
        }
        with tqdm(total=limit, desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                ok, reason = future.result()
                if ok:
                    saved += 1
                    pbar.update(1)
                    if saved >= limit:
                        for f in futures:
                            f.cancel()
                        break
                else:
                    failed += 1
                    errors[reason] = errors.get(reason, 0) + 1

    # ------------------------------------------------------------------
    # 6. Manifest
    # ------------------------------------------------------------------
    manifest = {
        "total_downloaded": saved,
        "total_failed": failed,
        "failure_reasons": errors,
        "output_dir": str(out_path.resolve()),
        "note": "Images prioritised by top/address-view URL score. Use annotate.py to curate.",
    }
    with open(out_path / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[scrape] Done: {saved} images → {out_path}")
    print(f"         Failed: {failed}  |  Top reasons: {dict(list(errors.items())[:5])}")
    print(f"\nProchaine étape : python3 annotate.py {out_path}/")
    return saved


# ===========================================================================
# STATS
# ===========================================================================

def stats(img_dir: str) -> None:
    p = Path(img_dir)
    images = list(p.glob("*.jpg")) + list(p.glob("*.png"))
    print(f"\n[stats] {p.resolve()}")
    print(f"        {len(images)} images")
    if not images:
        return
    widths, heights, sizes = [], [], []
    for ip in images:
        try:
            img = Image.open(ip)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            sizes.append(ip.stat().st_size / 1024)
        except Exception:
            pass
    if widths:
        print(f"        Résolution moy. : {int(sum(widths)/len(widths))} × {int(sum(heights)/len(heights))}")
        print(f"        Taille moy.     : {sum(sizes)/len(sizes):.1f} KB")
        print(f"        Taille totale   : {sum(sizes)/1024:.1f} MB")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape putter images (vue de dessus) pour l'entraînement YOLO"
    )
    sub = parser.add_subparsers(dest="cmd")

    sc = sub.add_parser("scrape", help="Télécharger les images")
    sc.add_argument("--out",          default="data/raw_images")
    sc.add_argument("--limit",        type=int,   default=400)
    sc.add_argument("--workers",      type=int,   default=6)
    sc.add_argument("--delay",        type=float, default=0.4)
    sc.add_argument("--skip-search",  action="store_true", help="Ne pas utiliser DDG/Bing")
    sc.add_argument("--skip-brands",  action="store_true", help="Ne pas scraper les marques directement")

    st = sub.add_parser("stats", help="Statistiques d'un dossier")
    st.add_argument("--dir", default="data/raw_images")

    args = parser.parse_args()
    if args.cmd == "scrape":
        scrape(
            out_dir=args.out,
            limit=args.limit,
            workers=args.workers,
            delay=args.delay,
            skip_search=args.skip_search,
            skip_brands=args.skip_brands,
        )
    elif args.cmd == "stats":
        stats(args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
