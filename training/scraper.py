"""
Putter image scraper — vue de dessus uniquement.

On cible exclusivement la vue "adresse" (shaft entrant par le bas de l'image),
typique des photos produit des marques golf.

Sources:
  1. Shopify product API  — fairwayjockey, runner.golf, bettinardi, lab golf, evnroll,
                            2ndswing, rockbottomgolf, globalgolf, ping store
  2. eBay search          — requêtes ciblées top-view (HTML statique, fiable)
  3. Reddit JSON API      — r/golf, r/PuttingReview, r/GolfEquipment
  4. DuckDuckGo / Bing   — requêtes très ciblées "top view / address view"

Brand scrapers React/Next.js (Scotty Cameron, Odyssey, TaylorMade, PXG, Wilson,
Cobra) ont été supprimés : ils ne retournent aucune image via BeautifulSoup.

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
import re
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
    # ── Vue de dessus générique — termes photographiques précis ────────────
    "putter head top view overhead address position product photo",
    "putter head address view from above white background",
    "putter head flat lay overhead white background product photo",
    "putter head birds eye view studio photography",
    "putter head aerial view product shot golf",
    "golf putter top down flat lay photography",
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

    # ── Ping ──────────────────────────────────────────────────────────────
    "ping putter top view overhead address product photo",
    "ping anser putter overhead product studio",
    "ping sigma 2 putter top view",

    # ── Cleveland / Srixon ────────────────────────────────────────────────
    "cleveland huntington beach putter overhead address view",
    "srixon cleveland putter top view product photo",

    # ── Callaway ──────────────────────────────────────────────────────────
    "callaway odyssey ten putter top view overhead",
    "callaway jaws putter overhead product photo",

    # ── Autres marques ────────────────────────────────────────────────────
    "fourteen golf putter top view overhead",
    "cure putter top view address overhead product",
    "yes golf putter head top view overhead",
    "never compromise putter top view overhead address",
    "la golf putter overhead product photo",

    # ── Sites de review golf (photos haute qualité, vue de dessus) ────────
    "mygolfspy putter review top view address photo",
    "golf wrx putter top view review photo",
    "the hackers paradise putter overhead review",
    "golf digest putter top view address photo",

    # ── Custom / milled ───────────────────────────────────────────────────
    "fairway jockey custom putter top view overhead",
    "custom putter milled top view overhead white background",
]

# ---------------------------------------------------------------------------
# URL keywords that suggest a top/address view image
# ---------------------------------------------------------------------------

TOP_VIEW_URL_HINTS = [
    "top", "overhead", "address", "crown",
    "_2.", "_02.", "-2.", "-02.",
    "_3.", "_03.", "-3.", "-03.",
]

# ---------------------------------------------------------------------------
# Image constraints
# ---------------------------------------------------------------------------

MIN_WIDTH    = 300
MIN_HEIGHT   = 300
MAX_WIDTH    = 4000
MAX_HEIGHT   = 4000
MIN_FILESIZE = 15_000
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
    {"name": "2ndswing",        "url": "https://2ndswing.com"},
    {"name": "rockbottomgolf",  "url": "https://rockbottomgolf.com"},
    {"name": "globalgolf",      "url": "https://www.globalgolf.com"},
    {"name": "ping_store",      "url": "https://www.pingstoredirect.com"},
]


def _shopify_product_images(
    store_url: str,
    session: requests.Session,
    putter_keywords: tuple[str, ...] = ("putter",),
) -> Iterator[str]:
    """
    Fetch all product images from a Shopify store via /products.json.
    Filters products whose title contains one of putter_keywords.
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
                for img in images:
                    src = img.get("src", "")
                    if src:
                        src = src.split("?")[0]
                        yield src

            page += 1
            time.sleep(0.5)

            if len(products) < 250:
                break

        except Exception as e:
            print(f"  [Shopify:{store_url}] {e}")
            break


# ===========================================================================
# EBAY SCRAPER — HTML statique, fiable, nombreuses vues de dessus
# ===========================================================================

EBAY_QUERIES: list[str] = [
    "putter top view address",
    "scotty cameron putter overhead",
    "odyssey putter top view",
    "blade putter top view product",
    "mallet putter top view studio",
    "ping putter top view",
    "taylormade spider putter overhead",
    "bettinardi putter top view",
]


def _scrape_ebay(
    session: requests.Session,
    queries: list[str] = EBAY_QUERIES,
    max_pages: int = 3,
) -> Iterator[str]:
    """
    Scrape eBay product listing pages for putter images.
    eBay uses static HTML — BeautifulSoup works perfectly here.
    Thumb URLs (s-l225.jpg) are upgraded to full-res (s-l1600.jpg).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  [eBay] pip install beautifulsoup4")
        return

    for query in queries:
        for page in range(1, max_pages + 1):
            url = (
                "https://www.ebay.com/sch/i.html"
                f"?_nkw={urllib.parse.quote(query)}"
                f"&_sacat=0&_pgn={page}"
            )
            try:
                r = session.get(url, headers=HEADERS, timeout=20)
                soup = BeautifulSoup(r.text, "html.parser")
                found = 0
                for img in soup.find_all("img", src=True):
                    src = img["src"]
                    # Upgrade thumbnail to full resolution
                    src = re.sub(r"s-l\d+\.(jpg|jpeg)", r"s-l1600.\1", src)
                    if "i.ebayimg.com" in src:
                        yield src
                        found += 1
                if found == 0:
                    break  # No more pages
                time.sleep(1.5)
            except Exception as e:
                print(f"  [eBay:{query} p{page}] {e}")
                break

        time.sleep(1.0)


# ===========================================================================
# REDDIT JSON — r/golf, r/PuttingReview (aucun JS requis)
# ===========================================================================

REDDIT_SUBREDDITS: list[str] = ["golf", "PuttingReview", "GolfEquipment"]
REDDIT_QUERIES: list[str] = [
    "putter top view",
    "putter overhead",
    "putter address view",
    "putter flat lay",
]


def _scrape_reddit(session: requests.Session) -> Iterator[str]:
    """
    Reddit JSON API — no JavaScript needed, no auth required.
    Extracts preview image URLs from posts.
    """
    reddit_headers = {**HEADERS, "Accept": "application/json"}

    for sub in REDDIT_SUBREDDITS:
        for q in REDDIT_QUERIES[:2]:  # limit per sub to avoid rate limit
            url = (
                f"https://www.reddit.com/r/{sub}/search.json"
                f"?q={urllib.parse.quote(q)}&type=link&limit=50&restrict_sr=1"
            )
            try:
                r = session.get(url, headers=reddit_headers, timeout=15)
                if r.status_code != 200:
                    continue
                children = r.json().get("data", {}).get("children", [])
                for post in children:
                    d = post.get("data", {})
                    # Direct image URL
                    url_post = d.get("url", "")
                    if any(url_post.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                        yield url_post
                    # Preview image (higher quality)
                    preview = d.get("preview", {}).get("images", [])
                    if preview:
                        src = (preview[0].get("source", {}).get("url", "")
                               .replace("&amp;", "&"))
                        if src:
                            yield src
                time.sleep(2.0)
            except Exception as e:
                print(f"  [Reddit r/{sub}] {e}")

        time.sleep(1.5)


# ===========================================================================
# PGA TOUR SUPERSTORE + GOLF GALAXY — kept as HTML scrapers
# ===========================================================================

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
    """
    DuckDuckGo image search — robuste face aux changements d'API.
    Retry ×3 avec backoff exponentiel sur rate-limit.
    Fallback HTTP si la librairie échoue.
    """
    try:
        from duckduckgo_search import DDGS

        for attempt in range(3):
            try:
                with DDGS() as d:
                    results = list(d.images(query, max_results=max_results))
                for r in results:
                    url = r.get("image") or r.get("thumbnail")
                    if url:
                        yield url
                return  # success
            except Exception as e:
                err = str(e)
                if ("atelimit" in err or "429" in err) and attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"  [DDG] rate-limit, attente {wait}s…")
                    time.sleep(wait)
                else:
                    print(f"  [DDG] {e}")
                    break
    except ImportError:
        pass

    # ── HTTP fallback ──────────────────────────────────────────────────────
    try:
        vqd_r = requests.get(
            "https://duckduckgo.com/",
            params={"q": query},
            headers=HEADERS,
            timeout=10,
        )
        vqd = None
        for part in vqd_r.text.split("vqd="):
            if len(part) > 5:
                candidate = part.split('"')[1] if '"' in part[:30] else part.split("'")[1]
                if candidate:
                    vqd = candidate
                    break
        if not vqd:
            return
        r = requests.get(
            "https://duckduckgo.com/i.js",
            params={"l": "us-en", "o": "json", "q": query,
                    "vqd": vqd, "f": ",,,,,", "p": "1"},
            headers=HEADERS,
            timeout=10,
        )
        for item in r.json().get("results", [])[:max_results]:
            url = item.get("image")
            if url:
                yield url
    except Exception as e:
        print(f"  [DDG fallback] {e}")


def _bing_search(query: str, max_results: int = 25) -> Iterator[str]:
    """
    Bing image search — deux méthodes d'extraction pour robustesse.
    1. Attributs data-src des balises <img>
    2. Attributs JSON "m" des balises <a> (fallback)
    """
    try:
        from bs4 import BeautifulSoup
        r = requests.get(
            "https://www.bing.com/images/search",
            params={"q": query, "form": "HDRSC2", "first": "1"},
            headers=HEADERS,
            timeout=12,
        )
        soup = BeautifulSoup(r.text, "html.parser")
        count = 0

        # Méthode 1 : attributs data-src
        for img in soup.find_all("img", attrs={"data-src": True}):
            src = img["data-src"]
            if src.startswith("http") and any(
                ext in src.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]
            ):
                yield src
                count += 1
                if count >= max_results:
                    return

        # Méthode 2 : attributs JSON "m" (ancienne structure Bing)
        for tag in soup.find_all(attrs={"m": True}):
            if count >= max_results:
                break
            try:
                m = json.loads(tag.get("m", "{}"))
                url = m.get("murl") or m.get("turl")
                if url:
                    yield url
                    count += 1
            except Exception:
                continue

    except Exception as e:
        print(f"  [Bing] {e}")


# ===========================================================================
# IMAGE SCORER — prefer top-view / address-view candidates
# ===========================================================================

def _score_url_for_top_view(url: str) -> int:
    lower = url.lower()
    score = 0
    for hint in ("top", "overhead", "address", "crown", "above", "flatlay", "flat_lay"):
        if hint in lower:
            score += 3
    for suffix in ("_2.", "_02.", "-2.", "-02.", "/2.", "_3.", "_03.", "-3.", "-03."):
        if suffix in lower:
            score += 2
    for bad in ("face", "front", "detail", "grip", "shaft", "back", "sole",
                "lifestyle", "hero", "model", "green", "grass", "putting"):
        if bad in lower:
            score -= 1
    return score


# ===========================================================================
# POST-DOWNLOAD CV FILTER — rejette les images clairement pas vue de dessus
# ===========================================================================

def _passes_top_view_filter(img_path: Path) -> bool:
    """
    Filtre rapide OpenCV : rejette les images où l'objet principal est
    très allongé verticalement (= vue de face, pas vue de dessus).

    Returns True  → garder l'image
    Returns False → rejeter (supprimer)
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True  # erreur de lecture → laisser passer par défaut

        h, w = img.shape

        # Threshold inversé : isole les objets sombres sur fond clair
        _, thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)

        # Ignorer les bords (évite les artefacts JPEG aux coins)
        border = 5
        thresh[:border, :] = 0
        thresh[-border:, :] = 0
        thresh[:, :border] = 0
        thresh[:, -border:] = 0

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Essai sur fond sombre (putter noir sur fond vert)
            _, thresh2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return True  # pas de contour détectable → laisser passer

        largest = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(largest) / (h * w)

        # Objet trop petit → probablement un logo ou une miniature
        if area_ratio < 0.03:
            return False

        # Ratio d'aspect du bounding box
        _, _, cw, ch = cv2.boundingRect(largest)
        aspect = max(cw, ch) / max(min(cw, ch), 1)

        # Trop allongé → probablement vue de côté ou shaft complet
        if aspect > 6:
            return False

        return True

    except Exception:
        return True  # si OpenCV non disponible ou erreur → laisser passer


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

        # Post-download CV filter
        if not _passes_top_view_filter(fname):
            fname.unlink(missing_ok=True)
            return False, "cv_filter:not_top_view"

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
        # 2. eBay
        # ------------------------------------------------------------------
        print("\n[scrape] ── eBay (HTML statique) ────────────────────────")
        try:
            urls = list(_scrape_ebay(session))
            print(f"  {len(urls)} images trouvées sur eBay")
            all_urls.extend(urls)
        except Exception as e:
            print(f"  [eBay] ERREUR: {e}")

        # ------------------------------------------------------------------
        # 3. Reddit
        # ------------------------------------------------------------------
        print("\n[scrape] ── Reddit JSON API ──────────────────────────────")
        try:
            urls = list(_scrape_reddit(session))
            print(f"  {len(urls)} images trouvées sur Reddit")
            all_urls.extend(urls)
        except Exception as e:
            print(f"  [Reddit] ERREUR: {e}")

        # ------------------------------------------------------------------
        # 4. PGA Superstore + Golf Galaxy
        # ------------------------------------------------------------------
        print("\n[scrape] ── PGA Superstore + Golf Galaxy ─────────────────")
        for name, fn in [
            ("PGA Tour Superstore", _scrape_pga_tour_superstore),
            ("Golf Galaxy",         _scrape_golf_galaxy),
        ]:
            try:
                print(f"  → {name}")
                urls = list(fn(session))
                print(f"    {len(urls)} images found")
                all_urls.extend(urls)
            except Exception as e:
                print(f"    ERROR: {e}")

    # ------------------------------------------------------------------
    # 5. Search engines
    # ------------------------------------------------------------------
    if not skip_search:
        print("\n[scrape] ── Search engines (top-view queries) ────────────")
        per_query = max(20, limit // len(SEARCH_QUERIES) + 5)
        for q in tqdm(SEARCH_QUERIES, desc="DDG", unit="query"):
            urls = list(_ddg_search(q, max_results=per_query))
            all_urls.extend(urls)
            time.sleep(delay + random.uniform(0.1, 0.3))

        print("[scrape] Bing fallback pour les 15 premières requêtes…")
        for q in tqdm(SEARCH_QUERIES[:15], desc="Bing", unit="query"):
            urls = list(_bing_search(q, max_results=20))
            all_urls.extend(urls)
            time.sleep(delay + random.uniform(0.2, 0.4))

    # ------------------------------------------------------------------
    # 6. Prioritise + deduplicate URLs
    # ------------------------------------------------------------------
    print(f"\n[scrape] {len(all_urls)} total URLs before dedup")
    seen_urls: set[str] = set()
    unique_urls: list[tuple[int, str]] = []
    for url in all_urls:
        if url not in seen_urls:
            seen_urls.add(url)
            unique_urls.append((_score_url_for_top_view(url), url))

    unique_urls.sort(key=lambda x: -x[0])
    sorted_urls = [u for _, u in unique_urls]
    print(f"[scrape] {len(sorted_urls)} unique URLs. Téléchargement jusqu'à {limit}…")

    # ------------------------------------------------------------------
    # 7. Parallel download
    # ------------------------------------------------------------------
    saved = 0
    failed = 0
    errors: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_image, url, out_path, session, seen_hashes): url
            for url in sorted_urls[: limit * 5]  # pool 5× plus grand pour compenser le filtre CV
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
    # 8. Manifest
    # ------------------------------------------------------------------
    manifest = {
        "total_downloaded": saved,
        "total_failed": failed,
        "failure_reasons": errors,
        "output_dir": str(out_path.resolve()),
        "queries_used": SEARCH_QUERIES,
        "note": (
            "Images prioritisées par score URL top/address-view. "
            "Filtre CV post-download appliqué (rejet si objet trop allongé). "
            "Utilise annotate.py pour la curation finale."
        ),
    }
    with open(out_path / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[scrape] Done: {saved} images → {out_path}")
    print(f"         Failed: {failed}  |  Top raisons: {dict(list(errors.items())[:5])}")
    cv_filtered = errors.get("cv_filter:not_top_view", 0)
    if cv_filtered:
        print(f"         Filtre CV: {cv_filtered} images rejetées (vue de face détectée)")
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
    sc.add_argument("--skip-brands",  action="store_true", help="Ne pas scraper les marques/eBay/Reddit")

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
