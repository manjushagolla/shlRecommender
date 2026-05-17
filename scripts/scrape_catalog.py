"""
SHL Catalog Scraper
===================
Run this on YOUR machine (not a sandboxed server).
Scrapes all Individual Test Solutions from shl.com.

Usage:
    python scripts/scrape_catalog.py
    python scripts/scrape_catalog.py --enrich   # also fetch detail pages
"""

import json, os, re, time, logging, argparse
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_URL     = "https://www.shl.com"
CATALOG_PATH = "/solutions/products/product-catalog/"
PAGE_SIZE    = 12
POLITE_DELAY = 1.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.shl.com/",
}

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude", "B": "Biodata & Situational Judgement",
    "C": "Competencies",       "D": "Development & 360",
    "E": "Assessment Exercises","K": "Knowledge & Skills",
    "P": "Personality & Behavior","S": "Simulations",
}

def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get(BASE_URL, timeout=15)
        time.sleep(1)
    except Exception:
        pass
    return s

@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=15),
       retry=retry_if_exception_type(requests.RequestException), reraise=True)
def fetch(session, url, params=None):
    r = session.get(url, params=params, timeout=25)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def _has_checkmark(cell):
    if any(c in cell.get_text(strip=True) for c in ("✓","●")): return True
    for el in cell.find_all(True):
        if any(k in " ".join(el.get("class",[])).lower() for k in ("yes","tick","check","active")): return True
        if el.name == "img": return True
    return False

def parse_page(soup):
    items = []
    table = soup.find("table")
    if not table:
        return items
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if not cells: continue
        a = cells[0].find("a", href=True)
        if not a: continue
        test_types = []
        for cell in cells[3:]:
            txt = cell.get_text(" ", strip=True)
            for letter in TEST_TYPE_LABELS:
                if re.search(rf'\b{letter}\b', txt): test_types.append(letter)
        items.append({
            "name":           a.get_text(strip=True),
            "url":            urljoin(BASE_URL, a["href"]),
            "remote_testing": _has_checkmark(cells[1]) if len(cells)>1 else False,
            "adaptive_irt":   _has_checkmark(cells[2]) if len(cells)>2 else False,
            "test_types":     test_types,
            "test_type":      test_types[0] if test_types else "",
        })
    return items

def detect_last_start(soup):
    mx = 0
    for a in soup.find_all("a", href=True):
        m = re.search(r'[?&]start=(\d+)', a["href"])
        if m: mx = max(mx, int(m.group(1)))
    return mx

def enrich_item(session, item):
    ALL_LEVELS = ["Entry-Level","General Population","Graduate","Front Line Manager",
                  "Supervisor","Manager","Mid-Professional","Professional Individual Contributor",
                  "Director","Executive"]
    try:
        soup = fetch(session, item["url"])
        page_text = soup.get_text(" ")
        for sel in [".product-catalogue__hero p",".product-description",".intro-text","main p"]:
            el = soup.select_one(sel)
            if el:
                t = el.get_text(" ", strip=True)
                if len(t) > 40: item.setdefault("description", t); break
        item.setdefault("job_levels", [lv for lv in ALL_LEVELS if lv.lower() in page_text.lower()])
        m = re.search(r'(\d+)\s*(?:min|minute)', page_text, re.I)
        if m: item.setdefault("duration_minutes", int(m.group(1)))
    except Exception as e:
        log.warning("detail fetch failed for '%s': %s", item["name"], e)
    return item

def scrape_catalog(enrich=False):
    session = make_session()
    params0 = {"action_doFilteringForm":"Search","f":"1","type":"1","start":"0"}
    soup0 = fetch(session, BASE_URL + CATALOG_PATH, params=params0)
    items = parse_page(soup0)
    log.info("page start=0 → %d items", len(items))
    last_start = detect_last_start(soup0)
    for start in range(PAGE_SIZE, last_start + PAGE_SIZE, PAGE_SIZE):
        time.sleep(POLITE_DELAY)
        try:
            soup = fetch(session, BASE_URL + CATALOG_PATH, params={**params0, "start": str(start)})
            page_items = parse_page(soup)
            if not page_items: break
            items.extend(page_items)
            log.info("start=%d → %d items (total: %d)", start, len(page_items), len(items))
        except Exception as e:
            log.warning("start=%d failed: %s", start, e)
    seen, unique = set(), []
    for it in items:
        if it["url"] not in seen:
            seen.add(it["url"]); unique.append(it)
    log.info("✅ %d unique assessments scraped", len(unique))
    if enrich:
        for i, item in enumerate(unique, 1):
            time.sleep(POLITE_DELAY)
            unique[i-1] = enrich_item(session, item)
            if i % 10 == 0: log.info("enriched %d/%d", i, len(unique))
    return unique

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrich", action="store_true")
    parser.add_argument("--out", default="catalog/shl_catalog.json")
    args = parser.parse_args()
    out_path = os.path.join(os.path.dirname(__file__), "..", args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = scrape_catalog(enrich=args.enrich)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Done. {len(data)} assessments saved to {args.out}")