"""
scripts/validate_catalog.py
============================
Quickly validate your shl_catalog.json.
Run any time after editing or re-scraping.

Usage:
    python scripts/validate_catalog.py
    python scripts/validate_catalog.py --verbose
"""

import json, os, sys, re, argparse
from collections import Counter, defaultdict

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "catalog", "shl_catalog.json")

REQUIRED_FIELDS = ["name", "url", "test_types", "test_type", "description",
                   "job_levels", "keywords", "embedding_text"]
VALID_TEST_TYPES = set("ABCDEKPS")
VALID_LEVELS     = {
    "Entry-Level","General Population","Graduate","Front Line Manager",
    "Supervisor","Manager","Mid-Professional","Professional Individual Contributor",
    "Director","Executive",
}

def validate(data, verbose=False):
    errors = []
    warnings = []

    seen_urls  = {}
    seen_names = {}

    for i, item in enumerate(data):
        label = f"[{i}] {item.get('name','UNNAMED')!r}"

        # required fields
        for f in REQUIRED_FIELDS:
            if not item.get(f):
                errors.append(f"{label}: missing '{f}'")

        # URL must be shl.com
        url = item.get("url","")
        if url and not url.startswith("https://www.shl.com"):
            errors.append(f"{label}: URL not from shl.com — {url}")

        # duplicate URLs
        if url in seen_urls:
            errors.append(f"{label}: duplicate URL (also at index {seen_urls[url]})")
        else:
            seen_urls[url] = i

        # duplicate names
        name = item.get("name","")
        if name in seen_names:
            warnings.append(f"{label}: duplicate name (also at index {seen_names[name]})")
        else:
            seen_names[name] = i

        # test type letters
        for tt in item.get("test_types", []):
            if tt not in VALID_TEST_TYPES:
                errors.append(f"{label}: unknown test_type letter '{tt}'")

        # job levels
        for lv in item.get("job_levels", []):
            if lv not in VALID_LEVELS:
                warnings.append(f"{label}: unrecognized job level '{lv}'")

        # keyword count
        kw_count = len(item.get("keywords", []))
        if kw_count < 3:
            warnings.append(f"{label}: only {kw_count} keywords — too few for good retrieval")

        # description length
        desc = item.get("description","")
        if len(desc) < 20:
            warnings.append(f"{label}: very short description ({len(desc)} chars)")

        if verbose:
            print(f"  ✓ {name}")

    return errors, warnings


def print_summary(data):
    type_names = {"A":"Ability","B":"Biodata/SJT","C":"Competencies","D":"Development",
                  "E":"Exercises","K":"Knowledge","P":"Personality","S":"Simulations"}
    types = Counter(t for item in data for t in item.get("test_types",[]))
    levels= Counter(lv for item in data for lv in item.get("job_levels",[]))

    print(f"\n{'─'*54}")
    print(f"  SHL Catalog Summary — {len(data)} assessments")
    print(f"{'─'*54}")
    print("  Test types:")
    for k,v in sorted(types.items()):
        bar = "█" * v
        print(f"    {k} ({type_names.get(k,k):<30}) {v:>3}  {bar}")
    print("\n  Top job levels:")
    for lv, cnt in levels.most_common(6):
        print(f"    {lv:<40} {cnt}")
    avg_kw  = sum(len(i.get("keywords",[])) for i in data) / len(data)
    has_emb = sum(1 for i in data if i.get("embedding_text"))
    print(f"\n  Avg keywords/item : {avg_kw:.1f}")
    print(f"  Items with embedding_text: {has_emb}/{len(data)}")
    print(f"{'─'*54}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(CATALOG_PATH):
        print(f"❌  File not found: {CATALOG_PATH}")
        sys.exit(1)

    with open(CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)

    errors, warnings = validate(data, verbose=args.verbose)

    print_summary(data)

    if warnings:
        print(f"⚠   {len(warnings)} warnings:")
        for w in warnings[:10]:
            print(f"    {w}")
        if len(warnings) > 10:
            print(f"    ... and {len(warnings)-10} more")
        print()

    if errors:
        print(f"❌  {len(errors)} ERRORS — fix before building index:")
        for e in errors:
            print(f"    {e}")
        sys.exit(1)
    else:
        print(f"✅  All {len(data)} catalog items are valid and ready for embedding.")