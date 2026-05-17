"""
scripts/patch_catalog.py
========================
Fixes missing descriptions for known items.
Run once, then re-validate.

Usage:
    python scripts/patch_catalog.py
"""

import json, os

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "catalog", "shl_catalog.json")

# Manual descriptions for items the scraper couldn't find text for
PATCHES = {
    "DSI v1.1 Interpretation Report": (
        "Interpretation report for the Dependability and Safety Instrument (DSI). "
        "Provides detailed scoring and behavioural insights for roles requiring "
        "reliability, rule-following, and workplace safety compliance."
    ),
    "Verify Interactive Ability Report": (
        "Candidate-facing report generated from Verify Interactive assessments. "
        "Summarises performance across ability dimensions (numerical, verbal, inductive) "
        "in a clear, accessible format suitable for feedback conversations."
    ),
    "Verify Interactive G+ Candidate Report": (
        "Personalised candidate report for the Verify Interactive G+ global ability "
        "assessment. Explains scores across cognitive ability dimensions and provides "
        "development suggestions for the candidate."
    ),
    "Verify Interactive G+ Report": (
        "Recruiter-facing interpretive report for the Verify Interactive G+ assessment. "
        "Provides norm-referenced cognitive ability scores across verbal, numerical and "
        "inductive reasoning for selection and benchmarking decisions."
    ),
}

with open(CATALOG_PATH, encoding="utf-8") as f:
    data = json.load(f)

patched = 0
for item in data:
    name = item.get("name", "")
    if name in PATCHES and not item.get("description"):
        item["description"] = PATCHES[name]
        # rebuild embedding_text to include the new description
        parts = [f"Assessment: {item['name']}"]
        if item.get("test_types"):
            parts.append(f"Type: {', '.join(item['test_types'])}")
        parts.append(f"Description: {item['description']}")
        if item.get("job_levels"):
            parts.append(f"Job levels: {', '.join(item['job_levels'])}")
        if item.get("keywords"):
            parts.append(f"Keywords: {', '.join(item['keywords'][:20])}")
        item["embedding_text"] = ". ".join(parts)
        print(f"  ✅ Patched: {name}")
        patched += 1

with open(CATALOG_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Patched {patched} items. Run validate_catalog.py to confirm.")