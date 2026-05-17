"""
scripts/enrich_catalog.py
=========================
Run AFTER scrape_catalog.py.
Adds keyword tags, normalizes job levels, validates every entry,
and writes a final enriched catalog ready for embedding.

Usage:
    python scripts/enrich_catalog.py
"""

import json
import os
import re
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

CATALOG_IN  = os.path.join(os.path.dirname(__file__), "..", "catalog", "shl_catalog.json")
CATALOG_OUT = os.path.join(os.path.dirname(__file__), "..", "catalog", "shl_catalog.json")

# ── canonical job level names (must match exactly for filtering) ─────────────
VALID_LEVELS = {
    "Entry-Level", "General Population", "Graduate",
    "Front Line Manager", "Supervisor", "Manager",
    "Mid-Professional", "Professional Individual Contributor",
    "Director", "Executive",
}

# ── keyword rules: (regex pattern on name+description) → [keywords to add] ──
KEYWORD_RULES: list[tuple[str, list[str]]] = [
    # Programming languages
    (r'\bjava\b',       ["java", "backend", "developer", "programming", "software engineer"]),
    (r'\bpython\b',     ["python", "developer", "data science", "scripting", "backend"]),
    (r'\bjavascript\b', ["javascript", "frontend", "web development", "developer"]),
    (r'\bc#\b|csharp',  ["c#", ".net", "microsoft", "backend", "developer"]),
    (r'\bc\+\+\b',      ["c++", "systems programming", "embedded", "developer"]),
    (r'\breact\b',      ["react", "frontend", "javascript", "web", "developer"]),
    (r'\bangular\b',    ["angular", "typescript", "frontend", "developer"]),
    (r'\bnode\.?js\b',  ["nodejs", "backend", "javascript", "api", "developer"]),
    (r'\bspring\b',     ["spring", "java", "backend", "framework", "developer"]),
    (r'\bsql\b',        ["sql", "database", "data", "queries", "backend"]),
    (r'\bmongodb\b',    ["mongodb", "nosql", "database", "backend"]),
    (r'\baws\b',        ["aws", "cloud", "devops", "infrastructure", "amazon"]),
    (r'\bdevops\b',     ["devops", "cicd", "docker", "kubernetes", "cloud", "infrastructure"]),
    # Roles
    (r'\bnumerical\b',  ["numerical reasoning", "quantitative", "data analysis", "finance"]),
    (r'\bverbal\b',     ["verbal reasoning", "communication", "language", "writing"]),
    (r'\binductive\b',  ["abstract reasoning", "problem solving", "logical"]),
    (r'\bpersonality\b|opq',["personality", "behavior", "traits", "culture fit"]),
    (r'\bsales\b',      ["sales", "business development", "revenue", "commercial"]),
    (r'\bleadership\b|manag',["leadership", "management", "people management", "executive"]),
    (r'\bcustomer.?serv|call.?cent|contact.?cent',["customer service", "BPO", "call center", "CX"]),
    (r'\bgraduate\b',   ["graduate", "early careers", "campus", "entry professional"]),
    (r'\bsafety\b',     ["safety", "HSE", "manufacturing", "industrial", "compliance"]),
    (r'\bsimulat',      ["simulation", "practical", "hands-on", "realistic"]),
    (r'\bsituational.?judg|sjt\b',["situational judgment", "SJT", "decision making"]),
    (r'\bdata.?scien|machine.?learn', ["data science", "ML", "AI", "analytics", "python"]),
    (r'\bproject.?manag', ["project management", "PMP", "agile", "scrum", "PM"]),
    (r'\bfinance|accounting|banking',["finance", "financial", "accounting", "banking"]),
    (r'\bcyber|security\b', ["cybersecurity", "infosec", "security", "network"]),
    (r'\bclerical|administ|data.?entry',["clerical", "administrative", "office", "admin"]),
    (r'\bcoach|develop|360\b',["development", "coaching", "feedback", "360", "growth"]),
]

TEST_TYPE_DESCRIPTIONS = {
    "A": "Ability and Aptitude test measuring cognitive skills",
    "B": "Biodata or Situational Judgement Test (SJT)",
    "C": "Competency assessment",
    "D": "Development and 360-degree assessment",
    "E": "Assessment Exercise",
    "K": "Knowledge and Skills test",
    "P": "Personality and Behavior questionnaire",
    "S": "Simulation-based assessment",
}


def normalize_job_levels(levels: list) -> list:
    """Standardize job level strings."""
    normalized = []
    for lv in levels:
        lv_clean = lv.strip()
        # fuzzy match
        for valid in VALID_LEVELS:
            if valid.lower() in lv_clean.lower() or lv_clean.lower() in valid.lower():
                if valid not in normalized:
                    normalized.append(valid)
                break
        else:
            # keep as-is if unrecognized
            if lv_clean and lv_clean not in normalized:
                normalized.append(lv_clean)
    return normalized


def build_keywords(item: dict) -> list[str]:
    """
    Generate a rich keyword list for an item by:
    1. Keeping existing keywords
    2. Applying pattern-based rules
    3. Adding test type descriptions
    4. Adding job level tokens
    """
    existing = set(item.get("keywords", []))
    text = (item.get("name", "") + " " + item.get("description", "")).lower()

    # rule-based additions
    for pattern, kws in KEYWORD_RULES:
        if re.search(pattern, text, re.I):
            for kw in kws:
                existing.add(kw.lower())

    # test type descriptions
    for tt in item.get("test_types", []):
        if tt in TEST_TYPE_DESCRIPTIONS:
            existing.add(TEST_TYPE_DESCRIPTIONS[tt].lower())

    # job level tokens
    for lv in item.get("job_levels", []):
        existing.add(lv.lower())

    # name tokens (meaningful words only)
    name_words = re.findall(r'\b[a-z][a-z+#.]{2,}\b', item.get("name", "").lower())
    existing.update(name_words)

    return sorted(existing)


def build_embedding_text(item: dict) -> str:
    """
    Build the text that will be embedded for semantic search.
    Pack in all relevant information in a structured way.
    """
    parts = [
        f"Assessment: {item['name']}",
        f"Type: {', '.join(item.get('test_types', []))}",
    ]
    if item.get("description"):
        parts.append(f"Description: {item['description']}")
    if item.get("job_levels"):
        parts.append(f"Job levels: {', '.join(item['job_levels'])}")
    if item.get("keywords"):
        parts.append(f"Keywords: {', '.join(item['keywords'][:20])}")
    if item.get("duration_minutes"):
        parts.append(f"Duration: {item['duration_minutes']} minutes")
    if item.get("remote_testing"):
        parts.append("Available for remote testing")
    if item.get("adaptive_irt"):
        parts.append("Adaptive / IRT technology")
    return ". ".join(parts)


def validate_item(item: dict, index: int) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors = []
    REQUIRED = ["name", "url", "test_types", "test_type"]
    for field in REQUIRED:
        if not item.get(field):
            errors.append(f"[{index}] Missing required field: {field}")
    if item.get("url") and not item["url"].startswith("https://www.shl.com"):
        errors.append(f"[{index}] URL not from shl.com: {item['url']}")
    if item.get("test_types"):
        for tt in item["test_types"]:
            if tt not in TEST_TYPE_DESCRIPTIONS:
                errors.append(f"[{index}] Unknown test type: {tt}")
    return errors


def enrich(data: list[dict]) -> list[dict]:
    all_errors = []
    for i, item in enumerate(data):
        # normalize job levels
        item["job_levels"] = normalize_job_levels(item.get("job_levels", []))

        # build / extend keywords
        item["keywords"] = build_keywords(item)

        # build embedding text
        item["embedding_text"] = build_embedding_text(item)

        # ensure test_type is set
        if not item.get("test_type") and item.get("test_types"):
            item["test_type"] = item["test_types"][0]

        # validate
        errors = validate_item(item, i)
        all_errors.extend(errors)

    if all_errors:
        log.warning("⚠  Validation warnings:")
        for err in all_errors:
            log.warning("   %s", err)
    else:
        log.info("✅  All %d items passed validation", len(data))

    return data


if __name__ == "__main__":
    if not os.path.exists(CATALOG_IN):
        log.error("❌  Catalog not found at %s — run scrape_catalog.py first", CATALOG_IN)
        sys.exit(1)

    with open(CATALOG_IN, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d catalog items", len(data))

    enriched = enrich(data)

    with open(CATALOG_OUT, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    log.info("💾  Saved enriched catalog → %s", CATALOG_OUT)

    # Print summary stats
    print(f"\n{'='*50}")
    print(f"  Catalog enrichment complete")
    print(f"  Total assessments : {len(enriched)}")
    from collections import Counter
    types = Counter(t for item in enriched for t in item.get("test_types", []))
    type_names = {"A":"Ability","B":"Biodata/SJT","C":"Competencies",
                  "D":"Development","E":"Exercises","K":"Knowledge","P":"Personality","S":"Simulations"}
    print(f"  Test type breakdown:")
    for k, v in sorted(types.items()):
        print(f"    {k} ({type_names.get(k, k)}): {v}")
    avg_kw = sum(len(i.get("keywords",[])) for i in enriched) / len(enriched)
    print(f"  Avg keywords/item : {avg_kw:.1f}")
    print(f"{'='*50}\n")