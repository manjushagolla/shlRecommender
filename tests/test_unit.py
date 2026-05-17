"""
tests/test_unit.py
==================
Fast unit tests — no server needed.
Tests schema validator, Recall@K math, and probe assertions.

Usage:
    python tests/test_unit.py
"""

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.evaluate import validate_schema, recall_at_k, BEHAVIOR_PROBES

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅  {name}")
        PASS += 1
    else:
        print(f"  ❌  {name}  {detail}")
        FAIL += 1

print("=" * 55)
print("  Unit Tests — Schema + Recall + Probes")
print("=" * 55)

# ── schema validation ─────────────────────────────────────
print("\n── Schema Validation")

good = {"reply": "Here are recs", "recommendations": [], "end_of_conversation": False}
check("valid response has 0 errors", len(validate_schema(good)) == 0)

missing_reply = {"recommendations": [], "end_of_conversation": False}
check("missing reply detected", len(validate_schema(missing_reply)) > 0)

bad_url = {
    "reply": "ok", "end_of_conversation": False,
    "recommendations": [{"name":"X","url":"https://evil.com/test","test_type":"A"}]
}
check("non-shl URL detected", len(validate_schema(bad_url)) > 0)

too_many = {
    "reply": "ok", "end_of_conversation": False,
    "recommendations": [{"name":f"T{i}","url":"https://www.shl.com/x","test_type":"A"} for i in range(11)]
}
check(">10 recommendations detected", len(validate_schema(too_many)) > 0)

missing_rec_field = {
    "reply": "ok", "end_of_conversation": False,
    "recommendations": [{"name": "Java 8 (New)", "test_type": "K"}]  # missing url
}
check("missing rec field detected", len(validate_schema(missing_rec_field)) > 0)

not_bool_eoc = {"reply":"ok","recommendations":[],"end_of_conversation":"false"}
check("non-bool end_of_conversation detected", len(validate_schema(not_bool_eoc)) > 0)

# ── recall@k math ─────────────────────────────────────────
print("\n── Recall@10 Math")

recs = [
    {"name": "Java 8 (New)",   "url": "https://www.shl.com/x", "test_type": "K"},
    {"name": "OPQ32r",         "url": "https://www.shl.com/x", "test_type": "P"},
    {"name": "Verify - Numerical Reasoning", "url": "https://www.shl.com/x", "test_type": "A"},
]
expected = ["Java 8 (New)", "OPQ32r", "Spring (New)"]
score = recall_at_k(recs, expected)
check(f"recall = 2/3 = 0.667  (got {score:.3f})", abs(score - 2/3) < 0.01)

perfect = recall_at_k(
    [{"name": e, "url":"https://www.shl.com","test_type":"K"} for e in expected],
    expected
)
check(f"perfect recall = 1.0  (got {perfect:.3f})", perfect == 1.0)

zero = recall_at_k(
    [{"name":"Unrelated Test","url":"https://www.shl.com","test_type":"A"}],
    expected
)
check(f"zero recall = 0.0  (got {zero:.3f})", zero == 0.0)

empty_expected = recall_at_k(recs, [])
check(f"empty expected = 1.0  (got {empty_expected:.3f})", empty_expected == 1.0)

# case insensitive
ci = recall_at_k(
    [{"name":"java 8 (new)","url":"https://www.shl.com","test_type":"K"}],
    ["Java 8 (New)"]
)
check(f"case-insensitive matching works  (got {ci:.3f})", ci == 1.0)

# ── probe assertion logic ─────────────────────────────────
print("\n── Behavior Probe Assertions (dry run)")

probe_cases = {
    "probe_vague_no_immediate_rec": {
        "reply": "Could you tell me more about the role?",
        "recommendations": [],
        "end_of_conversation": False,
    },
    "probe_max_10_recs": {
        "reply": "Here are some assessments",
        "recommendations": [{"name":f"T{i}","url":"https://www.shl.com/x","test_type":"A"} for i in range(8)],
        "end_of_conversation": False,
    },
    "probe_schema_compliance": {
        "reply": "Got it",
        "recommendations": [],
        "end_of_conversation": False,
    },
    "probe_url_catalog_only": {
        "reply": "Here you go",
        "recommendations": [{"name":"Java 8 (New)","url":"https://www.shl.com/solutions/products/product-catalog/view/java-8-new/","test_type":"K"}],
        "end_of_conversation": False,
    },
}

for probe in BEHAVIOR_PROBES:
    if probe["id"] in probe_cases:
        passed = probe["assert"](probe_cases[probe["id"]])
        check(f"{probe['id']}", passed)

# ── summary ───────────────────────────────────────────────
print(f"\n{'═'*55}")
total = PASS + FAIL
print(f"  {PASS}/{total} unit tests passed  {'✅' if FAIL==0 else '❌'}")
if FAIL > 0:
    print(f"  {FAIL} tests FAILED — fix before running evaluate.py")
print(f"{'═'*55}")
sys.exit(0 if FAIL == 0 else 1)