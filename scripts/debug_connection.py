"""
scripts/debug_connection.py
===========================
Run this to pinpoint exactly what is failing.
Usage:  python scripts/debug_connection.py
"""
import os, sys
from dotenv import load_dotenv
load_dotenv()

print("=" * 55)
print("  SHL Recommender — Connection Debug")
print("=" * 55)

# ── 1. Detect provider ───────────────────────────────────
provider = os.getenv("LLM_PROVIDER","").lower()
gemini_key    = os.getenv("GEMINI_API_KEY","")
groq_key      = os.getenv("GROQ_API_KEY","")
anthropic_key = os.getenv("ANTHROPIC_API_KEY","")

if not provider:
    if gemini_key:    provider = "gemini"
    elif groq_key:    provider = "groq"
    else:             provider = "anthropic"

print(f"\n── Provider: {provider.upper()}")

if provider == "gemini":
    if not gemini_key:
        print("❌  GEMINI_API_KEY not set in .env")
        sys.exit(1)
    print(f"✅  GEMINI_API_KEY: {gemini_key[:12]}...")
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        r = model.generate_content("Say OK in one word")
        print(f"✅  Gemini API call OK: {r.text.strip()!r}")
    except ImportError:
        print("❌  google-generativeai not installed")
        print("    Fix: pip install google-generativeai")
        sys.exit(1)
    except Exception as e:
        print(f"❌  Gemini call failed: {e}")
        sys.exit(1)

elif provider == "groq":
    if not groq_key:
        print("❌  GROQ_API_KEY not set in .env")
        sys.exit(1)
    print(f"✅  GROQ_API_KEY: {groq_key[:12]}...")
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":"Say OK"}],
            max_tokens=5,
        )
        print(f"✅  Groq API call OK: {r.choices[0].message.content!r}")
    except ImportError:
        print("❌  groq not installed — Fix: pip install groq")
        sys.exit(1)
    except Exception as e:
        print(f"❌  Groq call failed: {e}")
        sys.exit(1)

else:  # anthropic
    if not anthropic_key:
        print("❌  ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)
    print(f"✅  ANTHROPIC_API_KEY: {anthropic_key[:16]}...")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        r = client.messages.create(
            model=os.getenv("CLAUDE_MODEL","claude-haiku-4-5-20251001"),
            max_tokens=10,
            messages=[{"role":"user","content":"Say OK"}],
        )
        print(f"✅  Anthropic API call OK: {r.content[0].text!r}")
    except Exception as e:
        print(f"❌  Anthropic call failed: {e}")
        sys.exit(1)

# ── 2. Check FAISS files ─────────────────────────────────
print("\n── Catalog files:")
for path in ["catalog/shl_index.faiss","catalog/shl_meta.json","catalog/shl_catalog.json"]:
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    print(f"    {'✅' if exists else '❌'}  {path} ({size:,} bytes)")
if not os.path.exists("catalog/shl_index.faiss"):
    print("    Fix: python scripts/build_index.py")
    sys.exit(1)

# ── 3. End-to-end test ───────────────────────────────────
print("\n── End-to-end agent test...")
try:
    from app.retriever import retriever
    retriever.load()
    from app.models import ChatRequest, Message
    from app.agent import run_agent
    req  = ChatRequest(messages=[Message(role="user", content="I am hiring a Java developer, mid level")])
    resp = run_agent(req)
    print(f"✅  Reply: {resp.reply[:80]}")
    print(f"    Recs : {len(resp.recommendations)}")
    for r in resp.recommendations[:3]:
        print(f"      - {r.name} [{r.test_type}]")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*55)
print("  All checks passed ✅  Run: uvicorn app.main:app --reload")
print("="*55)