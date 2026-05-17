"""
app/llm_client.py — Multi-provider LLM client.
Recommended: Groq (free, fast) → console.groq.com
"""
import logging, os
from tenacity import retry, stop_after_attempt, wait_exponential
log = logging.getLogger(__name__)

def _detect_provider():
    p = os.getenv("LLM_PROVIDER","").lower().strip()
    if p: return p
    if os.getenv("GROQ_API_KEY","").startswith("gsk_"):     return "groq"
    if os.getenv("GEMINI_API_KEY","").startswith("AIza"):   return "gemini"
    if os.getenv("ANTHROPIC_API_KEY","").startswith("sk-"): return "anthropic"
    return "groq"

PROVIDER = _detect_provider()
log.info("LLM provider: %s", PROVIDER)

def _call_groq(system, messages, max_tokens):
    from groq import Groq
    key = os.getenv("GROQ_API_KEY","")
    if not key: raise RuntimeError("GROQ_API_KEY missing. Get free key at console.groq.com")
    r = Groq(api_key=key).chat.completions.create(
        model=os.getenv("CLAUDE_MODEL","llama-3.3-70b-versatile"),
        messages=[{"role":"system","content":system}]+messages,
        max_tokens=max_tokens, temperature=0.2)
    return r.choices[0].message.content

def _call_gemini(system, messages, max_tokens):
    key = os.getenv("GEMINI_API_KEY","")
    if not key: raise RuntimeError("GEMINI_API_KEY missing. Get free key at aistudio.google.com")
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        model  = os.getenv("CLAUDE_MODEL","gemini-2.0-flash")
        contents = [types.Content(
            role="user" if m["role"]=="user" else "model",
            parts=[types.Part(text=m["content"])]) for m in messages]
        cfg = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens, temperature=0.2)
        return client.models.generate_content(model=model, contents=contents, config=cfg).text
    except ImportError:
        import google.generativeai as g
        g.configure(api_key=key)
        m = g.GenerativeModel(os.getenv("CLAUDE_MODEL","gemini-1.5-flash"), system_instruction=system)
        hist = [{"role":"user" if x["role"]=="user" else "model","parts":[x["content"]]} for x in messages[:-1]]
        return m.start_chat(history=hist).send_message(messages[-1]["content"]).text

def _call_anthropic(system, messages, max_tokens):
    import anthropic
    key = os.getenv("ANTHROPIC_API_KEY","")
    if not key: raise RuntimeError("ANTHROPIC_API_KEY missing in .env")
    r = anthropic.Anthropic(api_key=key).messages.create(
        model=os.getenv("CLAUDE_MODEL","claude-haiku-4-5-20251001"),
        max_tokens=max_tokens, system=system, messages=messages)
    return r.content[0].text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1,max=6), reraise=True)
def call_llm(system, messages, max_tokens=1024):
    """Call configured LLM provider. Retries 3x."""
    try:
        if PROVIDER=="groq":      return _call_groq(system, messages, max_tokens)
        elif PROVIDER=="gemini":  return _call_gemini(system, messages, max_tokens)
        else:                     return _call_anthropic(system, messages, max_tokens)
    except Exception as e:
        log.error("LLM call failed [%s]: %s", PROVIDER, e)
        raise