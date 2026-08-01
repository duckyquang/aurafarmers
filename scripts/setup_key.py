"""Store an API key locally and check that it works.

    .venv/bin/python scripts/setup_key.py

Writes .env in the repo root (git-ignored, chmod 600). Nothing is printed
back, nothing is committed, and the key never enters the event log.
"""
import getpass
import os
import re
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "sk-", "https://platform.openai.com/api-keys"),
    "anthropic": ("ANTHROPIC_API_KEY", "sk-ant-",
                  "https://platform.claude.com/settings/keys"),
}


def load_env():
    """Read .env into os.environ. Import this from any script that needs a
    key without exporting it in the shell."""
    if not ENV.exists():
        return
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def write(var, key):
    lines = []
    if ENV.exists():
        lines = [l for l in ENV.read_text().splitlines()
                 if not l.startswith(f"{var}=")]
    lines.append(f"{var}={key}")
    ENV.write_text("\n".join(lines) + "\n")
    ENV.chmod(stat.S_IRUSR | stat.S_IWUSR)      # 600: owner only


def main():
    provider = (sys.argv[1] if len(sys.argv) > 1 else "openai").lower()
    if provider not in PROVIDERS:
        print(f"usage: setup_key.py [{'|'.join(PROVIDERS)}]")
        return 1
    var, prefix, url = PROVIDERS[provider]

    print(f"Setting up {provider}.")
    print(f"  1. Sign in and create a key at {url}")
    print("  2. Copy it (you only get to see it once)")
    print("  3. Paste below - it will not be echoed, and it goes to")
    print(f"     {ENV} (git-ignored, readable only by you).\n")
    key = getpass.getpass(f"{var}: ").strip()
    if not key:
        print("nothing entered, aborted.")
        return 1
    if not key.startswith(prefix):
        print(f"warning: {provider} keys normally start with '{prefix}'. "
              "Storing anyway.")
    if not re.fullmatch(r"[A-Za-z0-9_\-]{20,}", key):
        print("warning: that does not look like a key. Storing anyway.")
    write(var, key)
    os.environ[var] = key
    print(f"\nsaved to .env ({oct(ENV.stat().st_mode)[-3:]}). Verifying...")

    os.environ["AURAFARMERS_PROVIDER"] = provider
    import importlib

    from sim import llm
    importlib.reload(llm)
    try:
        out = llm.complete(llm.ROUTINE_MODEL, "Reply with one word.",
                           "Say OK.", 16)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {str(e)[:200]}")
        print("The key is saved; fix the error above and re-run to re-check.")
        return 1
    print(f"works. {llm.ROUTINE_MODEL} replied: {str(out).strip()[:40]!r}")
    print(f"cost of that call: ${llm.spend():.6f}")
    print(f"\nScripts pick this up automatically. To use it in your own "
          f"shell:\n  export {var}=$(grep '^{var}=' .env | cut -d= -f2-)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
