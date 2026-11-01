"""Create .env with random secrets (SR-04). Standard library only.

    python scripts/init_env.py [--force]
"""
import os
import secrets
import sys
from pathlib import Path

target = Path(__file__).resolve().parents[1] / ".env"
if target.exists() and "--force" not in sys.argv:
    print(f"{target} already exists (use --force to regenerate). Existing credentials are kept.")
    sys.exit(0)
vals = {"IFAP_JWT_SECRET": secrets.token_urlsafe(48), "IFAP_SERVICE_TOKEN": secrets.token_urlsafe(32),
        "IFAP_ADMIN_PASSWORD": secrets.token_urlsafe(9), "IFAP_ANALYST_PASSWORD": secrets.token_urlsafe(9),
        "IFAP_PORT": "8080"}
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    f.write("\n".join(f"{k}={v}" for k, v in vals.items()) + "\n")
print(f"Wrote {target}\n  Administrator: admin   / {vals['IFAP_ADMIN_PASSWORD']}\n"
      f"  Analyst:       analyst / {vals['IFAP_ANALYST_PASSWORD']}")
