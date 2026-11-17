"""TC-24 / NFR-07: clean docker compose deployment, timed, then smoke test.
Run from the repo root:  python3 scripts/tc24.py [--cold]
Appends REAL measured results to docs/TEST_RESULTS.md. Does not print or store passwords."""
import datetime, platform, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def run(cmd, **kw):
    print("\n$", " ".join("***" if i and cmd[i-1]=="--password" else c for i, c in enumerate(cmd))); return subprocess.run(cmd, cwd=ROOT, **kw)

if run(["docker", "info"], capture_output=True).returncode != 0:
    sys.exit("Docker is not running. Open Docker Desktop, wait until it says 'Engine running', then re-run. Nothing was changed.")
run([sys.executable, "scripts/init_env.py", "--force"], stdout=subprocess.DEVNULL)  # new secrets, not echoed
pw = next(l.split("=", 1)[1] for l in (ROOT / ".env").read_text().splitlines() if l.startswith("IFAP_ADMIN_PASSWORD="))
if "--cold" in sys.argv:  # true clean-build timing: remove images and build cache
    run(["docker", "compose", "down", "-v", "--rmi", "all"])
    run(["docker", "builder", "prune", "-a", "-f"])
else:
    run(["docker", "compose", "down", "-v"])
try:
    import httpx  # needed by scripts/smoke.py
except ImportError:
    run([sys.executable, "-m", "pip", "install", "httpx", "--break-system-packages"])  # Homebrew Python blocks plain pip

t0 = time.time()
up = run(["docker", "compose", "up", "--build", "-d", "--wait", "--wait-timeout", "1800"])
build_min = (time.time() - t0) / 60
smoke = run([sys.executable, "scripts/smoke.py", "--password", pw]) if up.returncode == 0 else None
smoke_ok = smoke is not None and smoke.returncode == 0
ps = subprocess.run(["docker", "compose", "ps", "--format", "{{.Service}}: {{.Status}}"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
ver = subprocess.run(["docker", "--version"], capture_output=True, text=True).stdout.strip()

out = ROOT / "docs" / "TEST_RESULTS.md"
out.parent.mkdir(exist_ok=True)
block = f"""
## TC-24 / NFR-07: one-command Docker Compose deployment

- Date: {datetime.date.today().isoformat()}
- Machine: {platform.system()} {platform.release()}, {platform.machine()}; {ver}
- Command: `docker compose up --build -d --wait` (after `docker compose down -v`, fresh `.env`; {"cold build: images and build cache removed first" if "--cold" in sys.argv else "warm build: image cache present"})
- Result: {"all services healthy" if up.returncode == 0 else "FAILED to become healthy"}
- Build + start time: {build_min:.1f} minutes (requirement: within 30 minutes) -> {"PASS" if up.returncode == 0 and build_min <= 30 else "FAIL"}
- Smoke test (`scripts/smoke.py`): {"PASS" if smoke_ok else "FAIL / not run"}

Service status:
```
{ps}
```
"""
with open(out, "a", encoding="utf-8") as f:
    f.write(("" if out.exists() and out.stat().st_size else "# Test results\n") + block)
print(f"\nBuild+start {build_min:.1f} min; smoke {'PASS' if smoke_ok else 'FAIL'}. Results appended to {out}")
sys.exit(0 if up.returncode == 0 and smoke_ok else 1)