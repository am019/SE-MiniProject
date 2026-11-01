"""Static repository checks: TC-24 (compose), TC-27 (only gateway published), TC-29 (no secrets in repo), TC-25 (contrast)."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIRS = sorted(p.name for p in (ROOT / "services").iterdir() if p.is_dir() and not p.name.startswith("__"))
COMPOSE_FILE = ROOT / "docker-compose.yml"
SKIP_DIRS = {".git", "var", "__pycache__", ".pytest_cache", ".venv", "venv"}


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())


# ------------------------------------------------------------------ TC-24 (static part)
def test_tc24_one_compose_service_per_service_dir(compose):
    assert set(compose["services"]) == set(SERVICE_DIRS)


def test_tc24_build_targets_and_args(compose):
    families = {"classical", "dl", "qml"}
    for name, svc in compose["services"].items():
        b = svc["build"]
        assert b["args"]["SERVICE"] == name
        assert b["target"] == (name if name in families else "base")


def test_tc24_healthchecks_volume_and_config(compose):
    for name, svc in compose["services"].items():
        assert "healthcheck" in svc, name
        assert any(str(v).startswith("ifap-data:/data") for v in svc["volumes"]), name
        assert "./config:/app/config:ro" in svc["volumes"], name
    assert "ifap-data" in compose["volumes"]


def test_tc24_stub_is_optional_profile(compose):
    assert compose["services"]["stub"]["profiles"] == ["stub"]


def test_tc24_secrets_required_from_env(compose):
    env = compose["services"]["gateway"]["environment"]
    for key in ("IFAP_JWT_SECRET", "IFAP_ADMIN_PASSWORD", "IFAP_ANALYST_PASSWORD", "IFAP_SERVICE_TOKEN"):
        assert re.fullmatch(r"\$\{%s:\?[^}]+\}" % key, str(env[key])), key


def test_tc24_model_services_resource_limited(compose):  # SR-07
    for name in ("classical", "dl", "qml"):
        svc = compose["services"][name]
        assert svc["cpus"] and svc["mem_limit"] and svc["pids_limit"], name
        limit = int(re.sub(r"\D", "", svc["mem_limit"])) * (1024 if svc["mem_limit"].endswith("g") else 1)
        assert int(svc["environment"]["IFAP_JOB_MEM_MB"]) < limit, name


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_tc24_docker_compose_config_valid(tmp_path):
    env = tmp_path / "e.env"
    env.write_text("IFAP_JWT_SECRET=a\nIFAP_SERVICE_TOKEN=b\nIFAP_ADMIN_PASSWORD=c\nIFAP_ANALYST_PASSWORD=d\n")
    r = subprocess.run(["docker", "compose", "--env-file", str(env), "--profile", "stub", "config", "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 and "compose" in r.stderr and "not a docker command" in r.stderr:
        pytest.skip("docker compose plugin missing")
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_tc24_compose_fails_without_secrets(tmp_path):
    empty = tmp_path / "e.env"
    empty.write_text("")
    r = subprocess.run(["docker", "compose", "--env-file", str(empty), "config", "-q"],
                       cwd=ROOT, capture_output=True, text=True, env={"PATH": __import__("os").environ["PATH"]})
    assert r.returncode != 0 and "init_env" in r.stderr


# ------------------------------------------------------------------ TC-27 / SR-02
def test_tc27_only_gateway_publishes_ports(compose):
    for name, svc in compose["services"].items():
        if name == "gateway":
            assert svc["ports"] == ["127.0.0.1:${IFAP_PORT:-8080}:8000"]
        else:
            assert "ports" not in svc, name


def test_tc27_internal_network_and_gateway_edge(compose):
    assert compose["networks"]["internal"]["internal"] is True
    assert set(compose["services"]["gateway"]["networks"]) == {"internal", "edge"}
    for name, svc in compose["services"].items():
        if name != "gateway":
            assert svc["networks"] == ["internal"], name


# ------------------------------------------------------------------ TC-29 / SR-04
def test_tc29_gitignore_covers_secrets_and_artifacts():
    lines = {l.strip() for l in (ROOT / ".gitignore").read_text().splitlines()}
    for must in (".env", ".secrets.json", "var/", "*.pt", "*.joblib", "*.pkl"):
        assert must in lines, must
    assert "!.env.example" in lines


def test_tc29_env_example_has_no_values():
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k != "IFAP_PORT":
                assert v == "", f"{k} must be empty in .env.example"


def test_tc29_no_secrets_committed():
    pat = re.compile(r"(?i)(secret|token|password|passwd|api[_-]?key)\s*[:=]\s*['\"][A-Za-z0-9_\-+/=]{12,}['\"]")
    hits = []
    for p in ROOT.rglob("*"):
        if (not p.is_file() or set(p.relative_to(ROOT).parts) & SKIP_DIRS or p.suffix in {".csv", ".docx", ".zip", ".png"}
                or p.name == "test_repo.py" or p.name == ".env"):
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for m in pat.finditer(text):
            hits.append(f"{p.relative_to(ROOT)}: {m.group(0)[:60]}")
    assert not hits, hits
    assert not (ROOT / ".secrets.json").exists() or ".secrets.json" in (ROOT / ".gitignore").read_text()


def test_tc29_dockerignore_excludes_env():
    assert ".env" in (ROOT / ".dockerignore").read_text().splitlines()


# ------------------------------------------------------------------ TC-25 (accessibility, static part)
def _lum(hex_):
    h = hex_.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _ratio(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_tc25_css_contrast_at_least_4_5():
    css = (ROOT / "web" / "style.css").read_text()
    v = dict(re.findall(r"--([\w-]+):(#[0-9a-fA-F]{3,6})", css))
    pairs = [("ink", "bg"), ("ink", "card"), ("muted", "bg"), ("muted", "card"), ("accent-ink", "accent"),
             ("accent", "card"), ("ok", "okbg"), ("warn", "warnbg"), ("bad", "badbg"), ("ok", "card"), ("bad", "card")]
    pairs += [("accent", "#e6f0fb"), ("#ffffff", "ink")]
    bad = []
    for fg, bg in pairs:
        f = fg if fg.startswith("#") else v[fg]
        b = bg if bg.startswith("#") else v[bg]
        r = _ratio(f, b)
        if r < 4.5:
            bad.append(f"{fg} on {bg}: {r:.2f}")
    assert not bad, bad


def test_tc25_html_basics():
    html = (ROOT / "web" / "index.html").read_text()
    assert re.search(r'<html[^>]*\blang=', html)
    assert 'name="viewport"' in html
