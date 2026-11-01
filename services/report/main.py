"""Report service (FR-17, SR-05): aggregate results + provenance, no raw rows."""
from __future__ import annotations

import csv
import io
import json
from xml.sax.saxutils import escape

from fastapi import Response
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ifap_common import client
from ifap_common.appfactory import create_app
from ifap_common.errors import AppError

app = create_app("report")

LIMITATIONS = [
    "Results describe one dataset and one chronological split; they are not a guarantee of future performance.",
    "Model quality is an experimental outcome. No model family is presumed better; no quantum advantage is claimed.",
    "Quantum models run on a classical simulator (limited qubits); runtimes are not comparable to quantum hardware.",
    "Sample/synthetic datasets are labelled as such; their metrics are not real market results.",
    "Outputs are statistical estimates and not investment advice.",
]
DEFAULT_METRICS = ["RMSE", "MAE", "QLIKE", "MAPE", "DA"]


def _collect(eid: str) -> dict:
    e = client.call("registry", "GET", f"/experiments/{eid}")
    if e["status"] != "COMPLETED":
        raise AppError("EXPERIMENT_NOT_COMPLETE", "The experiment has not completed; no report is available.")
    models = {m["modelId"]: m for m in client.call("registry", "GET", "/models",
                                                     params={"experimentId": eid})["models"]}
    ds = client.call("ingestion", "GET", f"/datasets/{e['datasetId']}")
    rows = []
    for r in e["leaderboard"]:
        m = models.get(r.get("modelId"), {})
        rows.append({**r, "artifactHash": m.get("artifactHash"), "version": m.get("version"),
                     "seed": m.get("seed"), "config": m.get("config"), "libraryVersions": m.get("libraryVersions"),
                     "failureReason": m.get("failureReason") or r.get("reason")})
    return {"experimentId": eid, "generatedAt": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                            __import__("time").gmtime()),
            "task": e["task"], "target": e["target"], "horizon": e["horizon"], "primaryMetric": e["primaryMetric"],
            "seed": e["seed"], "dataset": {"datasetId": ds["datasetId"], "name": ds["name"], "type": ds["type"],
                                           "datasetHash": ds["datasetHash"], "isSample": ds.get("isSample"),
                                           "rows": ds["rows"], "dateMin": ds["dateMin"], "dateMax": ds["dateMax"]},
            "featureSetId": e["featureSetId"], "featureSetVersion": e["featureSetVersion"],
            "splitRanges": e["splitRanges"], "testTimestampHash": e["testTimestampHash"],
            "splitHash": e["splitHash"], "selection": e["selection"], "leaderboard": rows,
            "configuration": {"margin": e["margin"], "tieTolerance": e["tieTolerance"],
                              "modelConfig": e.get("modelConfig", {}), "baseline": e["baseline"],
                              "baselineAutoAdded": e.get("baselineAutoAdded")},
            "limitations": LIMITATIONS}


def _num(v, nd=6):
    return "n/a" if v is None else (f"{v:.{nd}g}" if isinstance(v, (int, float)) else str(v))


def make_csv(d: dict) -> bytes:
    def safe(x):  # neutralise spreadsheet formula injection
        s = "" if x is None else str(x)
        return "'" + s if s[:1] in "=+-@" else s
    buf = io.StringIO()
    w = csv.writer(buf)
    mets = [m for m in DEFAULT_METRICS if any(r.get("val", {}).get(m) is not None or m in r.get("val", {})
                                                for r in d["leaderboard"])]
    w.writerow(["experimentId", "task", "primaryMetric", "rank", "model", "family", "status", "isBaseline",
                "isChampion", "reasonOrNote", *[f"val_{m}" for m in mets], *[f"test_{m}" for m in mets],
                "trainTimeSec", "inferenceTimeMs", "modelId", "version", "artifactSha256", "datasetSha256",
                "featureSetVersion", "testTimestampHash"])
    for r in d["leaderboard"]:
        w.writerow([d["experimentId"], d["task"], d["primaryMetric"], r.get("rank") or "", r["model"],
                    r["family"], r["status"], r.get("isBaseline"), r.get("isChampion"),
                    safe(r.get("failureReason") or r.get("note")),
                    *[_num(r.get("val", {}).get(m)) if r.get("val") else "" for m in mets],
                    *[_num(r.get("test", {}).get(m)) if r.get("test") else "" for m in mets],
                    r.get("trainTimeSec") or "", r.get("inferenceTimeMs") or "", r.get("modelId") or "",
                    r.get("version") or "", r.get("artifactHash") or "", d["dataset"]["datasetHash"],
                    d["featureSetVersion"], d["testTimestampHash"]])
    return buf.getvalue().encode("utf-8")


def make_pdf(d: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"IFAP report {d['experimentId']}", author="IFAP")
    ss = getSampleStyleSheet()
    small = ss["BodyText"].clone("small", fontSize=8, leading=10)
    P = lambda t, st=None: Paragraph(escape(str(t)), st or ss["BodyText"])  # noqa: E731
    el = [P(f"Experiment report {d['experimentId']}", ss["Title"]),
          P(f"Generated {d['generatedAt']} - task {d['task']} - primary metric {d['primaryMetric']} (validation)")]
    if d["dataset"].get("isSample"):
        el.append(P("SAMPLE / SYNTHETIC DATA - metrics are not real market results.", ss["Heading3"]))
    el += [Spacer(1, 4 * mm), P("Selection", ss["Heading2"]), P(d["selection"]["justification"])]
    el += [Spacer(1, 3 * mm), P("Configuration and provenance", ss["Heading2"])]
    cfgrows = [["Dataset", f"{d['dataset']['name']} ({d['dataset']['datasetId']}), {d['dataset']['rows']} rows, "
                           f"{d['dataset']['dateMin']} to {d['dataset']['dateMax']}"],
               ["Dataset SHA-256", d["dataset"]["datasetHash"]], ["Feature-set version", d["featureSetVersion"]],
               ["Target", d["target"]], ["Seed", d["seed"]],
               ["Test period", f"{d['splitRanges']['test']['from']} to {d['splitRanges']['test']['to']} "
                               f"({d['splitRanges']['test']['rows']} rows)"],
               ["Test-timestamp hash", d["testTimestampHash"]],
               ["Policy", f"margin {d['configuration']['margin']:.0%}, tie tolerance "
                          f"{d['configuration']['tieTolerance']:.0%}, selection on validation data only"]]
    t = Table([[P(a, small), P(b, small)] for a, b in cfgrows], colWidths=[40 * mm, 220 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    el += [t, Spacer(1, 4 * mm), P("Leaderboard (ranked by validation " + d["primaryMetric"] + ")", ss["Heading2"])]
    mets = [m for m in DEFAULT_METRICS if any(r.get("val") and r["val"].get(m) is not None for r in d["leaderboard"])]
    head = ["#", "Model", "Family", "Status", *[f"val {m}" for m in mets], *[f"test {m}" for m in mets],
            "Train s", "Infer ms", "Artifact SHA-256 (first 16)"]
    data = [[P(h, small) for h in head]]
    for r in d["leaderboard"]:
        flag = " [CHAMPION]" if r.get("isChampion") else (" [baseline]" if r.get("isBaseline") else "")
        data.append([P(x, small) for x in [
            r.get("rank") or "-", r["model"] + flag, r["family"], r["status"] + (f" ({r['failureReason']})"
                                                                              if r["status"] != "COMPLETED" else ""),
            *[_num(r.get("val", {}).get(m), 4) for m in mets], *[_num(r.get("test", {}).get(m), 4) for m in mets],
            _num(r.get("trainTimeSec"), 3), _num(r.get("inferenceTimeMs"), 3), (r.get("artifactHash") or "n/a")[:16]]])
    t2 = Table(data, repeatRows=1)
    t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    el += [t2, Spacer(1, 4 * mm), P("Limitations", ss["Heading2"])] + [P("- " + x, small) for x in d["limitations"]]
    doc.build(el)
    return buf.getvalue()


@app.get("/reports/{eid}")
def report(eid: str, format: str = "json"):
    if format not in ("pdf", "csv", "json"):
        raise AppError("VALIDATION_FAILED", "format must be pdf, csv or json.", fields=[{"field": "format"}])
    d = _collect(eid)
    if format == "json":
        body, mt = json.dumps(d, indent=2).encode(), "application/json"
    elif format == "csv":
        body, mt = make_csv(d), "text/csv; charset=utf-8"
    else:
        body, mt = make_pdf(d), "application/pdf"
    return Response(body, media_type=mt,
                    headers={"Content-Disposition": f'attachment; filename="ifap-{eid}.{format}"'})
