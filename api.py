"""
api.py — Flask REST API for Cybersecurity Intrusion Detection
=============================================================
Serves pre-computed outputs from outputs/ to the React frontend.
Does NOT retrain the model on every request.

Run:  .venv/Scripts/python api.py
      API available at http://localhost:5000
"""

import os, json
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow React dev server (port 5173) to call this API

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PREDS_CSV = os.path.join(BASE_DIR, "outputs", "predictions.csv")
METRICS_F = os.path.join(BASE_DIR, "outputs", "metrics.json")
CM_CSV    = os.path.join(BASE_DIR, "outputs", "confusion_matrix.csv")

# ── helpers ──────────────────────────────────────────────────────────────────
def outputs_ready():
    return os.path.exists(PREDS_CSV) and os.path.exists(METRICS_F)

def load_preds():
    df = pd.read_csv(PREDS_CSV)
    # Ensure numeric types are native Python (JSON serialisable)
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].round(6)
    return df

def load_metrics():
    with open(METRICS_F) as f:
        return json.load(f)

# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    ready = outputs_ready()
    return jsonify({
        "status":  "ready" if ready else "no_outputs",
        "message": "Outputs loaded" if ready else "Run python main.py first"
    })


@app.route("/api/metrics")
def metrics():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    return jsonify(load_metrics())


@app.route("/api/kpis")
def kpis():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df = load_preds()
    m  = load_metrics()
    total      = len(df)
    suspicious = int((df["predicted_status"] == "Suspicious").sum())
    benign     = int((df["predicted_status"] == "Benign").sum())
    has_bytes  = "sbytes" in df.columns and "dbytes" in df.columns
    total_bytes = int(df["sbytes"].fillna(0).sum() + df["dbytes"].fillna(0).sum()) if has_bytes else 0
    alert_rate  = round(suspicious / total * 100, 2) if total else 0
    return jsonify({
        "total_records":  total,
        "suspicious":     suspicious,
        "benign":         benign,
        "total_bytes":    total_bytes,
        "total_bytes_gb": round(total_bytes / 1e9, 3),
        "alert_rate_pct": alert_rate,
        "accuracy":       m.get("accuracy", 0),
        "roc_auc":        m.get("roc_auc", 0),
        "attack_f1":      m.get("attack_f1", 0),
    })


@app.route("/api/predictions")
def predictions():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df = load_preds()

    # --- query params ---
    status  = request.args.get("status",   "All")
    risk    = request.args.get("risk",     "All")
    proto   = request.args.get("proto",    "All")
    page    = int(request.args.get("page", 1))
    per_page= int(request.args.get("per_page", 50))

    if status != "All" and "predicted_status" in df.columns:
        df = df[df["predicted_status"] == status]
    if risk != "All" and "risk_level" in df.columns:
        df = df[df["risk_level"] == risk]
    if proto != "All" and "proto" in df.columns:
        df = df[df["proto"] == proto]

    total_filtered = len(df)
    start = (page - 1) * per_page
    page_df = df.iloc[start:start + per_page]

    return jsonify({
        "total":    total_filtered,
        "page":     page,
        "per_page": per_page,
        "pages":    max(1, -(-total_filtered // per_page)),  # ceil div
        "data":     page_df.fillna("").to_dict(orient="records"),
    })


@app.route("/api/filter-options")
def filter_options():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df = load_preds()
    return jsonify({
        "statuses":   sorted(df["predicted_status"].dropna().unique().tolist()) if "predicted_status" in df.columns else [],
        "risk_levels":["High", "Medium", "Low"],
        "protocols":  sorted(df["proto"].dropna().unique().tolist()) if "proto" in df.columns else [],
    })


@app.route("/api/charts/risk-distribution")
def risk_distribution():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df = load_preds()
    counts = df["risk_level"].value_counts().reindex(["High","Medium","Low"]).fillna(0)
    return jsonify([{"risk": k, "count": int(v)} for k, v in counts.items()])


@app.route("/api/charts/attack-categories")
def attack_categories():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df = load_preds()
    if "attack_cat" not in df.columns:
        return jsonify({"categories": []})
    cats = df["attack_cat"].value_counts().head(15)
    data = [{"attack_cat": k, "count": int(v)} for k, v in cats.items()]
    return jsonify({"categories": data})


@app.route("/api/charts/attack-categories-suspicious")
def attack_categories_suspicious():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df = load_preds()
    if "attack_cat" not in df.columns:
        return jsonify({"categories": []})
    susp = df[df["predicted_status"] == "Suspicious"]
    cats = susp["attack_cat"].value_counts().head(15)
    data = [{"attack_cat": k, "count": int(v)} for k, v in cats.items()]
    return jsonify({"categories": data})


@app.route("/api/charts/protocol-traffic")
def protocol_traffic():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df = load_preds()
    if "proto" not in df.columns:
        return jsonify({"protocols": []})
    has_bytes = "sbytes" in df.columns and "dbytes" in df.columns
    result = []
    for proto, group in df.groupby("proto"):
        rec = {
            "proto": proto,
            "count": int(len(group)),
            "total_src_bytes": int(group["sbytes"].fillna(0).sum()) if has_bytes else 0,
            "total_dst_bytes": int(group["dbytes"].fillna(0).sum()) if has_bytes else 0,
        }
        result.append(rec)
    result.sort(key=lambda x: x["count"], reverse=True)
    return jsonify({"protocols": result[:15]})


@app.route("/api/charts/protocol-status")
def protocol_status():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df = load_preds()
    if "proto" not in df.columns or "predicted_status" not in df.columns:
        return jsonify({"protocols": []})
    top_protos = df["proto"].value_counts().head(12).index.tolist()
    sub = df[df["proto"].isin(top_protos)]
    result = []
    for proto, group in sub.groupby("proto"):
        benign_c = int((group["predicted_status"] == "Benign").sum())
        susp_c   = int((group["predicted_status"] == "Suspicious").sum())
        result.append({"proto": proto, "benign_count": benign_c, "suspicious_count": susp_c})
    result.sort(key=lambda x: x["benign_count"] + x["suspicious_count"], reverse=True)
    return jsonify({"protocols": result})


@app.route("/api/charts/prediction-split")
def prediction_split():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df = load_preds()
    suspicious = int((df["predicted_status"] == "Suspicious").sum())
    benign     = int((df["predicted_status"] == "Benign").sum())
    return jsonify([
        {"name": "Benign",     "value": benign},
        {"name": "Suspicious", "value": suspicious},
    ])


@app.route("/api/charts/actual-vs-predicted")
def actual_vs_predicted():
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df  = load_preds()
    m   = load_metrics()
    cm  = m.get("confusion_matrix", {})
    actual_normal  = int((df["label"] == 0).sum()) if "label" in df.columns else 0
    actual_attack  = int((df["label"] == 1).sum()) if "label" in df.columns else 0
    pred_benign    = int((df["predicted_status"] == "Benign").sum())
    pred_suspicious= int((df["predicted_status"] == "Suspicious").sum())
    return jsonify({
        "actual_normal":   actual_normal,
        "actual_attack":   actual_attack,
        "pred_benign":     pred_benign,
        "pred_suspicious": pred_suspicious,
        "confusion_matrix": cm,
    })


@app.route("/api/charts/prob-distribution")
def prob_distribution():
    """Histogram buckets for attack_prob of suspicious predictions."""
    if not outputs_ready():
        return jsonify({"error": "Run python main.py first"}), 503
    df = load_preds()
    if "attack_prob" not in df.columns:
        return jsonify([])
    susp = df[df["predicted_status"] == "Suspicious"]["attack_prob"].dropna()
    # 20 buckets 0–1
    buckets = []
    for i in range(20):
        lo = i * 0.05
        hi = lo + 0.05
        cnt = int(((susp >= lo) & (susp < hi)).sum())
        buckets.append({"range": f"{lo:.2f}-{hi:.2f}", "lo": round(lo,2), "count": cnt})
    return jsonify(buckets)


if __name__ == "__main__":
    if not outputs_ready():
        print("=" * 55)
        print("WARNING: outputs not found.")
        print("Run  python main.py  first to generate them.")
        print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
