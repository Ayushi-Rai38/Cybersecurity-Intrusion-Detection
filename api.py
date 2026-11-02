"""
api.py — FastAPI Backend for Cybersecurity Intrusion Detection
==============================================================
Serves pre-computed outputs from outputs/ to the React frontend.
Does NOT retrain the model on dashboard refresh.

Run:  .venv/Scripts/python api.py
      API available at http://localhost:5000
      Docs at         http://localhost:5000/docs
"""

import os, json, sys
import pandas as pd
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ── Env vars for Spark/Java ──────────────────────────────────────────────────
os.environ.setdefault("JAVA_HOME",   r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot")
os.environ.setdefault("SPARK_HOME",  r"C:\spark-4.2.0-bin-hadoop3")
os.environ.setdefault("HADOOP_HOME", r"C:\hadoopsetup\hadoop-3.2.4")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PREDS_CSV  = os.path.join(BASE_DIR, "outputs", "predictions.csv")
METRICS_F  = os.path.join(BASE_DIR, "outputs", "metrics.json")
CM_CSV     = os.path.join(BASE_DIR, "outputs", "confusion_matrix.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "intrusion_rf_model")

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="CyberShield IDS API",
    description="Cybersecurity Intrusion Detection — PySpark MLlib + UNSW-NB15",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache ──────────────────────────────────────────────────────────
_cache = {}

def outputs_ready():
    return os.path.exists(PREDS_CSV) and os.path.exists(METRICS_F)

def load_preds() -> pd.DataFrame:
    if "_df" not in _cache:
        df = pd.read_csv(PREDS_CSV)
        for col in df.select_dtypes(include="float64").columns:
            df[col] = df[col].round(6)
        _cache["_df"] = df
    return _cache["_df"]

def load_metrics() -> dict:
    if "_metrics" not in _cache:
        with open(METRICS_F) as f:
            _cache["_metrics"] = json.load(f)
    return _cache["_metrics"]

def require_outputs():
    if not outputs_ready():
        raise HTTPException(status_code=503, detail="Run python main.py first to generate outputs.")

# ── Request schema for /predict ──────────────────────────────────────────────
class TrafficRecord(BaseModel):
    proto:    str   = "tcp"
    service:  str   = "-"
    state:    str   = "FIN"
    dur:      float = 0.0
    sbytes:   float = 0.0
    dbytes:   float = 0.0
    spkts:    int   = 0
    dpkts:    int   = 0
    rate:     float = 0.0
    sttl:     int   = 64
    dttl:     int   = 0
    sload:    float = 0.0
    dload:    float = 0.0
    sloss:    int   = 0
    dloss:    int   = 0
    sinpkt:   float = 0.0
    dinpkt:   float = 0.0
    sjit:     float = 0.0
    djit:     float = 0.0
    swin:     int   = 0
    stcpb:    int   = 0
    dtcpb:    int   = 0
    dwin:     int   = 0
    tcprtt:   float = 0.0
    synack:   float = 0.0
    ackdat:   float = 0.0
    smean:    int   = 0
    dmean:    int   = 0
    trans_depth:       int   = 0
    response_body_len: int   = 0
    ct_srv_src:        int   = 0
    ct_state_ttl:      int   = 0
    ct_dst_ltm:        int   = 0
    ct_src_dport_ltm:  int   = 0
    ct_dst_sport_ltm:  int   = 0
    ct_dst_src_ltm:    int   = 0
    is_ftp_login:      int   = 0
    ct_ftp_cmd:        int   = 0
    ct_flw_http_mthd:  int   = 0
    ct_src_ltm:        int   = 0
    ct_srv_dst:        int   = 0
    is_sm_ips_ports:   int   = 0

# ── Spark model cache ────────────────────────────────────────────────────────
_spark_model = None
_spark_session = None

def get_spark_model():
    global _spark_model, _spark_session
    if _spark_model is not None:
        return _spark_model, _spark_session
    if not os.path.exists(MODEL_DIR):
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_DIR}. Run python main.py first."
        )
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml import PipelineModel
        spark = (
            SparkSession.builder
            .appName("CyberShieldPredict")
            .master("local[2]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.hadoop.fs.defaultFS", "file:///")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        model = PipelineModel.load(MODEL_DIR)
        _spark_model = model
        _spark_session = spark
        return model, spark
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to load Spark model: {e}")

# ════════════════════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    ready = outputs_ready()
    return {
        "status":  "ready" if ready else "no_outputs",
        "message": "Outputs loaded" if ready else "Run python main.py first",
        "model_saved": os.path.exists(MODEL_DIR),
    }


@app.get("/api/metrics")
def metrics():
    require_outputs()
    return load_metrics()


@app.get("/api/kpis")
def kpis():
    require_outputs()
    df = load_preds()
    m  = load_metrics()
    total      = len(df)
    suspicious = int((df["predicted_status"] == "Suspicious").sum())
    benign     = int((df["predicted_status"] == "Benign").sum())
    has_bytes  = "sbytes" in df.columns and "dbytes" in df.columns
    total_bytes = int(df["sbytes"].fillna(0).sum() + df["dbytes"].fillna(0).sum()) if has_bytes else 0
    alert_rate  = round(suspicious / total * 100, 2) if total else 0
    return {
        "total_records":  total,
        "suspicious":     suspicious,
        "benign":         benign,
        "total_bytes":    total_bytes,
        "total_bytes_gb": round(total_bytes / 1e9, 3),
        "alert_rate_pct": alert_rate,
        "accuracy":       m.get("accuracy", 0),
        "roc_auc":        m.get("roc_auc", 0),
        "attack_f1":      m.get("attack_f1", 0),
    }


@app.get("/api/predictions")
def predictions(
    status:   str = Query("All"),
    risk:     str = Query("All"),
    proto:    str = Query("All"),
    page:     int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=500),
):
    require_outputs()
    df = load_preds()

    if status != "All" and "predicted_status" in df.columns:
        df = df[df["predicted_status"] == status]
    if risk   != "All" and "risk_level"        in df.columns:
        df = df[df["risk_level"] == risk]
    if proto  != "All" and "proto"             in df.columns:
        df = df[df["proto"] == proto]

    total_filtered = len(df)
    start   = (page - 1) * per_page
    page_df = df.iloc[start:start + per_page]
    pages   = max(1, -(-total_filtered // per_page))  # ceil div

    return {
        "total":    total_filtered,
        "page":     page,
        "per_page": per_page,
        "pages":    pages,
        "data":     page_df.fillna("").to_dict(orient="records"),
    }


@app.get("/api/filter-options")
def filter_options():
    require_outputs()
    df = load_preds()
    return {
        "statuses":    sorted(df["predicted_status"].dropna().unique().tolist()) if "predicted_status" in df.columns else [],
        "risk_levels": ["High", "Medium", "Low"],
        "protocols":   sorted(df["proto"].dropna().unique().tolist())            if "proto"             in df.columns else [],
    }


@app.get("/api/charts/risk-distribution")
def risk_distribution():
    require_outputs()
    df = load_preds()
    counts = df["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    return [{"risk": k, "count": int(v)} for k, v in counts.items()]


@app.get("/api/charts/attack-categories")
def attack_categories():
    require_outputs()
    df = load_preds()
    if "attack_cat" not in df.columns:
        return {"categories": []}
    cats = df["attack_cat"].value_counts().head(15)
    return {"categories": [{"attack_cat": k, "count": int(v)} for k, v in cats.items()]}


@app.get("/api/charts/attack-categories-suspicious")
def attack_categories_suspicious():
    require_outputs()
    df = load_preds()
    if "attack_cat" not in df.columns:
        return {"categories": []}
    susp = df[df["predicted_status"] == "Suspicious"]
    cats = susp["attack_cat"].value_counts().head(15)
    return {"categories": [{"attack_cat": k, "count": int(v)} for k, v in cats.items()]}


@app.get("/api/charts/protocol-traffic")
def protocol_traffic():
    require_outputs()
    df = load_preds()
    if "proto" not in df.columns:
        return {"protocols": []}
    has_bytes = "sbytes" in df.columns and "dbytes" in df.columns
    result = []
    for proto, group in df.groupby("proto"):
        rec = {
            "proto":           proto,
            "count":           int(len(group)),
            "total_src_bytes": int(group["sbytes"].fillna(0).sum()) if has_bytes else 0,
            "total_dst_bytes": int(group["dbytes"].fillna(0).sum()) if has_bytes else 0,
        }
        result.append(rec)
    result.sort(key=lambda x: x["count"], reverse=True)
    return {"protocols": result[:15]}


@app.get("/api/charts/protocol-status")
def protocol_status():
    require_outputs()
    df = load_preds()
    if "proto" not in df.columns or "predicted_status" not in df.columns:
        return {"protocols": []}
    top_protos = df["proto"].value_counts().head(12).index.tolist()
    sub = df[df["proto"].isin(top_protos)]
    result = []
    for proto, group in sub.groupby("proto"):
        benign_c = int((group["predicted_status"] == "Benign").sum())
        susp_c   = int((group["predicted_status"] == "Suspicious").sum())
        result.append({"proto": proto, "benign_count": benign_c, "suspicious_count": susp_c})
    result.sort(key=lambda x: x["benign_count"] + x["suspicious_count"], reverse=True)
    return {"protocols": result}


@app.get("/api/charts/prediction-split")
def prediction_split():
    require_outputs()
    df = load_preds()
    suspicious = int((df["predicted_status"] == "Suspicious").sum())
    benign     = int((df["predicted_status"] == "Benign").sum())
    return [
        {"name": "Benign",     "value": benign},
        {"name": "Suspicious", "value": suspicious},
    ]


@app.get("/api/charts/actual-vs-predicted")
def actual_vs_predicted():
    require_outputs()
    df = load_preds()
    m  = load_metrics()
    cm = m.get("confusion_matrix", {})
    actual_normal   = int((df["label"] == 0).sum()) if "label" in df.columns else 0
    actual_attack   = int((df["label"] == 1).sum()) if "label" in df.columns else 0
    pred_benign     = int((df["predicted_status"] == "Benign").sum())
    pred_suspicious = int((df["predicted_status"] == "Suspicious").sum())
    return {
        "actual_normal":    actual_normal,
        "actual_attack":    actual_attack,
        "pred_benign":      pred_benign,
        "pred_suspicious":  pred_suspicious,
        "confusion_matrix": cm,
    }


@app.get("/api/charts/prob-distribution")
def prob_distribution():
    require_outputs()
    df = load_preds()
    if "attack_prob" not in df.columns:
        return []
    susp = df[df["predicted_status"] == "Suspicious"]["attack_prob"].dropna()
    buckets = []
    for i in range(20):
        lo  = i * 0.05
        hi  = lo + 0.05
        cnt = int(((susp >= lo) & (susp < hi)).sum())
        buckets.append({"range": f"{lo:.2f}-{hi:.2f}", "lo": round(lo, 2), "count": cnt})
    return buckets


@app.post("/api/predict")
def predict(record: TrafficRecord):
    """
    Predict whether a single network connection is Normal or Attack.
    Loads the saved PySpark PipelineModel (lazy, cached after first call).
    """
    model, spark = get_spark_model()

    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, IntegerType, LongType
    )

    row_dict = record.dict()

    # Build a Spark DataFrame with the single record
    # The schema must match the columns the pipeline was trained on
    # (string cols first, then numeric)
    schema = StructType([
        StructField("proto",              StringType(),  True),
        StructField("service",            StringType(),  True),
        StructField("state",              StringType(),  True),
        StructField("dur",                DoubleType(),  True),
        StructField("sbytes",             DoubleType(),  True),
        StructField("dbytes",             DoubleType(),  True),
        StructField("spkts",              DoubleType(),  True),
        StructField("dpkts",              DoubleType(),  True),
        StructField("rate",               DoubleType(),  True),
        StructField("sttl",               DoubleType(),  True),
        StructField("dttl",               DoubleType(),  True),
        StructField("sload",              DoubleType(),  True),
        StructField("dload",              DoubleType(),  True),
        StructField("sloss",              DoubleType(),  True),
        StructField("dloss",              DoubleType(),  True),
        StructField("sinpkt",             DoubleType(),  True),
        StructField("dinpkt",             DoubleType(),  True),
        StructField("sjit",               DoubleType(),  True),
        StructField("djit",               DoubleType(),  True),
        StructField("swin",               DoubleType(),  True),
        StructField("stcpb",              DoubleType(),  True),
        StructField("dtcpb",              DoubleType(),  True),
        StructField("dwin",               DoubleType(),  True),
        StructField("tcprtt",             DoubleType(),  True),
        StructField("synack",             DoubleType(),  True),
        StructField("ackdat",             DoubleType(),  True),
        StructField("smean",              DoubleType(),  True),
        StructField("dmean",              DoubleType(),  True),
        StructField("trans_depth",        DoubleType(),  True),
        StructField("response_body_len",  DoubleType(),  True),
        StructField("ct_srv_src",         DoubleType(),  True),
        StructField("ct_state_ttl",       DoubleType(),  True),
        StructField("ct_dst_ltm",         DoubleType(),  True),
        StructField("ct_src_dport_ltm",   DoubleType(),  True),
        StructField("ct_dst_sport_ltm",   DoubleType(),  True),
        StructField("ct_dst_src_ltm",     DoubleType(),  True),
        StructField("is_ftp_login",       DoubleType(),  True),
        StructField("ct_ftp_cmd",         DoubleType(),  True),
        StructField("ct_flw_http_mthd",   DoubleType(),  True),
        StructField("ct_src_ltm",         DoubleType(),  True),
        StructField("ct_srv_dst",         DoubleType(),  True),
        StructField("is_sm_ips_ports",    DoubleType(),  True),
    ])

    # Cast all numeric fields to float
    row_data = [(
        str(row_dict["proto"]),
        str(row_dict["service"]),
        str(row_dict["state"]),
        float(row_dict["dur"]),
        float(row_dict["sbytes"]),
        float(row_dict["dbytes"]),
        float(row_dict["spkts"]),
        float(row_dict["dpkts"]),
        float(row_dict["rate"]),
        float(row_dict["sttl"]),
        float(row_dict["dttl"]),
        float(row_dict["sload"]),
        float(row_dict["dload"]),
        float(row_dict["sloss"]),
        float(row_dict["dloss"]),
        float(row_dict["sinpkt"]),
        float(row_dict["dinpkt"]),
        float(row_dict["sjit"]),
        float(row_dict["djit"]),
        float(row_dict["swin"]),
        float(row_dict["stcpb"]),
        float(row_dict["dtcpb"]),
        float(row_dict["dwin"]),
        float(row_dict["tcprtt"]),
        float(row_dict["synack"]),
        float(row_dict["ackdat"]),
        float(row_dict["smean"]),
        float(row_dict["dmean"]),
        float(row_dict["trans_depth"]),
        float(row_dict["response_body_len"]),
        float(row_dict["ct_srv_src"]),
        float(row_dict["ct_state_ttl"]),
        float(row_dict["ct_dst_ltm"]),
        float(row_dict["ct_src_dport_ltm"]),
        float(row_dict["ct_dst_sport_ltm"]),
        float(row_dict["ct_dst_src_ltm"]),
        float(row_dict["is_ftp_login"]),
        float(row_dict["ct_ftp_cmd"]),
        float(row_dict["ct_flw_http_mthd"]),
        float(row_dict["ct_src_ltm"]),
        float(row_dict["ct_srv_dst"]),
        float(row_dict["is_sm_ips_ports"]),
    )]

    try:
        df_spark = spark.createDataFrame(row_data, schema=schema)
        result   = model.transform(df_spark)
        row_out  = result.select("prediction", "probability").collect()[0]
        pred     = int(row_out["prediction"])
        prob_vec = row_out["probability"]
        # probability is DenseVector [p_normal, p_attack]
        attack_prob = float(prob_vec[1])
        risk_level  = "High" if attack_prob >= 0.8 else "Medium" if attack_prob >= 0.5 else "Low"
        return {
            "prediction":   pred,
            "label":        "Attack" if pred == 1 else "Normal",
            "confidence":   round(attack_prob if pred == 1 else float(prob_vec[0]), 4),
            "attack_prob":  round(attack_prob, 4),
            "risk_level":   risk_level,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    if not outputs_ready():
        print("=" * 60)
        print("WARNING: outputs/ not found.")
        print("Run  python main.py  first to generate them.")
        print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
