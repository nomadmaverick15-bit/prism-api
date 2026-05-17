# main.py — PRISM FastAPI Backend + Frontend Server
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="PRISM API",
    description="Political Risk Intelligence System — Country Risk Scores",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load data at startup ──────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "prism_features.csv")
df = pd.read_csv(DATA_PATH)

RISK_DIMS = [
    "risk_governance", "risk_conflict", "risk_economic",
    "risk_democracy", "risk_corruption", "risk_social_unrest",
    "risk_sanctions", "risk_geopolitical"
]

def latest(iso3: str):
    rows = df[df["iso3"] == iso3.upper()]
    if rows.empty:
        return None
    return rows.sort_values("year", ascending=False).iloc[0]

# ── Frontend ──────────────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/app")
def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "PRISM API",
        "version": "1.0.0",
        "dashboard": "/app",
        "endpoints": [
            "/score/{iso3}",
            "/rankings",
            "/trend/{iso3}",
            "/countries"
        ]
    }

@app.get("/score/{iso3}")
def get_score(iso3: str):
    row = latest(iso3)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Country '{iso3}' not found")
    return {
        "iso3":        row["iso3"],
        "country":     row["country_name"],
        "year":        int(row["year"]),
        "prism_score": round(float(row["prism_score"]), 2),
        "dimensions": {
            dim: round(float(row[dim]), 2)
            for dim in RISK_DIMS
            if dim in row and pd.notna(row[dim])
        }
    }

@app.get("/rankings")
def get_rankings(year: int = None, limit: int = 215):
    data = df.copy()
    if year:
        data = data[data["year"] == year]
    else:
        data = data.sort_values("year", ascending=False)
        data = data.drop_duplicates(subset="iso3", keep="first")

    data = data.dropna(subset=["prism_score"])
    data = data.sort_values("prism_score", ascending=False).head(limit)
    data = data.reset_index(drop=True)

    return {
        "year":     year or "latest",
        "count":    len(data),
        "rankings": [
            {
                "rank":        int(i + 1),
                "iso3":        row["iso3"],
                "country":     row["country_name"],
                "prism_score": round(float(row["prism_score"]), 2),
                "year":        int(row["year"])
            }
            for i, (_, row) in enumerate(data.iterrows())
        ]
    }

@app.get("/trend/{iso3}")
def get_trend(iso3: str):
    rows = df[df["iso3"] == iso3.upper()].sort_values("year")
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Country '{iso3}' not found")
    return {
        "iso3":    rows.iloc[0]["iso3"],
        "country": rows.iloc[0]["country_name"],
        "trend": [
            {
                "year":        int(row["year"]),
                "prism_score": round(float(row["prism_score"]), 2),
                "dimensions": {
                    dim: round(float(row[dim]), 2)
                    for dim in RISK_DIMS
                    if dim in row and pd.notna(row[dim])
                }
            }
            for _, row in rows.iterrows()
        ]
    }

@app.get("/countries")
def get_countries():
    latest_df = df.sort_values("year", ascending=False).drop_duplicates("iso3")
    return {
        "count":     len(latest_df),
        "countries": latest_df[["iso3", "country_name"]].to_dict(orient="records")
    }