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
    version="2.0.0"
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

def safe_float(val, default=50.0):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except:
        return default

# ── Frontend ──────────────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/app")
def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))

# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "PRISM API",
        "version": "2.0.0",
        "dashboard": "/app",
        "endpoints": [
            "/score/{iso3}",
            "/rankings",
            "/trend/{iso3}",
            "/brief/{iso3}",
            "/countries",
        ]
    }

# ── Score ─────────────────────────────────────────────────────────────────────
@app.get("/score/{iso3}")
def get_score(iso3: str):
    row = latest(iso3)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Country '{iso3}' not found")
    return {
        "iso3":        row["iso3"],
        "country":     row["country_name"],
        "year":        int(row["year"]),
        "prism_score": round(safe_float(row["prism_score"]), 2),
        "dimensions": {
            dim: round(safe_float(row[dim]), 2)
            for dim in RISK_DIMS
            if dim in row and pd.notna(row[dim])
        }
    }

# ── Rankings ──────────────────────────────────────────────────────────────────
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
                "prism_score": round(safe_float(row["prism_score"]), 2),
                "year":        int(row["year"])
            }
            for i, (_, row) in enumerate(data.iterrows())
        ]
    }

# ── Trend ─────────────────────────────────────────────────────────────────────
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
                "prism_score": round(safe_float(row["prism_score"]), 2),
                "dimensions": {
                    dim: round(safe_float(row[dim]), 2)
                    for dim in RISK_DIMS
                    if dim in row and pd.notna(row[dim])
                }
            }
            for _, row in rows.iterrows()
        ]
    }

# ── Countries ─────────────────────────────────────────────────────────────────
@app.get("/countries")
def get_countries():
    latest_df = df.sort_values("year", ascending=False).drop_duplicates("iso3")
    return {
        "count":     len(latest_df),
        "countries": latest_df[["iso3", "country_name"]].to_dict(orient="records")
    }

# ── Intelligence Brief (rule-based, no API key needed) ────────────────────────
def generate_brief(iso3: str) -> dict:
    row = latest(iso3)
    if row is None:
        return {"error": "Country not found"}

    name     = row["country_name"]
    score    = safe_float(row["prism_score"])
    year     = int(row["year"])
    gov      = safe_float(row.get("risk_governance", 50))
    conflict = safe_float(row.get("risk_conflict", 50))
    econ     = safe_float(row.get("risk_economic", 50))
    dem      = safe_float(row.get("risk_democracy", 50))
    corr     = safe_float(row.get("risk_corruption", 50))
    social   = safe_float(row.get("risk_social_unrest", 50))
    sanction = safe_float(row.get("risk_sanctions", 50))
    geo      = safe_float(row.get("risk_geopolitical", 50))

    # Risk level
    if score >= 72:   level = "CRITICAL"
    elif score >= 65: level = "HIGH"
    elif score >= 55: level = "ELEVATED"
    elif score >= 48: level = "MODERATE"
    else:             level = "LOW"

    # Top and bottom dimensions
    dims = {
        "governance":    gov,
        "conflict":      conflict,
        "economic":      econ,
        "democracy":     dem,
        "corruption":    corr,
        "social unrest": social,
        "sanctions":     sanction,
        "geopolitical":  geo,
    }
    sorted_dims = sorted(dims.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_dims[:3]
    low3 = sorted_dims[-3:]

    def dim_desc(name, val):
        if val >= 80: return f"severely elevated {name} ({val:.0f})"
        if val >= 65: return f"high {name} ({val:.0f})"
        if val >= 50: return f"moderate {name} ({val:.0f})"
        return f"low {name} ({val:.0f})"

    drivers   = ", ".join([dim_desc(n, v) for n, v in top3])
    strengths = ", ".join([dim_desc(n, v) for n, v in low3])

    # Governance sentence
    if gov >= 75:
        gov_sent = "Governance institutions are severely weakened, with rule of law and government effectiveness at critical lows."
    elif gov >= 60:
        gov_sent = "Governance capacity is strained, with notable deficiencies in institutional effectiveness and regulatory quality."
    elif gov >= 45:
        gov_sent = "Governance structures are functional but face persistent challenges in accountability and effectiveness."
    else:
        gov_sent = "Governance indicators are comparatively strong, reflecting stable institutions and regulatory frameworks."

    # Conflict sentence
    if conflict >= 80:
        conflict_sent = "Active conflict or severe instability is ongoing, representing an immediate threat to operational continuity."
    elif conflict >= 60:
        conflict_sent = "Elevated conflict risk persists, with historical patterns of instability influencing the current security environment."
    elif conflict >= 40:
        conflict_sent = "Conflict risk is moderate, with localised tensions that require monitoring but are unlikely to escalate broadly."
    else:
        conflict_sent = "Conflict indicators are low, reflecting a relatively stable security environment."

    # Economic sentence
    if econ >= 75:
        econ_sent = "Economic fragility is acute, with inflation, unemployment, or debt levels presenting significant macro-financial risks."
    elif econ >= 55:
        econ_sent = "Economic conditions are under pressure, with structural vulnerabilities that could amplify other risk factors."
    else:
        econ_sent = "Economic fundamentals are relatively stable, providing a degree of resilience against other risk drivers."

    # Democracy sentence
    if dem >= 75:
        dem_sent = "Democratic backsliding is severe, with significant erosion of electoral integrity and civil liberties."
    elif dem >= 55:
        dem_sent = "Democratic indicators are under strain, with concerns around electoral processes and political freedoms."
    else:
        dem_sent = "Democratic institutions are relatively robust, supporting political stability and accountability."

    # Corruption sentence
    if corr >= 75:
        corr_sent = "Corruption is pervasive across public institutions, undermining governance and investment confidence."
    elif corr >= 55:
        corr_sent = "Corruption remains a significant concern, with systemic issues in public procurement and judicial independence."
    else:
        corr_sent = "Corruption levels are comparatively contained, supporting institutional credibility."

    # Outlook
    if score >= 70:
        outlook = f"The overall outlook for {name} remains deeply concerning. Sustained engagement requires robust risk mitigation protocols and contingency planning."
    elif score >= 55:
        outlook = f"The risk environment in {name} warrants careful monitoring. Conditions could deteriorate rapidly if key drivers are not addressed."
    else:
        outlook = f"{name} presents a manageable risk profile. Standard due diligence procedures are recommended for operational planning."

    brief_text = (
        f"{name} presents a {level} risk profile with a PRISM score of {score:.1f}/100 as of {year}, "
        f"driven primarily by {drivers}. "
        f"{gov_sent} "
        f"{conflict_sent} "
        f"{econ_sent} "
        f"{dem_sent} "
        f"{corr_sent} "
        f"Relative stability is observed in {strengths}. "
        f"{outlook}"
    )

    return {
        "iso3":        iso3.upper(),
        "country":     name,
        "year":        year,
        "risk_level":  level,
        "prism_score": round(score, 2),
        "brief":       brief_text,
        "top_drivers": [{"dimension": n, "score": round(v, 1)} for n, v in top3],
        "strengths":   [{"dimension": n, "score": round(v, 1)} for n, v in low3],
    }

@app.get("/brief/{iso3}")
def get_brief(iso3: str):
    result = generate_brief(iso3)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result