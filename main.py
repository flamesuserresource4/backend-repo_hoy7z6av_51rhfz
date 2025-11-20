import os
from datetime import date, datetime, timedelta
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Mortality Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    dob: date = Field(..., description="Date of birth in ISO format (YYYY-MM-DD)")
    sex: Literal["male", "female"]
    country: str
    bmi: float
    smoking_status: Literal["none", "former", "light", "moderate", "heavy"]
    alcohol_use: Literal["none", "moderate", "heavy"]
    exercise_level: Literal["sedentary", "light", "moderate", "high"]


class PredictionResponse(BaseModel):
    estimated_death_date: date
    base_life_expectancy_years: float
    adjusted_life_expectancy_years: float
    current_age_years: float
    modifiers_applied: dict


# Simple, approximate life expectancy table (years at birth) per country & sex.
# Source inspiration: WHO/World Bank 2019-2022 aggregates (approximate values).
# These are intentionally simplified for entertainment-only purposes.
LIFE_TABLE = {
    "usa": {"male": 74.5, "female": 80.2},
    "united states": {"male": 74.5, "female": 80.2},
    "uk": {"male": 79.0, "female": 82.9},
    "united kingdom": {"male": 79.0, "female": 82.9},
    "canada": {"male": 80.0, "female": 84.0},
    "australia": {"male": 81.2, "female": 85.3},
    "germany": {"male": 78.7, "female": 83.4},
    "india": {"male": 67.2, "female": 69.5},
    "japan": {"male": 81.5, "female": 87.5},
    "france": {"male": 79.4, "female": 85.3},
}


def get_base_life_expectancy(country: str, sex: str) -> float:
    key = country.strip().lower()
    if key not in LIFE_TABLE:
        # Fallback to a global average if country not found
        global_avg = {"male": 70.0, "female": 75.0}
        return global_avg[sex]
    return LIFE_TABLE[key][sex]


def apply_lifestyle_modifiers(
    base: float,
    bmi: float,
    smoking_status: str,
    alcohol_use: str,
    exercise_level: str,
) -> tuple[float, dict]:
    adjustments: dict[str, float] = {}
    adjusted = base

    # BMI impact (simplified U-shaped curve)
    if bmi < 18.5:
        adjustments["bmi"] = -2.0
    elif 18.5 <= bmi < 25:
        adjustments["bmi"] = +0.5
    elif 25 <= bmi < 30:
        adjustments["bmi"] = -0.5
    elif 30 <= bmi < 35:
        adjustments["bmi"] = -2.0
    else:  # >= 35
        adjustments["bmi"] = -4.0
    adjusted += adjustments["bmi"]

    # Smoking impact
    smoking_map = {
        "none": +1.5,
        "former": -1.0,
        "light": -2.0,
        "moderate": -4.0,
        "heavy": -7.0,
    }
    adjustments["smoking"] = smoking_map.get(smoking_status, -2.0)
    adjusted += adjustments["smoking"]

    # Alcohol use impact (simplified)
    alcohol_map = {
        "none": +0.5,
        "moderate": 0.0,
        "heavy": -3.0,
    }
    adjustments["alcohol"] = alcohol_map.get(alcohol_use, 0.0)
    adjusted += adjustments["alcohol"]

    # Exercise impact
    exercise_map = {
        "sedentary": -2.0,
        "light": -0.5,
        "moderate": +1.0,
        "high": +2.0,
    }
    adjustments["exercise"] = exercise_map.get(exercise_level, 0.0)
    adjusted += adjustments["exercise"]

    return adjusted, adjustments


def years_between(d1: date, d2: date) -> float:
    return (d2 - d1).days / 365.2425


@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    today = date.today()
    if req.dob > today:
        raise HTTPException(status_code=400, detail="Date of birth cannot be in the future")

    base_expectancy = get_base_life_expectancy(req.country, req.sex)
    adjusted_expectancy, modifiers = apply_lifestyle_modifiers(
        base_expectancy,
        req.bmi,
        req.smoking_status,
        req.alcohol_use,
        req.exercise_level,
    )

    current_age = max(0.0, years_between(req.dob, today))

    # Ensure adjusted expectancy is at least current age + a minimal buffer
    minimal_future = 0.25  # 3 months buffer to avoid past dates
    lifespan_years = max(adjusted_expectancy, current_age + minimal_future)

    # Estimated death date = dob + lifespan_years
    estimated_days = int(round(lifespan_years * 365.2425))
    estimated_date = req.dob + timedelta(days=estimated_days)

    return PredictionResponse(
        estimated_death_date=estimated_date,
        base_life_expectancy_years=round(base_expectancy, 2),
        adjusted_life_expectancy_years=round(lifespan_years, 2),
        current_age_years=round(current_age, 2),
        modifiers_applied=modifiers,
    )


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
