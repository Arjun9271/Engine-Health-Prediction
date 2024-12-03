from fastapi import FastAPI, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
import pandas as pd
from joblib import load
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

# FastAPI instance
app = FastAPI()

# Templates directory
templates = Jinja2Templates(directory="templates")

# Database configuration
DATABASE_URL = "mysql+pymysql://root:1729@localhost/fastapi_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    sensor2 = Column(Float)
    sensor3 = Column(Float)
    sensor4 = Column(Float)
    sensor7 = Column(Float)
    sensor8 = Column(Float)
    sensor9 = Column(Float)
    sensor11 = Column(Float)
    sensor12 = Column(Float)
    sensor13 = Column(Float)
    sensor14 = Column(Float)
    sensor15 = Column(Float)
    sensor17 = Column(Float)
    sensor20 = Column(Float)
    sensor21 = Column(Float)
    engine_health = Column(String(10))

# Create database tables
Base.metadata.create_all(bind=engine)

# Pydantic model for input data
class SensorData(BaseModel):
    sensor2: float
    sensor3: float
    sensor4: float
    sensor7: float
    sensor8: float
    sensor9: float
    sensor11: float
    sensor12: float
    sensor13: float
    sensor14: float
    sensor15: float
    sensor17: float
    sensor20: float
    sensor21: float

# Load machine learning model and scaler
model_rf = load('best_rf_model.joblib')
scaler = load('best_rf_scaler.joblib')

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(sensor_data: SensorData, db: Session = Depends(get_db)):
    try:
        # Convert input data into DataFrame for processing
        sensor_data_dict = sensor_data.dict()
        sensor_values = [list(sensor_data_dict.values())]
        data = pd.DataFrame(sensor_values, columns=sensor_data_dict.keys())
        
        # Scale the input data
        scaled_data = scaler.transform(data)
        
        # Make prediction
        result = model_rf.predict(scaled_data)[0]
        engine_health = 'Failure' if result == 1 else 'Normal'

        # Save prediction in the database
        prediction = Prediction(**sensor_data_dict, engine_health=engine_health)
        db.add(prediction)
        db.commit()

        return {"engine_health": engine_health}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
