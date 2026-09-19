import os
import json
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

load_dotenv()

# PostgreSQL Database Configuration
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    pg_user = os.getenv("PGUSER", "postgres")
    pg_pass = os.getenv("PGPASSWORD", "")
    pg_host = os.getenv("PGHOST", "localhost")
    pg_port = os.getenv("PGPORT", "5432")
    pg_db = os.getenv("PGDATABASE", "agriyield_db")
    
    if os.getenv("PGHOST") or os.getenv("DATABASE_URL"):
        DB_URL = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    else:
        # Local SQLite fallback if no PostgreSQL credentials provided
        DB_URL = "sqlite:///agriyield.db"

# Fix Heroku/Railway postgres:// prefix to postgresql://
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

engine_kwargs = {}
if DB_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20

engine = create_engine(DB_URL, **engine_kwargs)
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    crop = Column(String(100), nullable=False)
    location = Column(String(150), nullable=False)
    season = Column(String(50))
    irrigation = Column(String(50))
    field_area = Column(Float)
    soil_ph = Column(Float)
    moisture = Column(Float)
    nitrogen = Column(Float)
    phosphorus = Column(Float)
    potassium = Column(Float)
    organic_matter = Column(Float)
    algorithm_used = Column(String(100))
    predicted_yield = Column(Float)
    total_harvest = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class SoilAnalysisRecord(Base):
    __tablename__ = "soil_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    soil_ph = Column(Float)
    moisture = Column(Float)
    nitrogen = Column(Float)
    phosphorus = Column(Float)
    potassium = Column(Float)
    organic_matter = Column(Float)
    intended_crop = Column(String(100))
    soil_status = Column(String(50))
    recommendation_summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatLogRecord(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_message = Column(Text, nullable=False)
    ai_reply = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    """Initialize database tables."""
    global engine, db_session
    try:
        with engine.connect() as conn:
            pass
        Base.metadata.create_all(bind=engine)
        db_type = "PostgreSQL" if DB_URL.startswith("postgresql") else "SQLite (Fallback)"
        print(f"Database initialized successfully. Type: {db_type}")
    except Exception as e:
        print(f"PostgreSQL connection failed ({e}). Falling back to SQLite agriyield.db...")
        engine = create_engine("sqlite:///agriyield.db", connect_args={"check_same_thread": False})
        db_session.configure(bind=engine)
        Base.metadata.create_all(bind=engine)

def save_prediction(crop, location, season, irrigation, field_area, soil_ph, moisture, n, p, k, om, algo, predicted_yield, total_harvest):
    session = db_session()
    try:
        rec = PredictionRecord(
            crop=crop, location=location, season=season, irrigation=irrigation,
            field_area=field_area, soil_ph=soil_ph, moisture=moisture,
            nitrogen=n, phosphorus=p, potassium=k, organic_matter=om,
            algorithm_used=algo, predicted_yield=predicted_yield, total_harvest=total_harvest
        )
        session.add(rec)
        session.commit()
        return rec.id
    except Exception as e:
        session.rollback()
        print("Error saving prediction to database:", e)
        return None
    finally:
        session.close()

def save_soil_analysis(ph, moisture, n, p, k, om, crop, soil_status, recommendation_summary):
    session = db_session()
    try:
        rec = SoilAnalysisRecord(
            soil_ph=ph, moisture=moisture, nitrogen=n, phosphorus=p, potassium=k,
            organic_matter=om, intended_crop=crop, soil_status=soil_status,
            recommendation_summary=recommendation_summary
        )
        session.add(rec)
        session.commit()
        return rec.id
    except Exception as e:
        session.rollback()
        print("Error saving soil analysis to database:", e)
        return None
    finally:
        session.close()

def save_chat_log(user_message, ai_reply):
    session = db_session()
    try:
        rec = ChatLogRecord(user_message=user_message, ai_reply=ai_reply)
        session.add(rec)
        session.commit()
        return rec.id
    except Exception as e:
        session.rollback()
        print("Error saving chat log to database:", e)
        return None
    finally:
        session.close()

def get_db_stats():
    session = db_session()
    try:
        db_type = "PostgreSQL" if DB_URL.startswith("postgresql") else "SQLite"
        preds_count = session.query(PredictionRecord).count()
        soil_count = session.query(SoilAnalysisRecord).count()
        chat_count = session.query(ChatLogRecord).count()
        return {
            "database_type": db_type,
            "predictions_count": preds_count,
            "soil_tests_count": soil_count,
            "chat_logs_count": chat_count
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        session.close()
