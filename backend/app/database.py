from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

# create the engine
# using pool_pre_ping so it gracefully recycles stale connections instead of blowing up
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600
)

# spin up our session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# the declarative base for our models
Base = declarative_base()

# dependency injection magic to hand out db sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
