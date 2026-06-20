import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import time

from app.api.routes import router as api_router
from app.database import engine, Base, SessionLocal
from app.services.cv_pipeline import cv_monitor
from app.services.db_services import get_or_create_default_employee

app = FastAPI(
    title="Employee Activity Monitoring System API",
    description="Backend API for Camera-based Employee Activity Tracking POC",
    version="1.0.0"
)

# setting up cors so the frontend can talk to us locally without throwing a fit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# tuck all the api routes under the /api prefix so it's neat
app.include_router(api_router, prefix="/api")

# things to do when the server wakes up
@app.on_event("startup")
def startup_event():
    print("[Server] Initializing database tables...")
    # let's try connecting to the database a few times in case it's still groggy and waking up
    retries = 5
    while retries > 0:
        try:
            Base.metadata.create_all(bind=engine)
            db = SessionLocal()
            # throw in a default employee to start with
            get_or_create_default_employee(db)
            db.close()
            print("[Server] Database connection established and tables validated.")
            break
        except Exception as e:
            print(f"[Server] Database connection failed: {e}. Retrying in 2 seconds... ({retries} left)")
            time.sleep(2)
            retries -= 1
            if retries == 0:
                print("[Server] CRITICAL: Could not connect to MySQL database. Server starting with DB errors.")

    print("[Server] Starting Computer Vision Monitoring engine...")
    cv_monitor.start()

# things to clean up before the server goes to sleep
@app.on_event("shutdown")
def shutdown_event():
    print("[Server] Stopping Computer Vision Monitoring engine...")
    cv_monitor.stop()

# serve up the frontend html/css/js
# pro tip: put this at the very end so our api routes don't get swallowed
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    print(f"[Server] Static frontend successfully mounted from: {frontend_dir}")
else:
    print(f"[Server] WARNING: Frontend directory not found at: {frontend_dir}. Static web server disabled.")
