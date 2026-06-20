#!/usr/bin/env python3
"""
Quick health check script for the system.
Makes sure you've got all the python packages and the database is talking to us.
"""
import sys
import os

# tweak the path so we can import stuff from the backend folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

def check_dependencies():
    print("Step 1: Checking Python Dependencies...")
    libs = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("sqlalchemy", "SQLAlchemy"),
        ("pymysql", "PyMySQL (MySQL driver)"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("numpy", "NumPy"),
        ("pydantic", "Pydantic"),
        ("dotenv", "Python Dotenv")
    ]
    
    all_ok = True
    for module_name, clean_name in libs:
        try:
            if module_name == "dotenv":
                import dotenv
            else:
                __import__(module_name)
            print(f"  [✓] {clean_name} is installed.")
        except ImportError as e:
            print(f"  [✗] {clean_name} is missing! Error: {e}")
            all_ok = False
    return all_ok

def check_database():
    print("\nStep 2: Checking MySQL Database Connectivity...")
    try:
        import dotenv
        dotenv.load_dotenv("backend/.env")
        from sqlalchemy import inspect
        from app.database import engine, SessionLocal
        from app.models import Employee, ActivityLog, DailySummary
        
        # let's see if we can actually connect
        connection = engine.connect()
        print("  [✓] Successfully connected to MySQL server.")
        connection.close()

        # make sure all the tables we need are actually there
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        required_tables = ["employees", "activity_logs", "daily_summary"]
        
        tables_ok = True
        for table in required_tables:
            if table in existing_tables:
                print(f"  [✓] Table '{table}' exists.")
            else:
                print(f"  [✗] Table '{table}' is missing from the database.")
                tables_ok = False
                
        if tables_ok:
            # just checking if our default test user is in there
            db = SessionLocal()
            try:
                emp = db.query(Employee).filter(Employee.employee_id == 1).first()
                if emp:
                    print(f"  [✓] Default employee (ID: 1, Name: {emp.name}) is seeded in database.")
                else:
                    print("  [WARNING] Default employee is not seeded yet. (This is done automatically when backend starts).")
            finally:
                db.close()
                
        return tables_ok
        
    except Exception as e:
        print(f"  [✗] Database verification failed! Error: {e}")
        print("      Make sure MySQL is running and credentials in backend/app/config.py or .env are correct.")
        return False

def main():
    print("==================================================")
    print("AuraSense - System Verification & Diagnostics")
    print("==================================================")
    
    deps_ok = check_dependencies()
    db_ok = check_database()
    
    print("\n==================================================")
    if deps_ok and db_ok:
        print("STATUS: SUCCESS - All components are operational!")
        print("You can start the backend by running: ./run.sh")
        sys.exit(0)
    else:
        print("STATUS: FAILED - Please resolve the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
