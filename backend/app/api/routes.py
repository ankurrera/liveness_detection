import time
import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.models import Employee, ActivityLog, DailySummary
from app.schemas import EmployeeCreate, EmployeeResponse, ActivityLogResponse, DailySummaryResponse, LiveStatusResponse
from app.services.cv_pipeline import cv_monitor
from app.services.db_services import get_live_status, recalculate_daily_summary, format_seconds_to_hhmmss
from app.config import settings

router = APIRouter()

# 1. the video pipeline
@router.get("/video_feed")
def get_video_feed():
    """Pipes the live webcam or mock feed straight to the frontend with all the overlays."""
    def frame_generator():
        while True:
            frame_bytes = cv_monitor.latest_frame
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.04)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# 2. stuff for managing employees
@router.get("/employees", response_model=List[EmployeeResponse])
def read_employees(db: Session = Depends(get_db)):
    return db.query(Employee).all()

@router.post("/employees", response_model=EmployeeResponse)
def create_employee(employee: EmployeeCreate, db: Session = Depends(get_db)):
    db_employee = Employee(name=employee.name, department=employee.department)
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    return db_employee

@router.get("/employee/{employee_id}", response_model=EmployeeResponse)
def read_employee(employee_id: int, db: Session = Depends(get_db)):
    db_employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return db_employee

@router.get("/active_employee")
def get_active_employee():
    """Finds out who the computer vision is looking at right now."""
    return {"active_employee_id": cv_monitor.active_employee_id}

@router.post("/active_employee")
def set_active_employee(employee_id: int, db: Session = Depends(get_db)):
    """Tells the computer vision to watch someone else."""
    db_employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    cv_monitor.set_active_employee(employee_id)
    return {"status": "success", "active_employee_id": employee_id}

# 3. stuff for activity tracking
@router.get("/activity/live", response_model=LiveStatusResponse)
def read_live_activity(employee_id: int = settings.DEFAULT_EMPLOYEE_ID, db: Session = Depends(get_db)):
    """
    Grabs what the user is doing right this second,
    plus the current day's score and some behind-the-scenes metrics.
    """
    live_cv_metrics = cv_monitor.get_debug_metrics()
    status_data = get_live_status(db, employee_id, live_cv_metrics)
    if not status_data:
        raise HTTPException(status_code=404, detail="Employee not found")
    return status_data

@router.get("/activity/history", response_model=List[ActivityLogResponse])
def read_activity_history(
    employee_id: int = settings.DEFAULT_EMPLOYEE_ID,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    Gets the history of what they were doing.
    Also does the math to make the time readable (like HH:MM:SS).
    """
    logs = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id
    ).order_by(ActivityLog.start_time.desc()).limit(limit).all()
    
    response_logs = []
    now = datetime.datetime.now()
    for log in logs:
        # if they're currently in this state, figure out how long it's been going on
        if log.end_time is None:
            dur = int((now - log.start_time).total_seconds())
        else:
            dur = log.duration_seconds

        response_logs.append({
            "id": log.id,
            "employee_id": log.employee_id,
            "state": log.state,
            "start_time": log.start_time,
            "end_time": log.end_time,
            "duration_seconds": dur,
            "confidence": log.confidence,
            "notes": log.notes,
            "created_at": log.created_at,
            "duration_formatted": format_seconds_to_hhmmss(dur)
        })
        
    return response_logs

# 4. the analytics and charts stuff
@router.get("/analytics/daily", response_model=DailySummaryResponse)
def read_daily_analytics(
    employee_id: int = settings.DEFAULT_EMPLOYEE_ID,
    target_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Rolls up the whole day into nice readable stats."""
    if target_date:
        try:
            parsed_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        parsed_date = datetime.date.today()

    summary = recalculate_daily_summary(db, employee_id, parsed_date)
    
    w_sec = summary.working_seconds if summary else 0
    i_sec = summary.idle_seconds if summary else 0
    a_sec = summary.absent_seconds if summary else 0
    total_sec = w_sec + i_sec + a_sec
    score = summary.productivity_score if summary else 0.0
    
    return {
        "id": summary.id if summary else 0,
        "employee_id": employee_id,
        "date": parsed_date,
        "working_seconds": w_sec,
        "idle_seconds": i_sec,
        "absent_seconds": a_sec,
        "working_time": format_seconds_to_hhmmss(w_sec),
        "idle_time": format_seconds_to_hhmmss(i_sec),
        "absent_time": format_seconds_to_hhmmss(a_sec),
        "total_monitored_time": format_seconds_to_hhmmss(total_sec),
        "productivity_score": score
    }

@router.get("/analytics/weekly", response_model=List[DailySummaryResponse])
def read_weekly_analytics(employee_id: int = settings.DEFAULT_EMPLOYEE_ID, db: Session = Depends(get_db)):
    """Grabs the stats for the past week so we can chart it."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=6)
    
    # make sure today's numbers are fresh before we send them over
    recalculate_daily_summary(db, employee_id, end_date)

    summaries = db.query(DailySummary).filter(
        DailySummary.employee_id == employee_id,
        DailySummary.date >= start_date,
        DailySummary.date <= end_date
    ).order_by(DailySummary.date.asc()).all()

    date_map = {s.date: s for s in summaries}
    result_list = []
    
    for i in range(7):
        curr_date = start_date + datetime.timedelta(days=i)
        if curr_date in date_map:
            s = date_map[curr_date]
            w_sec = s.working_seconds
            i_sec = s.idle_seconds
            a_sec = s.absent_seconds
            total_sec = w_sec + i_sec + a_sec
            score = s.productivity_score
            sum_id = s.id
        else:
            w_sec = i_sec = a_sec = total_sec = 0
            score = 0.0
            sum_id = 0
            
        result_list.append({
            "id": sum_id,
            "employee_id": employee_id,
            "date": curr_date,
            "working_seconds": w_sec,
            "idle_seconds": i_sec,
            "absent_seconds": a_sec,
            "working_time": format_seconds_to_hhmmss(w_sec),
            "idle_time": format_seconds_to_hhmmss(i_sec),
            "absent_time": format_seconds_to_hhmmss(a_sec),
            "total_monitored_time": format_seconds_to_hhmmss(total_sec),
            "productivity_score": score
        })
            
    return result_list
