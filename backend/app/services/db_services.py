import datetime
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Employee, ActivityLog, DailySummary
from app.config import settings

def format_seconds_to_hhmmss(seconds: int) -> str:
    """Turns boring seconds into a nice human readable HH:MM:SS format."""
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_or_create_default_employee(db: Session) -> Employee:
    """Makes sure our default test users are actually in the database."""
    # make sure employee 1 is Ankur Bag
    employee = db.query(Employee).filter(Employee.employee_id == settings.DEFAULT_EMPLOYEE_ID).first()
    if not employee:
        employee = Employee(
            employee_id=settings.DEFAULT_EMPLOYEE_ID,
            name="Ankur Bag",
            department="Engineering"
        )
        db.add(employee)
        db.commit()
        db.refresh(employee)
    elif employee.name == "John Doe":
        employee.name = "Ankur Bag"
        db.commit()
        db.refresh(employee)

    # make sure employee 2 is Sayan Sarkar
    employee2 = db.query(Employee).filter(Employee.employee_id == 2).first()
    if not employee2:
        employee2 = Employee(
            employee_id=2,
            name="Sayan Sarkar",
            department="Engineering"
        )
        db.add(employee2)
        db.commit()
        db.refresh(employee2)

    return employee

def resolve_orphaned_sessions(db: Session):
    """
    Cleans up any sessions that were left hanging when the server crashed or stopped.
    We just zero them out so they don't mess up our stats.
    """
    orphans = db.query(ActivityLog).filter(ActivityLog.end_time == None).all()
    if orphans:
        print(f"[DB Service] Resolving {len(orphans)} orphaned sessions...")
        for o in orphans:
            o.end_time = o.start_time
            o.duration_seconds = 0
            o.notes = "Session orphaned and automatically closed on restart."
        db.commit()

def start_new_state_session(
    db: Session, 
    employee_id: int, 
    state: str, 
    confidence: float, 
    raw_score: float = 0.0,
    smoothed_score: float = 0.0,
    transition_reason: Optional[str] = None,
    notes: Optional[str] = None
) -> ActivityLog:
    """
    Moves a person from one state to another:
      1. wraps up whatever they were just doing.
      2. starts a new record for what they are doing now.
      3. updates the daily scoreboard.
    """
    now = datetime.datetime.now()
    
    # 1. wrap up any hanging sessions
    open_sessions = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id,
        ActivityLog.end_time == None
    ).all()
    
    for sess in open_sessions:
        sess.end_time = now
        sess.duration_seconds = max(0, int((now - sess.start_time).total_seconds()))
    
    # 2. start the new session and log why it happened
    new_log = ActivityLog(
        employee_id=employee_id,
        state=state,
        start_time=now,
        end_time=None,
        duration_seconds=0,
        confidence=confidence,
        raw_score=raw_score,
        smoothed_score=smoothed_score,
        transition_reason=transition_reason,
        notes=notes
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    
    # 3. refresh the daily stats
    recalculate_daily_summary(db, employee_id, datetime.date.today())
    return new_log

def close_active_session(db: Session, employee_id: int) -> None:
    """
    Tidies up any open sessions when we turn off the server.
    """
    now = datetime.datetime.now()
    open_sessions = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id,
        ActivityLog.end_time == None
    ).all()
    
    for sess in open_sessions:
        sess.end_time = now
        sess.duration_seconds = max(0, int((now - sess.start_time).total_seconds()))
        sess.notes = "Session closed gracefully on server stop."
    
    db.commit()
    recalculate_daily_summary(db, employee_id, datetime.date.today())

def recalculate_daily_summary(db: Session, employee_id: int, target_date: datetime.date) -> Optional[DailySummary]:
    """
    Tallies up the total time spent in each state for a given day.
    Also handles the edge cases where a session crosses midnight.
    """
    now = datetime.datetime.now()
    start_of_day = datetime.datetime.combine(target_date, datetime.time.min)
    end_of_day = datetime.datetime.combine(target_date, datetime.time.max)
    
    # if we're looking at today, only count up to right now
    end_of_time = now if target_date == now.date() else end_of_day
    
    # grab all the activity chunks that happened on this day
    logs = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id,
        ActivityLog.start_time <= end_of_time,
        (ActivityLog.end_time == None) | (ActivityLog.end_time >= start_of_day)
    ).all()
    
    durations = {"WORKING": 0, "IDLE": 0, "ABSENT": 0}
    
    for log in logs:
        # figure out exactly how much of this chunk fits into the day we're looking at
        seg_start = max(log.start_time, start_of_day)
        seg_end = min(log.end_time or now, end_of_time)
        
        overlap_seconds = int((seg_end - seg_start).total_seconds())
        if overlap_seconds > 0:
            durations[log.state] += overlap_seconds
            
    w_sec = durations["WORKING"]
    i_sec = durations["IDLE"]
    a_sec = durations["ABSENT"]
    total_sec = w_sec + i_sec + a_sec
    
    # productivity is just working time divided by total monitored time
    productivity_score = 0.0
    if total_sec > 0:
        productivity_score = round((w_sec / total_sec) * 100.0, 1)
        
    summary = db.query(DailySummary).filter(
        DailySummary.employee_id == employee_id,
        DailySummary.date == target_date
    ).first()
    
    try:
        if not summary:
            summary = DailySummary(
                employee_id=employee_id,
                date=target_date,
                working_seconds=w_sec,
                idle_seconds=i_sec,
                absent_seconds=a_sec,
                productivity_score=productivity_score
            )
            db.add(summary)
        else:
            summary.working_seconds = w_sec
            summary.idle_seconds = i_sec
            summary.absent_seconds = a_sec
            summary.productivity_score = productivity_score
        db.commit()
    except Exception:
        db.rollback()
        summary = db.query(DailySummary).filter(
            DailySummary.employee_id == employee_id,
            DailySummary.date == target_date
        ).first()
        if summary:
            summary.working_seconds = w_sec
            summary.idle_seconds = i_sec
            summary.absent_seconds = a_sec
            summary.productivity_score = productivity_score
            db.commit()
        else:
            raise

    db.refresh(summary)
    return summary

def get_live_status(db: Session, employee_id: int, live_cv_metrics: dict) -> dict:
    """
    Pulls together everything we need for the live dashboard,
    combining database history with live computer vision metrics.
    """
    employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if not employee:
        return {}
        
    # update the daily stats right now so the dashboard is perfectly accurate
    summary = recalculate_daily_summary(db, employee_id, datetime.date.today())
    
    # grab what they are doing at this exact moment
    latest_log = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id
    ).order_by(ActivityLog.start_time.desc()).first()
    
    if latest_log:
        status = latest_log.state
        confidence = latest_log.confidence
        if latest_log.end_time is None:
            time_in_state = int((datetime.datetime.now() - latest_log.start_time).total_seconds())
        else:
            time_in_state = latest_log.duration_seconds
    else:
        status = "ABSENT"
        confidence = 0.0
        time_in_state = 0
        
    # figure out when they started and when we last saw them
    start_of_day = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
    first_log = db.query(ActivityLog).filter(
        ActivityLog.employee_id == employee_id,
        ActivityLog.start_time >= start_of_day
    ).order_by(ActivityLog.start_time.asc()).first()
    
    first_act = first_log.start_time.strftime("%H:%M:%S") if first_log else None
    last_act = latest_log.start_time.strftime("%H:%M:%S") if latest_log else None
    
    w_sec = summary.working_seconds if summary else 0
    i_sec = summary.idle_seconds if summary else 0
    a_sec = summary.absent_seconds if summary else 0
    total_sec = w_sec + i_sec + a_sec
    productivity = summary.productivity_score if summary else 0.0
    
    return {
        "employee_id": employee.employee_id,
        "name": employee.name,
        "department": employee.department,
        "status": status,
        "confidence": confidence,
        "time_in_state": format_seconds_to_hhmmss(time_in_state),
        "working_time": format_seconds_to_hhmmss(w_sec),
        "idle_time": format_seconds_to_hhmmss(i_sec),
        "absent_time": format_seconds_to_hhmmss(a_sec),
        "productivity_score_today": productivity,
        "first_activity": first_act,
        "last_activity": last_act,
        "total_monitored_time": format_seconds_to_hhmmss(total_sec),
        
        # mix in the live debug info from the cv engine
        "raw_score": live_cv_metrics.get("raw_score", 0.0),
        "smoothed_score": live_cv_metrics.get("smoothed_score", 0.0),
        "movement_threshold": live_cv_metrics.get("movement_threshold", 0.50),
        "idle_countdown": live_cv_metrics.get("idle_countdown", 0),
        "working_countdown": live_cv_metrics.get("working_countdown", 0),
        "epsilon_filter": live_cv_metrics.get("epsilon_filter", 0.0015)
    }
