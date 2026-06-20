from pydantic import BaseModel
from datetime import datetime, date
from typing import List, Optional

# shapes for employee data
class EmployeeBase(BaseModel):
    name: str
    department: Optional[str] = "Engineering"

class EmployeeCreate(EmployeeBase):
    pass

class EmployeeResponse(EmployeeBase):
    employee_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# how we track a chunk of activity (a session segment)
class ActivityLogBase(BaseModel):
    state: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: int = 0
    confidence: float
    raw_score: float = 0.0
    smoothed_score: float = 0.0
    transition_reason: Optional[str] = None
    notes: Optional[str] = None

class ActivityLogResponse(ActivityLogBase):
    id: int
    employee_id: int
    created_at: datetime
    duration_formatted: str  # making it pretty like "00:15:22"

    class Config:
        from_attributes = True

# the rolled-up daily stats
class DailySummaryResponse(BaseModel):
    id: int
    employee_id: int
    date: date
    working_seconds: int
    idle_seconds: int
    absent_seconds: int
    working_time: str            # "HH:MM:SS"
    idle_time: str               # "HH:MM:SS"
    absent_time: str             # "HH:MM:SS"
    total_monitored_time: str    # "HH:MM:SS"
    productivity_score: float

    class Config:
        from_attributes = True

# what the frontend needs right now, plus some dev variables
class LiveStatusResponse(BaseModel):
    employee_id: int
    name: str
    department: str
    status: str
    confidence: float
    time_in_state: str           # "HH:MM:SS"
    working_time: str            # "HH:MM:SS"
    idle_time: str               # "HH:MM:SS"
    absent_time: str             # "HH:MM:SS"
    productivity_score_today: float
    first_activity: Optional[str] = None  # "HH:MM:SS"
    last_activity: Optional[str] = None   # "HH:MM:SS"
    total_monitored_time: str    # "HH:MM:SS"
    
    # internal debug stuff to see what the math is doing
    raw_score: float
    smoothed_score: float
    movement_threshold: float
    idle_countdown: int
    working_countdown: int
    epsilon_filter: float
