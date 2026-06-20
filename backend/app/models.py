from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Date, Enum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class Employee(Base):
    __tablename__ = "employees"

    employee_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    department = Column(String(255), default="Engineering")
    created_at = Column(DateTime, server_default=func.now())

    # how this links to other tables
    logs = relationship("ActivityLog", back_populates="employee", cascade="all, delete-orphan")
    summaries = relationship("DailySummary", back_populates="employee", cascade="all, delete-orphan")


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.employee_id", ondelete="CASCADE"), nullable=False)
    state = Column(Enum("WORKING", "IDLE", "ABSENT", name="state_enum"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, default=0)
    confidence = Column(Float, nullable=False)
    raw_score = Column(Float, default=0.0)
    smoothed_score = Column(Float, default=0.0)
    transition_reason = Column(String(255), nullable=True)
    notes = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    # tie it back to the employee
    employee = relationship("Employee", back_populates="logs")

    # speed up queries with these indexes
    __table_args__ = (
        Index("idx_logs_employee_start", "employee_id", "start_time"),
        Index("idx_logs_employee_state", "employee_id", "state"),
        Index("idx_logs_open_session", "employee_id", "end_time"),
    )


class DailySummary(Base):
    __tablename__ = "daily_summary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.employee_id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    working_seconds = Column(Integer, default=0)
    idle_seconds = Column(Integer, default=0)
    absent_seconds = Column(Integer, default=0)
    productivity_score = Column(Float, default=0.0)

    # relationship hooks
    employee = relationship("Employee", back_populates="summaries")

    # indexes so things stay snappy
    __table_args__ = (
        Index("idx_summary_employee_date", "employee_id", "date"),
    )
