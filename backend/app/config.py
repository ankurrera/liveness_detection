import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # where does our mysql live?
    MYSQL_HOST: str = Field(default="localhost")
    MYSQL_PORT: int = Field(default=3306)
    MYSQL_USER: str = Field(default="root")
    MYSQL_PASSWORD: str = Field(default="")
    MYSQL_DB: str = Field(default="employee_activity_db")

    # camera stuff
    CAMERA_INDEX: int = Field(default=0)
    MOCK_CAMERA: bool = Field(default=False)  # flip this on to fake a feed if you don't have a webcam
    
    # cv logic constraints
    MOVEMENT_THRESHOLD: float = Field(default=0.015)  # how much they need to move (euclidean dist)
    ABSENT_TIMEOUT: float = Field(default=10.0)      # seconds until we assume they walked away
    IDLE_TIMEOUT: float = Field(default=60.0)        # seconds until they're slacking

    # yolo model configs
    YOLO_MODEL_NAME: str = Field(default="yolov8n-pose.pt")
    YOLO_INFERENCE_SIZE: int = Field(default=320)
    YOLO_CONFIDENCE_THRESHOLD: float = Field(default=0.25)
    YOLO_DEVICE: str = Field(default="cpu")
    YOLO_SMOOTHING_FACTOR: float = Field(default=0.35)
    YOLO_EPSILON_FILTER: float = Field(default=0.0015)
    YOLO_ACTIVITY_THRESHOLD: float = Field(default=0.50)
    YOLO_TRACKER: str = Field(default="bytetrack.yaml")
    YOLO_DEBUG_MODE: bool = Field(default=False)
    
    # who are we watching right now? (just for the poc since it's 1 camera)
    DEFAULT_EMPLOYEE_ID: int = Field(default=1)

    @property
    def database_url(self) -> str:
        # string together the pymysql uri
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
