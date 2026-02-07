from datetime import datetime
from database.settings import Base, engine
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Time, Date, DateTime

class Courses(Base):
    __tablename__ = "Courses"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    content = Column(String, nullable=False)
    quiz_keywords = Column(String, nullable=True)
    total_duration = Column(String, nullable=True)
    date_created = Column(DateTime, default=datetime.now)


class User(Base):
    __tablename__ = "User"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    adhd_type = Column(String, nullable=True)
    learning_style = Column(String, nullable=True)
    age = Column(Integer, nullable=True)


class Statistics(Base):
    __tablename__ = "Statistics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("User.id"), nullable=False)
    progress = Column(Integer, default=0)
    total_study_minute = Column(Integer, default=0)
    completed_lessons = Column(Integer, default=0)
    focus_points = Column(Integer, default=0)

class Feedback(Base):
    __tablename__ = "Feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("User.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("Courses.id"), nullable=False)
    rating = Column(Integer, nullable=False)
    comments = Column(String, nullable=True)

class EyeTrackingSession(Base):
    __tablename__ = "EyeTrackingSession"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("Courses.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("User.id"), nullable=True)
    tracking_mode = Column(String, nullable=False)  # 'webcam' or 'mouse'
    total_duration = Column(Integer, nullable=False)  # in seconds
    tab_switches = Column(Integer, default=0)
    focus_status_good = Column(Integer, default=0)  # seconds
    focus_status_warning = Column(Integer, default=0)  # seconds
    focus_status_alert = Column(Integer, default=0)  # seconds
    session_data = Column(String, nullable=True)  # JSON formatted detailed data
    created_at = Column(DateTime, default=datetime.now)