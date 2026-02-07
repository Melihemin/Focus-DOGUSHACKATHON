from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Depends, Request, HTTPException, UploadFile, File
from typing import Annotated
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database.settings import SessionLocal
from database.models import Courses, EyeTrackingSession, Statistics, User
from models.main import create_and_save_retriever_from_db, generate_rag_answer, evaluate_user_performance
import markdown
import PyPDF2
import tempfile
import os
from pathlib import Path

router = APIRouter(prefix="/edu", tags=["Education"])

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Function to get DB session
def connect():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(connect)]

class CourseCreate(BaseModel):
    title: str
    description: str
    content: str
    quiz_keywords: str = None
    total_duration: str = None

class EyeTrackingData(BaseModel):
    course_id: int
    tracking_mode: str
    total_duration: int
    tab_switches: int
    focus_status_good: int = 0
    focus_status_warning: int = 0
    focus_status_alert: int = 0
    session_data: str = None

@router.get("/", summary="Render the courses page")
async def read_root(request: Request):
    return templates.TemplateResponse("courses.html", {"request": request})

@router.post("/courses", summary="Add a new course document")
async def add_course(course: CourseCreate, db: db_dependency):
    """
    Users can add their own course documents with title, description, and content.
    The RAG system will automatically index this new content.
    """
    try:
        new_course = Courses(
            title=course.title,
            description=course.description,
            content=course.content,
            quiz_keywords=course.quiz_keywords,
            total_duration=course.total_duration
        )
        db.add(new_course)
        db.commit()
        db.refresh(new_course)
        
        # Rebuild the retriever to include the new course
        create_and_save_retriever_from_db(db)
        
        return {
            "success": True,
            "message": "Course successfully added and integrated into knowledge base.",
            "course_id": new_course.id,
            "course_title": new_course.title
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error adding course: {str(e)}")

@router.get("/courses", summary="Get all courses")
async def get_all_courses(request: Request, db: db_dependency):
    """
    Retrieves all available courses from the database.
    Returns HTML if requested from browser, JSON otherwise.
    """
    try:
        courses = db.query(Courses).all()
        
        # Check if the request accepts HTML
        accept_header = request.headers.get("accept", "")
        if "text/html" in accept_header:
            return templates.TemplateResponse("courses.html", {
                "request": request,
                "courses": courses,
                "total_courses": len(courses)
            })
        
        # Return JSON response for API calls
        return {
            "success": True,
            "total_courses": len(courses),
            "courses": [
                {
                    "id": course.id,
                    "title": course.title,
                    "description": course.description,
                    "total_duration": course.total_duration,
                    "date_created": course.date_created
                }
                for course in courses
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving courses: {str(e)}")
    

@router.get("/courses/{course_id}", summary="Get course by ID")
async def get_course_by_id(course_id: int, request: Request, db: db_dependency):
    """
    Retrieves a specific course by its ID and returns HTML with RAG-based explanation.
    For JSON responses, pass Accept: application/json header.
    """
    try:
        course = db.query(Courses).filter(Courses.id == course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found.")
        
        # Generate RAG-based explanation using course title and content
        create_and_save_retriever_from_db(db)
        
        # Create a prompt that asks for explanation of the course topic
        rag_prompt = f"Lütfen '{course.title}' konusunu detaylı bir şekilde açıkla. Tanım, önemli noktalar ve pratik örnekler ekle."
        response_text = generate_rag_answer(rag_prompt, db)
        
        # Convert markdown to HTML
        response_html = markdown.markdown(response_text, extensions=['fenced_code', 'codehilite'])
        
        # Check if the request accepts JSON
        accept_header = request.headers.get("accept", "")
        
        if "application/json" in accept_header:
            # Return JSON response for API calls (including RAG response)
            return {
                "success": True,
                "response": response_html,
                "course": {
                    "id": course.id,
                    "title": course.title,
                    "description": course.description,
                    "content": course.content,
                    "quiz_keywords": course.quiz_keywords,
                    "total_duration": course.total_duration,
                    "date_created": course.date_created
                }
            }
        else:
            # Return HTML response
            return templates.TemplateResponse("courses_detail.html", {
                "request": request,
                "lesson": {
                    "title": course.title,
                    "description": course.description,
                },
                "course": {
                    "id": course.id,
                    "title": course.title,
                    "description": course.description,
                    "content": course.content,
                    "quiz_keywords": course.quiz_keywords,
                    "total_duration": course.total_duration,
                    "date_created": course.date_created
                },
                "response": response_html
            })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving course: {str(e)}")


@router.post("/save-eye-tracking", summary="Save eye tracking session data")
async def save_eye_tracking(data: EyeTrackingData, db: db_dependency):
    """
    Saves eye tracking session data to the database after course completion.
    """
    try:
        # Create new eye tracking session record
        eye_tracking_session = EyeTrackingSession(
            course_id=data.course_id,
            user_id=None,  # Can be updated if user authentication is added
            tracking_mode=data.tracking_mode,
            total_duration=data.total_duration,
            tab_switches=data.tab_switches,
            focus_status_good=data.focus_status_good,
            focus_status_warning=data.focus_status_warning,
            focus_status_alert=data.focus_status_alert,
            session_data=data.session_data
        )
        
        db.add(eye_tracking_session)
        db.commit()
        db.refresh(eye_tracking_session)
        
        return {
            "success": True,
            "message": "Eye tracking session saved successfully.",
            "session_id": eye_tracking_session.id,
            "tracking_data": {
                "total_duration": eye_tracking_session.total_duration,
                "focus_status_good": eye_tracking_session.focus_status_good,
                "focus_status_warning": eye_tracking_session.focus_status_warning,
                "focus_status_alert": eye_tracking_session.focus_status_alert,
                "tab_switches": eye_tracking_session.tab_switches
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error saving eye tracking data: {str(e)}")


@router.get("/user-sessions", summary="Get user sessions for a course")
async def get_user_sessions(course_id: int, db: db_dependency = None):
    """
    Retrieves previous eye tracking sessions for the current user in a specific course.
    Used for comparison and trend analysis.
    """
    try:
        # For now, we get all sessions for the course (no user auth yet)
        # Once user auth is added, filter by user_id as well
        sessions = db.query(EyeTrackingSession).filter(
            EyeTrackingSession.course_id == course_id
        ).order_by(EyeTrackingSession.created_at.desc()).all()
        
        session_data = []
        for session in sessions:
            # Calculate focus percentage
            total = session.focus_status_good + session.focus_status_warning + session.focus_status_alert
            focus_good_percent = round((session.focus_status_good / total * 100) if total > 0 else 0, 2)
            
            session_data.append({
                "session_id": session.id,
                "focus_status_good": session.focus_status_good,
                "focus_status_warning": session.focus_status_warning,
                "focus_status_alert": session.focus_status_alert,
                "tab_switches": session.tab_switches,
                "focus_percentage": focus_good_percent,
                "created_at": session.created_at.isoformat() if session.created_at else None
            })
        
        return {
            "success": True,
            "total_sessions": len(session_data),
            "sessions": session_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Oturumlar alınırken hata oluştu: {str(e)}")


@router.get("/eye-tracking-sessions", summary="Get all eye tracking sessions")
async def get_eye_tracking_sessions(course_id: int = None, db: db_dependency = None):
    """
    Retrieves all eye tracking sessions from the database.
    Optionally filter by course_id.
    Returns detailed information about focus tracking metrics.
    """
    try:
        query = db.query(EyeTrackingSession)
        
        if course_id:
            query = query.filter(EyeTrackingSession.course_id == course_id)
        
        sessions = query.order_by(EyeTrackingSession.created_at.desc()).all()
        
        session_data = []
        for session in sessions:
            # Get course info
            course = db.query(Courses).filter(Courses.id == session.course_id).first()
            
            # Calculate focus percentages
            total_seconds = session.total_duration
            good_percentage = round((session.focus_status_good / total_seconds * 100) if total_seconds > 0 else 0, 2)
            warning_percentage = round((session.focus_status_warning / total_seconds * 100) if total_seconds > 0 else 0, 2)
            alert_percentage = round((session.focus_status_alert / total_seconds * 100) if total_seconds > 0 else 0, 2)
            
            session_data.append({
                "session_id": session.id,
                "course_id": session.course_id,
                "course_title": course.title if course else "Unknown",
                "tracking_mode": session.tracking_mode,
                "total_duration_seconds": session.total_duration,
                "total_duration_minutes": round(session.total_duration / 60, 2),
                "tab_switches": session.tab_switches,
                "focus_metrics": {
                    "good_focus": {
                        "seconds": session.focus_status_good,
                        "percentage": good_percentage
                    },
                    "warning_focus": {
                        "seconds": session.focus_status_warning,
                        "percentage": warning_percentage
                    },
                    "alert_focus": {
                        "seconds": session.focus_status_alert,
                        "percentage": alert_percentage
                    }
                },
                "created_at": session.created_at.isoformat() if session.created_at else None
            })
        
        return {
            "success": True,
            "total_sessions": len(session_data),
            "sessions": session_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving eye tracking sessions: {str(e)}")


@router.get("/course-documents/{course_id}", summary="Get documents for a course")
async def get_course_documents(course_id: int):
    """
    Retrieves all documents (PDFs) available for a specific course.
    Returns a list of documents with their paths and metadata.
    """
    import os
    from pathlib import Path
    
    try:
        # Get the pdfs directory
        pdfs_dir = Path("pdfs")
        
        if not pdfs_dir.exists():
            return {
                "success": True,
                "course_id": course_id,
                "documents": []
            }
        
        # Get all PDF files in the pdfs directory
        pdf_files = sorted([f.name for f in pdfs_dir.glob("*.pdf")])
        
        documents = [
            {
                "name": pdf_file,
                "path": f"/pdfs/{pdf_file}",
                "encoded_path": f"/pdfs/{pdf_file.replace(' ', '%20')}"
            }
            for pdf_file in pdf_files
        ]
        
        return {
            "success": True,
            "course_id": course_id,
            "total_documents": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dökümanlar alınırken hata oluştu: {str(e)}")


@router.post("/upload-pdf/{course_id}", summary="Upload and process PDF for RAG")
async def upload_pdf(course_id: int, file: UploadFile = None, db: db_dependency = None):
    """
    Upload a PDF file, extract text, and add to RAG retriever.
    The file is temporarily stored and processed, then added to the knowledge base.
    """
    
    if not file:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    
    temp_file_path = None
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Save temp file
        temp_file_path = temp_dir / file.filename
        contents = await file.read()
        with open(temp_file_path, 'wb') as f:
            f.write(contents)
        
        # Extract text from PDF
        extracted_text = ""
        with open(temp_file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="PDF'den metin çıkarılamadı")
        
        # Create course entry if needed and add to database
        course = db.query(Courses).filter(Courses.id == course_id).first()
        if not course:
            # Create a temporary course entry for uploaded documents
            course = Courses(
                title=f"Yüklenen: {file.filename}",
                description=f"Uploaded PDF file: {file.filename}",
                content=extracted_text,
                quiz_keywords=None,
                total_duration="Değişken"
            )
            db.add(course)
            db.commit()
            db.refresh(course)
            course_id = course.id
        else:
            # Append to existing course
            course.content += f"\n\n--- Uploaded File: {file.filename} ---\n{extracted_text}"
            db.commit()
        
        # Rebuild retriever with updated documents
        create_and_save_retriever_from_db(db)
        
        return {
            "success": True,
            "message": f"{file.filename} başarıyla yüklendi ve RAG'e eklendi",
            "filename": file.filename,
            "course_id": course_id,
            "extracted_text_length": len(extracted_text)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"PDF işlenirken hata oluştu: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file_path and Path(temp_file_path).exists():
            try:
                Path(temp_file_path).unlink()
            except Exception as cleanup_error:
                print(f"⚠️ Temp file cleanup hatası: {cleanup_error}")


@router.get("/eye-tracking-stats", summary="Get eye tracking statistics")
async def get_eye_tracking_stats(db: db_dependency = None):
    """
    Returns aggregate statistics about all eye tracking sessions.
    Shows overall focus metrics and trends.
    """
    try:
        sessions = db.query(EyeTrackingSession).all()
        
        if not sessions:
            return {
                "success": True,
                "message": "No eye tracking sessions found yet",
                "stats": {}
            }
        
        total_sessions = len(sessions)
        total_duration = sum(s.total_duration for s in sessions)
        total_tab_switches = sum(s.tab_switches for s in sessions)
        total_good_focus = sum(s.focus_status_good for s in sessions)
        total_warning_focus = sum(s.focus_status_warning for s in sessions)
        total_alert_focus = sum(s.focus_status_alert for s in sessions)
        
        # Mode breakdown
        webcam_sessions = len([s for s in sessions if s.tracking_mode == 'webcam'])
        mouse_sessions = len([s for s in sessions if s.tracking_mode == 'mouse'])
        
        return {
            "success": True,
            "stats": {
                "total_sessions": total_sessions,
                "total_study_time": {
                    "seconds": total_duration,
                    "minutes": round(total_duration / 60, 2),
                    "hours": round(total_duration / 3600, 2)
                },
                "tracking_modes": {
                    "webcam": webcam_sessions,
                    "mouse": mouse_sessions
                },
                "overall_focus_metrics": {
                    "good_focus": {
                        "seconds": total_good_focus,
                        "percentage": round((total_good_focus / total_duration * 100) if total_duration > 0 else 0, 2)
                    },
                    "warning_focus": {
                        "seconds": total_warning_focus,
                        "percentage": round((total_warning_focus / total_duration * 100) if total_duration > 0 else 0, 2)
                    },
                    "alert_focus": {
                        "seconds": total_alert_focus,
                        "percentage": round((total_alert_focus / total_duration * 100) if total_duration > 0 else 0, 2)
                    }
                },
                "total_tab_switches": total_tab_switches,
                "average_tab_switches_per_session": round(total_tab_switches / total_sessions, 2) if total_sessions > 0 else 0,
                "average_focus_duration_minutes": round(total_duration / total_sessions / 60, 2) if total_sessions > 0 else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"İstatistikler alınırken hata oluştu: {str(e)}")


@router.get("/dashboard", summary="Get comprehensive dashboard data")
async def get_dashboard_data(request: Request, db: db_dependency):
    """
    Returns comprehensive dashboard data for visualization.
    Includes courses, statistics, eye tracking, and user data.
    """
    try:
        # Courses data
        courses = db.query(Courses).all()
        courses_data = [{
            "id": c.id,
            "title": c.title,
            "description": c.description,
            "date_created": c.date_created.isoformat() if c.date_created else None
        } for c in courses]
        
        # Eye tracking sessions data
        eye_sessions = db.query(EyeTrackingSession).all()
        total_sessions = len(eye_sessions)
        
        good_focus_total = sum(s.focus_status_good for s in eye_sessions)
        warning_focus_total = sum(s.focus_status_warning for s in eye_sessions)
        alert_focus_total = sum(s.focus_status_alert for s in eye_sessions)
        total_focus_time = good_focus_total + warning_focus_total + alert_focus_total
        
        total_duration = sum(s.total_duration for s in eye_sessions)
        total_tab_switches = sum(s.tab_switches for s in eye_sessions)
        
        eye_tracking_data = {
            "total_sessions": total_sessions,
            "total_duration_hours": round(total_duration / 3600, 2) if total_duration > 0 else 0,
            "total_tab_switches": total_tab_switches,
            "avg_session_duration_minutes": round(total_duration / total_sessions / 60, 2) if total_sessions > 0 else 0,
            "focus_breakdown": {
                "good": {
                    "seconds": good_focus_total,
                    "percentage": round((good_focus_total / total_focus_time * 100) if total_focus_time > 0 else 0, 1)
                },
                "warning": {
                    "seconds": warning_focus_total,
                    "percentage": round((warning_focus_total / total_focus_time * 100) if total_focus_time > 0 else 0, 1)
                },
                "alert": {
                    "seconds": alert_focus_total,
                    "percentage": round((alert_focus_total / total_focus_time * 100) if total_focus_time > 0 else 0, 1)
                }
            }
        }
        
        # Per-course eye tracking data
        course_sessions = {}
        for session in eye_sessions:
            if session.course_id not in course_sessions:
                course_sessions[session.course_id] = {
                    "sessions": 0,
                    "duration": 0,
                    "focus_good": 0,
                    "focus_warning": 0,
                    "focus_alert": 0,
                    "tab_switches": 0
                }
            course_sessions[session.course_id]["sessions"] += 1
            course_sessions[session.course_id]["duration"] += session.total_duration
            course_sessions[session.course_id]["focus_good"] += session.focus_status_good
            course_sessions[session.course_id]["focus_warning"] += session.focus_status_warning
            course_sessions[session.course_id]["focus_alert"] += session.focus_status_alert
            course_sessions[session.course_id]["tab_switches"] += session.tab_switches
        
        # Statistics data
        statistics = db.query(Statistics).all()
        total_study_minutes = sum(s.total_study_minute for s in statistics)
        total_focus_points = sum(s.focus_points for s in statistics)
        total_completed = sum(s.completed_lessons for s in statistics)
        
        # Return HTML template if browser request, JSON otherwise
        if "text/html" in request.headers.get("accept", ""):
            return templates.TemplateResponse("statistics.html", {
                "request": request,
                "courses": courses_data,
                "eye_tracking": eye_tracking_data,
                "course_sessions": course_sessions,
                "total_study_minutes": total_study_minutes,
                "total_focus_points": total_focus_points,
                "total_completed": total_completed
            })
        else:
            return {
                "success": True,
                "courses": courses_data,
                "eye_tracking": eye_tracking_data,
                "course_sessions": course_sessions,
                "statistics": {
                    "total_study_minutes": total_study_minutes,
                    "total_focus_points": total_focus_points,
                    "total_completed_lessons": total_completed
                }
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dashboard verileri alınırken hata oluştu: {str(e)}")


@router.get("/dashboard-ai-evaluation", summary="Get AI-powered performance evaluation")
async def get_ai_evaluation(db: db_dependency):
    """
    Gets comprehensive AI-powered evaluation based on user's performance data.
    Provides personalized recommendations and insights.
    """
    try:
        # Get dashboard data first
        eye_sessions = db.query(EyeTrackingSession).all()
        total_sessions = len(eye_sessions)
        
        good_focus_total = sum(s.focus_status_good for s in eye_sessions)
        warning_focus_total = sum(s.focus_status_warning for s in eye_sessions)
        alert_focus_total = sum(s.focus_status_alert for s in eye_sessions)
        total_focus_time = good_focus_total + warning_focus_total + alert_focus_total
        
        total_duration = sum(s.total_duration for s in eye_sessions)
        total_tab_switches = sum(s.tab_switches for s in eye_sessions)
        
        dashboard_data = {
            "eye_tracking": {
                "total_sessions": total_sessions,
                "total_duration_hours": round(total_duration / 3600, 2) if total_duration > 0 else 0,
                "total_tab_switches": total_tab_switches,
                "avg_session_duration_minutes": round(total_duration / total_sessions / 60, 2) if total_sessions > 0 else 0,
                "focus_breakdown": {
                    "good": {
                        "seconds": good_focus_total,
                        "percentage": round((good_focus_total / total_focus_time * 100) if total_focus_time > 0 else 0, 1)
                    },
                    "warning": {
                        "seconds": warning_focus_total,
                        "percentage": round((warning_focus_total / total_focus_time * 100) if total_focus_time > 0 else 0, 1)
                    },
                    "alert": {
                        "seconds": alert_focus_total,
                        "percentage": round((alert_focus_total / total_focus_time * 100) if total_focus_time > 0 else 0, 1)
                    }
                }
            }
        }
        
        # Get AI evaluation
        evaluation_result = evaluate_user_performance(dashboard_data)
        
        return {
            "success": evaluation_result.get("success", False),
            "evaluation": evaluation_result.get("evaluation", ""),
            "data_summary": dashboard_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during AI evaluation: {str(e)}")


@router.get("/resources", summary="Get all resources")
async def get_resources(request: Request):
    """
    Returns all resources (PDFs) from the pdfs directory.
    Returns HTML template if browser request, JSON otherwise.
    """
    try:
        from pathlib import Path
        
        # Get the pdfs directory
        pdfs_dir = Path("pdfs")
        
        if not pdfs_dir.exists():
            documents = []
        else:
            # Get all PDF files in the pdfs directory
            pdf_files = sorted([f.name for f in pdfs_dir.glob("*.pdf")])
            
            documents = [
                {
                    "name": pdf_file,
                    "path": f"/pdfs/{pdf_file}",
                    "encoded_path": f"/pdfs/{pdf_file.replace(' ', '%20')}",
                    "size_kb": round(pdfs_dir.joinpath(pdf_file).stat().st_size / 1024, 2)
                }
                for pdf_file in pdf_files
            ]
        
        # Check if the request accepts HTML
        accept_header = request.headers.get("accept", "")
        
        if "text/html" in accept_header:
            return templates.TemplateResponse("resources.html", {
                "request": request,
                "documents": documents,
                "total_documents": len(documents)
            })
        else:
            return {
                "success": True,
                "total_documents": len(documents),
                "documents": documents
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving resources: {str(e)}")