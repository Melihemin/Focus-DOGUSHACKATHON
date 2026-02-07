from typing import Annotated
from fastapi import Depends, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from database.settings import SessionLocal, engine, Base
# Routers
from routers.edu import router as edu_router
from models.main import generate_rag_answer, initialize_retriever

class RAGRequest(BaseModel):
    question: str

# Sunucu başlatılırken RAG retriever'ını hazırla
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Sunucu başlangıçında retriever'ı yükle (hız için)
    """
    import logging
    logging.info("\n" + "="*60)
    logging.info("🚀 Sunucu başlatılıyor...")
    logging.info("="*60)
    
    # Startup
    db = SessionLocal()
    try:
        logging.info("⚡ RAG Retriever hazırlanıyor (veritabanı + PDF)...")
        initialize_retriever(db)
        logging.info("✅ RAG Retriever başarıyla hazırlandı!\n")
    finally:
        db.close()
    
    yield
    
    # Shutdown
    logging.info("\n" + "="*60)
    logging.info("⏹️  Sunucu kapatılıyor...")
    logging.info("="*60)

# FastAPI run and Routers
app = FastAPI(title="Educational RAG API", version="1.0.0", lifespan=lifespan)
app.include_router(edu_router)

Base.metadata.create_all(bind=engine)

# Function to get DB session
def connect():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(connect)]

# Statik dosyaları bağla
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/rag/answer", summary="Generate RAG answer")
async def rag_answer(req: RAGRequest, db: db_dependency):
    """
    Generates an educational explanation based on user question using RAG system.
    """
    try:
        answer = generate_rag_answer(req.question, db)
        return {
            "success": True,
            "question": req.question,
            "answer": answer
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Yanıt oluşturulurken bir hata oluştu."
        }

# Uvicorn ile çalıştırma komutu
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="localhost", port=8000, reload=True)