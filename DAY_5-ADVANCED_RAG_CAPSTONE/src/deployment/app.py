"""
FastAPI Application - Production-Ready API
"""

import os
import sys
import json
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

# Fix imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configuration
CONFIG_PATH = ROOT_DIR / 'config' / 'config.yaml'
LOGS_DIR = ROOT_DIR / 'logs'
DATA_DIR = ROOT_DIR / 'data'

# Load config
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

API_HOST = config['api']['host']
API_PORT = config['api']['port']

# Create directories
LOGS_DIR.mkdir(exist_ok=True)
(DATA_DIR / 'chat_logs').mkdir(parents=True, exist_ok=True)
(DATA_DIR / 'uploads').mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Chat logs path
CHAT_LOGS_PATH = DATA_DIR / 'chat_logs' / 'CHAT-LOGS.json'

# Global components
pipeline = None
memory_store = None


# Request Models
class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    use_memory: bool = True
    use_refinement: bool = True


class AskSQLRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    use_memory: bool = True


# Helper Functions
def save_chat_log(log_entry: dict):
    """Save chat interaction to log file"""
    try:
        if CHAT_LOGS_PATH.exists():
            with open(CHAT_LOGS_PATH) as f:
                logs = json.load(f)
        else:
            logs = {"interactions": []}
        
        logs["interactions"].append(log_entry)
        
        with open(CHAT_LOGS_PATH, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save chat log: {e}")


# Lifespan Event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global pipeline, memory_store
    
    # Startup
    logger.info("🚀 Starting Advanced RAG API...")
    
    try:
        from src.deployment.unified_pipeline import UnifiedRAGPipeline
        from src.memory.memory_store import MemoryStore
        
        pipeline = UnifiedRAGPipeline(str(CONFIG_PATH))
        memory_store = MemoryStore(str(CONFIG_PATH))
        
        logger.info("✅ All components initialized")
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Advanced RAG API...")


# FastAPI App
app = FastAPI(
    title="Advanced RAG Capstone System",
    version="1.0.0",
    description="Day 5 - Advanced RAG with Memory, Evaluation, and Multi-modal Support",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Advanced RAG Capstone System",
        "version": "1.0.0",
        "status": "running",
        "port": API_PORT,
        "endpoints": {
            "docs": f"http://localhost:{API_PORT}/docs - Interactive API documentation",
            "ask": "/ask - General question answering",
            "ask_sql": "/ask-sql - SQL question answering",
            "ask_image": "/ask-image - Image analysis",
            "memory": "/memory/{session_id} - Get conversation history",
            "clear_memory": "/memory/{session_id}/clear - Clear session memory",
            "health": "/health - Health check",
            "stats": "/stats - System statistics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "pipeline": pipeline is not None,
            "memory": memory_store is not None
        }
    }


@app.post("/ask")
async def ask(request: AskRequest):
    """General question answering endpoint"""
    
    logger.info(f"📝 /ask - Session: {request.session_id}, Q: {request.question}")
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.process_query(
            query=request.question,
            session_id=request.session_id,
            query_type='auto',
            use_memory=request.use_memory,
            use_refinement=request.use_refinement
        )
        
        response = {
            "success": result.get('success', True),
            "session_id": request.session_id,
            "question": request.question,
            "answer": result['answer'],
            "query_type": result.get('query_type', 'text'),
            "timestamp": result['timestamp'],
            "evaluation": {
                "confidence": result['evaluation']['confidence_score'],
                "quality": result['evaluation']['overall_quality'],
                "hallucination_detected": result['evaluation']['hallucination_detected'],
                "flags": result['evaluation']['flags']
            },
            "metadata": {
                "refined": result.get('refined', False),
                "sql": result.get('sql'),
                "row_count": result.get('row_count')
            }
        }
        
        save_chat_log({
            "endpoint": "/ask",
            "session_id": request.session_id,
            "timestamp": result['timestamp'],
            "question": request.question,
            "answer": result['answer'],
            "evaluation": response['evaluation']
        })
        
        logger.info(f"✅ /ask - Quality: {response['evaluation']['quality']}")
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"❌ /ask error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-sql")
async def ask_sql(request: AskSQLRequest):
    """SQL question answering endpoint"""
    
    logger.info(f"🗄️  /ask-sql - Session: {request.session_id}, Q: {request.question}")
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.process_query(
            query=request.question,
            session_id=request.session_id,
            query_type='sql',
            use_memory=request.use_memory,
            use_refinement=False
        )
        
        # Convert data to dict if DataFrame
        data_dict = None
        if result.get('data') is not None:
            try:
                data_dict = result['data'].to_dict('records')
            except:
                data_dict = str(result.get('data'))
        
        response = {
            "success": result.get('success', True),
            "session_id": request.session_id,
            "question": request.question,
            "answer": result['answer'],
            "sql": result.get('sql'),
            "data": data_dict,
            "row_count": result.get('row_count', 0),
            "timestamp": result['timestamp'],
            "evaluation": {
                "confidence": result['evaluation']['confidence_score'],
                "quality": result['evaluation']['overall_quality']
            }
        }
        
        save_chat_log({
            "endpoint": "/ask-sql",
            "session_id": request.session_id,
            "timestamp": result['timestamp'],
            "question": request.question,
            "sql": result.get('sql'),
            "answer": result['answer'],
            "row_count": result.get('row_count', 0)
        })
        
        logger.info(f"✅ /ask-sql - Rows: {response['row_count']}")
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"❌ /ask-sql error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-image")
async def ask_image(
    file: UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form("default")
):
    """Image analysis endpoint"""
    
    logger.info(f"🖼️  /ask-image - Session: {session_id}, Q: {question}")
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        upload_dir = DATA_DIR / 'uploads'
        file_path = upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        result = pipeline.process_query(
            query=question,
            session_id=session_id,
            query_type='image',
            image_path=str(file_path),
            use_memory=True,
            use_refinement=False
        )
        
        response = {
            "success": result.get('success', True),
            "session_id": session_id,
            "question": question,
            "answer": result['answer'],
            "image_path": str(file_path),
            "caption": result.get('caption'),
            "timestamp": result['timestamp'],
            "evaluation": {
                "confidence": result['evaluation']['confidence_score'],
                "quality": result['evaluation']['overall_quality']
            }
        }
        
        save_chat_log({
            "endpoint": "/ask-image",
            "session_id": session_id,
            "timestamp": result['timestamp'],
            "question": question,
            "answer": result['answer'],
            "image": file.filename
        })
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"❌ /ask-image error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{session_id}")
async def get_memory(session_id: str, limit: Optional[int] = 10):
    """Get conversation history"""
    
    if memory_store is None:
        raise HTTPException(status_code=503, detail="Memory store not initialized")
    
    try:
        history = memory_store.get_history(session_id, limit=limit)
        stats = memory_store.get_session_stats(session_id)
        
        return {"session_id": session_id, "history": history, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{session_id}/clear")
async def clear_memory(session_id: str):
    """Clear conversation history"""
    
    if memory_store is None:
        raise HTTPException(status_code=503, detail="Memory store not initialized")
    
    try:
        memory_store.clear_session(session_id)
        return {"success": True, "session_id": session_id, "message": "Memory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    
    try:
        sessions = memory_store.get_all_sessions() if memory_store else []
        
        total_interactions = 0
        if CHAT_LOGS_PATH.exists():
            with open(CHAT_LOGS_PATH) as f:
                logs = json.load(f)
                total_interactions = len(logs.get('interactions', []))
        
        return {
            "sessions": {"total": len(sessions), "session_ids": sessions[:10]},
            "interactions": {"total": total_interactions}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"🚀 ADVANCED RAG CAPSTONE API")
    print(f"{'='*80}")
    print(f"📂 Working directory: {ROOT_DIR}")
    print(f"📄 Config: {CONFIG_PATH}")
    print(f"🌐 Port: {API_PORT}")
    print(f"{'='*80}\n")
    
    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=config['api']['reload'],
        log_level="info"
    )
