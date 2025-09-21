from fastapi import FastAPI, UploadFile, File, HTTPException , Depends , Path
from fastapi.responses import JSONResponse
from typing import List, Optional , Dict , Any
from contextlib import asynccontextmanager
import magic
from src.main_pipeline import DocumentProcessor
from src.Retrievers import RetrieverAndChat
import mimetypes
import logging
from datetime import datetime
import time
from dataclasses import dataclass
from pydantic import BaseModel  , Field
import os
import redis

app = FastAPI(title="Legal Document Processor")

@dataclass
class DOC_PROCESSING_CONFIG:
    MAX_CHUNK_CHARS : int = 3000
    MAX_TOKEN_LIMIT : int = 2048

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="User's question or query")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results to retrieve")

class ImageChatRequest(BaseModel):
    common_id: str
    query: str

logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)
SUPPORTED_TYPES = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/msword': '.doc',
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/tiff': '.tiff',
    'text/plain': '.txt'
}

# Redis client
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logging.info(f"✅ Connected to Redis: {redis_url}")
    except Exception as e:
        logging.error(f"❌ Failed to initialize Redis: {e}")
    yield
    if redis_client:
        redis_client.close()
        logging.info("Redis connection closed")

###Retreiver Dependency
async def get_retriever():
    try:
        retriever = RetrieverAndChat()
        retriever.test_connection()
        return retriever
    except Exception as e:
        logging.error(f"Failed to initialize retriever: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Service initialization failed: {str(e)}"
        )

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

@app.post("/upload/pdfs")
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    try:
        files_data = []
        filenames = []
        for file in files:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File too large: {file.filename}")
            if file.filename is None:
                raise HTTPException(status_code=400, detail="Filename is required")
            
            file_type = detect_file_type(content, file.filename)
            if file_type not in SUPPORTED_TYPES:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type} for file {file.filename}")
            
            files_data.append(content)
            filenames.append(file.filename)
        
        processor = DocumentProcessor(mode="pdf")
        result = await processor.process_pdfs(files_data, filenames)
        return result
    except Exception as e:
        logging.error(f"Error processing multiple files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/multiple/images")
async def upload_multiple_images(files: List[UploadFile] = File(...)):
    try:
        if not files or len(files) > 10:
            raise HTTPException(status_code=400, detail="Invalid number of files. Provide 1 to 10 images.")

        valid_files_data = []
        valid_filenames = []
        
        for file in files:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                continue
            
            file_type = detect_file_type(content, file.filename or "")
            if file.filename and file_type.startswith("image/"):
                valid_files_data.append(content)
                valid_filenames.append(file.filename)
        
        if not valid_files_data:
            raise HTTPException(status_code=400, detail="No valid images found.")

        processor = DocumentProcessor()
        result = await processor.process_image(files_data=valid_files_data, filenames=valid_filenames)
        return result
    except Exception as e:
        logging.error(f"Error processing multiple images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/{collection_id}")
async def chat(
    request: ChatRequest,
    collection_id: str = Path(..., description="ID of the document collection to query"),
    retriever: RetrieverAndChat = Depends(get_retriever)
):
    try:
        result = await retriever.process_qdrant_retrieval(
            query=request.query,
            collection_name=collection_id.strip(),
            top_k=request.top_k
        )
        return JSONResponse(content=result)
            
    except Exception as e:
        logging.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/chat_image")
async def image_chat(request: ImageChatRequest, retriever: RetrieverAndChat = Depends(get_retriever)):
    try:
        result = await retriever.process_image_chat(common_id=request.common_id, user_query=request.query)
        return JSONResponse(content=result)

    except Exception as e:
        logging.error(f"Error in image chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during image chat: {str(e)}")


def detect_file_type(content: bytes, filename: str) -> str:
    try:
        return magic.from_buffer(content, mime=True)
    except Exception:
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")

