from fastapi import FastAPI, UploadFile, File, HTTPException , Depends , Path 
# from fastapi.responses import JSONResponse
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
import asyncio 
import redis 

app = FastAPI(title="Legal Document Processor")

# To start the FastAPI server at a particular port, use:
# uvicorn main:app --host 0.0.0.0 --port 8080
# (Run this command in your terminal from the src directory)




@dataclass 
class DOC_PROCESSING_CONFIG:
    MAX_CHUNK_CHARS : int = 3000 
    MAX_TOKEN_LIMIT : int = 2048 
    
     
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="User's question or query")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What does my insurance cover for car damage?",
                "top_k": 5
            }
        }
class ChatResponse(BaseModel):
    status: str
    response: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "response": "Your insurance covers damage from accidents, theft, storms, and fires...",
                "metadata": {
                    "query": "What does my insurance cover?",
                    "collection": "insurance_docs",
                    "retrieved_points": 5,
                    "embedding_time": 0.15
                },
                "processing_time": 2.34
            }
        }    
class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    message: str
    processing_time: Optional[float] = None
    
    


    

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

async def clear_redis_cache():
    """Clear all Redis cache data"""
    try:
        if redis_client:
            redis_client.flushdb()  # Clear current database
            logging.info(f"✅ Redis cache cleared at {datetime.now()}")
        else:
            logging.warning("Redis client not available")
    except Exception as e:
        logging.error(f"❌ Failed to clear Redis cache: {e}")

async def cache_cleaner_task():
    """Background task that runs every 10 minutes to clear cache"""
    while True:
        try:
            await asyncio.sleep(600)  # 600 seconds = 10 minutes
            await clear_redis_cache()
        except Exception as e:
            logging.error(f"Cache cleaner task error: {e}")
            await asyncio.sleep(600)  # Continue even if there's an error
@asynccontextmanager
async def lifespan(app: FastAPI):

    global redis_client

    try:

        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        redis_client = redis.from_url(redis_url, decode_responses=True)
        
  
        redis_client.ping()
        logging.info(f"✅ Connected to Redis: {redis_url}")
      
        asyncio.create_task(cache_cleaner_task())
        logging.info("🧹 Cache cleaner started - will clear every 10 minutes")
        
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
        # Test connection on first use
        retriever.test_connection()
        return retriever
    except Exception as e:
        logging.error(f"Failed to initialize retriever: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Service initialization failed: {str(e)}"
        )    


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit
@app.post("/upload/pdf")
async def upload_single_document(file = File(...)):
    try:
        logging.info(f"Received file upload: {file.filename}")

        content = await file.read()
        logging.info(f"Read {len(content)} bytes from file: {file.filename}")


        if len(content) > MAX_FILE_SIZE:
            logging.warning(f"File too large: {file.filename} ({len(content)} bytes)")
            raise HTTPException(status_code=413, detail="File too large")


        if file.filename is None:
            logging.error("Filename is missing in upload")
            raise HTTPException(status_code=400, detail="Filename is required")
        file_type = detect_file_type(content, file.filename)
        logging.info(f"Detected file type for {file.filename}: {file_type}")

        if file_type not in SUPPORTED_TYPES:
            logging.warning(f"Unsupported file type: {file_type} for file {file.filename}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_type}"
            )

        # Process the file
        logging.info(f"Processing document: {file.filename}")
        
        ### implement main_pipeline 
        if not file_type.startswith("image/"):
            logging.debug("Using GeminiOCRProcessor for image file.")
            processor = DocumentProcessor(mode="pdf") 
            result = await processor.process_pdfs(files=[content] , filenames  = [file.filename])


        return result 

    except Exception as e:
        logging.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/pdfs")
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    try:
        logging.info(f"Received {len(files)} files for processing")
        
        files_data = []
        filenames = []
        
        # Read and validate all files first
        for file in files:
            logging.info(f"Processing upload: {file.filename}")
            content = await file.read()
            logging.info(f"Read {len(content)} bytes from file: {file.filename}")
            
            # File size validation
            if len(content) > MAX_FILE_SIZE:
                logging.warning(f"File too large: {file.filename} ({len(content)} bytes)")
                raise HTTPException(
                    status_code=413, 
                    detail=f"File too large: {file.filename}"
                )
            
            if file.filename is None:
                logging.error("Filename is missing in upload")
                raise HTTPException(status_code=400, detail="Filename is required")
            
            # File type validation
            file_type = detect_file_type(content, file.filename)
            logging.info(f"Detected file type for {file.filename}: {file_type}")
            
            if file_type not in SUPPORTED_TYPES:
                logging.warning(f"Unsupported file type: {file_type} for file {file.filename}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_type} for file {file.filename}"
                )
            
            files_data.append(content)
            filenames.append(file.filename)
        
        # Process all files
        processor = DocumentProcessor(mode="pdf")
        result = await processor.process_pdfs(files_data, filenames)
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logging.error(f"Error processing multiple files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



# @app.post("/upload/single/image")
# async def upload_single_image(file: UploadFile = File(...)):
#     try:
#         logging.info(f"Received image upload: {file.filename}")
#         content = await file.read()
#         logging.info(f"Read {len(content)} bytes from image: {file.filename}")
        
#         if len(content) > MAX_FILE_SIZE:
#             logging.warning(f"Image too large: {file.filename} ({len(content)} bytes)")
#             raise HTTPException(status_code=413, detail="Image too large")
        
#         if file.filename is None:
#             logging.error("Filename is missing in upload")
#             raise HTTPException(status_code=400, detail="Filename is required")
        
#         file_type = detect_file_type(content, file.filename)
#         logging.info(f"Detected file type for {file.filename}: {file_type}")
        
#         if not file_type.startswith("image/"):
#             logging.warning(f"Not an image file: {file_type} for file {file.filename}")
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"File must be an image. Detected type: {file_type}"
#             )
        
#         # Process single image using updated method
#         processor = DocumentProcessor()
#         result = await processor.process_image(
#             files_data=[content],  # Single image in list
#             filenames=[file.filename]  # Single filename in list
#         )
        
#         return result
        
#     except Exception as e:
#         logging.error(f"Error processing image {file.filename}: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# Updated multiple images endpoint
@app.post("/upload/multiple/images") 
async def upload_multiple_images(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 10:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Too many files. Maximum 10 images allowed")
        
        logging.info(f"Received {len(files)} images for processing")
        
        # Collect all valid images
        valid_files_data = []
        valid_filenames = []
        errors = []
        
        logging.info(f"Validating {len(files)}for processing")
        for i, file in enumerate(files):
            try:
                if file.filename is None:
                    errors.append(f"File {i+1} has no filename")
                    continue
                
                content = await file.read()
                
                if len(content) > MAX_FILE_SIZE:
                    errors.append(f"{file.filename}: Image too large")
                    continue
                
                file_type = detect_file_type(content, file.filename)
                if not file_type.startswith("image/"):
                    errors.append(f"{file.filename}: Not an image file")
                    continue
                
                valid_files_data.append(content)
                valid_filenames.append(file.filename)
                
            except Exception as e:
                errors.append(f"{file.filename or f'file_{i+1}'}: {str(e)}")
        
        if not valid_files_data:
            raise HTTPException(status_code=400, detail=f"No valid images found. Errors: {errors}")
        
        # Process all valid images together
        processor = DocumentProcessor()
        logging.info(f"Generating Results for {len(files)} images")
        result = await processor.process_image(
            files_data=valid_files_data,
            filenames=valid_filenames
        )
        
  
        if errors:
            result["validation_errors"] = errors
            result["processed_count"] = len(valid_files_data)
            result["total_submitted"] = len(files)
        
        logging.info(f"Successfully processed {len(valid_files_data)} out of {len(files)} images")
        return result
        
    except Exception as e:
        logging.error(f"Error processing multiple images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def get_cache_stats():
    """Get Redis cache statistics"""
    try:
        processor = DocumentProcessor()
        stats = processor.image_processor.cache.get_stats()
        return {
            "status": "success",
            "cache_stats": stats
        }
    except Exception as e:
        logging.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache/images/{image_hash}")
async def delete_cached_image(image_hash: str):
    """Delete specific cached image"""
    try:
        processor = DocumentProcessor()
        processor.image_processor.cache.delete(image_hash)
        return {
            "status": "success",
            "message": f"Deleted cached image: {image_hash}"
        }
    except Exception as e:
        logging.error(f"Error deleting cached image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/images/{common_id}")
async def get_cached_images(common_id: str):
    """Retrieve cached image data by common_id (works for single or multiple images)"""
    try:
        processor = DocumentProcessor()
        cached_data = processor.image_processor.cache.get(common_id)
        
        if not cached_data:
            raise HTTPException(status_code=404, detail="Cached images not found")
        
        # Return without base64 data for API response
        return {
            "common_id": common_id,
            "filenames": cached_data.get("filenames", []),
            "summary": cached_data.get("summary"),
            "processed_at": cached_data.get("processed_at"),
            "source": cached_data.get("source", "cache"),
            "image_count": cached_data.get("image_count", len(cached_data.get("filenames", []))),
            "individual_hashes": cached_data.get("individual_hashes", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error retrieving cached images {common_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache/images/{common_id}")
async def delete_cached_images(common_id: str):
    """Delete cached image data by common_id"""
    try:
        processor = DocumentProcessor()
        processor.image_processor.cache.delete(common_id)
        
        return {
            "status": "success",
            "message": f"Deleted cached images: {common_id}"
        }
        
    except Exception as e:
        logging.error(f"Error deleting cached images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))








####chat endpoints 
@app.get(
    "/chat/{collection_id}",
    response_model=ChatResponse,
    summary="Chat with Legal Documents",
    description="Send a query to chat with documents in a specific collection and get demystified legal answers",
    responses={
        200: {"model": ChatResponse, "description": "Successful response with legal analysis"},
        400: {"model": ErrorResponse, "description": "Bad request - invalid input"},
        404: {"model": ErrorResponse, "description": "Collection not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
async def chat(
    request: ChatRequest ,
    collection_id: str = Path(..., description="ID of the document collection to query"),
    retriever: RetrieverAndChat = Depends(get_retriever)
):
   
    start_time = time.time()
    
    logging.info(f"Received chat request for collection '{collection_id}' with query: '{request.query[:50]}...'")
    
    try:
        # Validate collection_id
        if not collection_id or len(collection_id.strip()) == 0:
            logging.warning("Empty collection_id provided")
            raise HTTPException(
                status_code=400,
                detail="Collection ID cannot be empty"
            )
        
        # Validate query
        if not request.query or len(request.query.strip()) == 0:
            logging.warning("Empty query provided")
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Clean inputs
        clean_collection_id = collection_id.strip()
        clean_query = request.query.strip()
        
        logging.info(f"Processing request - Collection: {clean_collection_id}, Query length: {len(clean_query)}, Top-K: {request.top_k}")
        
        # Process the request using retriever
        result = await retriever.process_qdrant_retrieval(
            query=clean_query,
            collection_name=clean_collection_id,
            top_k=request.top_k
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Handle different result statuses
        if result["status"] == "success":
            logging.info(f"Chat request completed successfully in {processing_time:.2f}s")
            
            return ChatResponse(
                status="success",
                response=result["response"],
                metadata=result.get("metadata", {}),
                processing_time=processing_time
            )
            
        elif result["status"] == "error":
            logging.error(f"Chat request failed: {result.get('error', 'Unknown error')}")
            
            # Determine appropriate HTTP status code based on error type
            error_message = result.get("error", "Unknown error occurred")
            
            if "collection" in error_message.lower() and ("not found" in error_message.lower() or "does not exist" in error_message.lower()):
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection '{clean_collection_id}' not found"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Processing error: {error_message}"
                )
        
        else:
            logging.error(f"Unexpected result status: {result.get('status', 'unknown')}")
            raise HTTPException(
                status_code=500,
                detail="Unexpected processing result"
            )
            
    except HTTPException:
        raise
        
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "An unexpected error occurred while processing your request",
                "processing_time": processing_time
            }
        )
        
@app.get("/chat_image")
async def image_chat(request: Dict[str, Any]):
    try:
        common_id = request.get("common_id")
        query = request.get("query")
        
        if not common_id:
            raise HTTPException(status_code=400, detail="common_id is required")
        
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        

        retriever = RetrieverAndChat()
        
    
        result = await retriever.process_image_chat(common_id=common_id, user_query=query)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in image chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))        
        
        

# @app.get("/collections")
# async def list_collections(retriever: RetrieverAndChat = Depends(get_retriever)):
#     """List available collections"""
#     try:
#         with retriever.qdrant_pool.get_client() as client:
#             collections = client.get_collections()
#             collection_names = [col.name for col in collections.collections]
            
#             logging.info(f"Retrieved {len(collection_names)} collections")
            
#             return {
#                 "status": "success",
#                 "collections": collection_names,
#                 "count": len(collection_names)
#             }
            
#     except Exception as e:
#         logging.error(f"Failed to list collections: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to retrieve collections: {str(e)}"
#         )


























def detect_file_type(content: bytes, filename: str) -> str:
    """
    Detect file type using both content analysis and filename
    """
    # Method 1: Use python-magic (most reliable)
    try:
        mime_type = magic.from_buffer(content, mime=True)
        return mime_type
    except:
        pass
    
    # Method 2: Use filename extension
    try:
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type
    except:
        pass
    
    # Method 3: Check file signature (magic numbers)
    if content.startswith(b'%PDF'):
        return 'application/pdf'
    elif content.startswith(b'PK'):  # ZIP-based formats (docx, etc.)
        if filename.lower().endswith('.docx'):
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    elif content.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    elif content.startswith(b'\x89PNG'):
        return 'image/png'
    
    return 'application/octet-stream'  # Unknown type




if __name__ == "__main__":
    import uvicorn
    import uuid
   
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")