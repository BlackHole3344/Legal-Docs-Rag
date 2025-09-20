


from dataclasses import dataclass, asdict 
from typing import List, Optional , Dict , Any , Union , Tuple 
import asyncio
from datetime import datetime
import aiohttp 
import time 
import logging 
import uuid 
from pydantic import BaseModel , field_validator , Field 
from langchain_core.documents import Document 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 
import os 
from enum import Enum
from dotenv import load_dotenv 

load_dotenv() 

EMBEDDING_BASE_URL = os.environ["EMBEDDING_SERVICE"]

class EmbeddingType(str, Enum):
    QUERY = "query"
    DOCUMENT = "document"
    
class EmbeddingConfig:
    MODEL_ID = "onnx-community/embeddinggemma-300m-ONNX"
    DEFAULT_TASK_TYPE = os.getenv("DEFAULT_TASK_TYPE", EmbeddingType.DOCUMENT.value)
    DEFAULT_DIMENSIONS = int(os.getenv("DEFAULT_DIMENSIONS", "768"))
    MAX_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "2048"))
    BATCH_SIZE_LIMIT = int(os.getenv("BATCH_SIZE_LIMIT", "100"))

    
class ChunkItem(BaseModel):
    chunk_id: str = Field(..., description="Unique ID for the text chunk")
    text: str = Field(..., min_length=1, description="max it can go upto 2048 tokens")

    ## if no chunk_id provided, generate a UUID 
    def __init__(self, **data):
        if 'chunk_id' not in data or not data['chunk_id']:
            data['chunk_id'] = str(uuid.uuid4())
        super().__init__(**data)

        
class EmbedRequest(BaseModel):
    chunk : ChunkItem 
    task_type: Optional[str] = Field(EmbeddingConfig.DEFAULT_TASK_TYPE)
    truncate_dim: Optional[int] = Field(EmbeddingConfig.DEFAULT_DIMENSIONS)
    title: Optional[str] = Field("Testing", description="Document title for document embeddings")
    
    @field_validator('truncate_dim')
    def validate_dimensions(cls, v):
        if v not in [128, 256, 512, 768]:
            raise ValueError('truncate_dim must be one of: 128, 256, 512, 768')
        return v
    
    @field_validator('task_type')
    def validate_task_type(cls, v):
        valid_tasks = [
            "query" , "document"
        ]
        if v not in valid_tasks:
            raise ValueError(f'task_type must be one of: {valid_tasks}')
        return v
        
        
        
class BatchEmbedRequest(BaseModel):
    chunks : List[ChunkItem] = Field(
        ..., 
        description="List of ChunkItem to embed",
        min_length=1,
        max_length=EmbeddingConfig.BATCH_SIZE_LIMIT
    )
    truncate_dim: Optional[int] = Field(EmbeddingConfig.DEFAULT_DIMENSIONS)
    title: Optional[str] = Field(None, description="Title for document embeddings (same length as texts)")
    task_type: Optional[str] = Field(EmbeddingConfig.DEFAULT_TASK_TYPE) 
    
    
    @field_validator('truncate_dim')
    def validate_dimensions(cls, v):
        if v not in [128, 256, 512, 768]:
            raise ValueError('truncate_dim must be one of: 128, 256, 512, 768')
        return v
    
    
class EmbeddingClient:

    def __init__(self):
        self.base_url = str(EMBEDDING_BASE_URL).rstrip('/')
        self.session = None
        self.batch_size = EmbeddingConfig.BATCH_SIZE_LIMIT 
        self.max_retries = 3 
        
    async def __aenter__(self):
        # Generous timeout for cold starts, but simple setup
        timeout = aiohttp.ClientTimeout(total=120)  # 2 minutes for cold starts
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Simple health check with cold start tolerance"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('status') == 'healthy'
                return False
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def embed_single(self, 
                          text: str, 
                          task_type: str = "document",
                          truncate_dim: int = 768) -> Tuple[List[float] , float]:
  
        
        chunk_item = ChunkItem(text=text) 
        payload = EmbedRequest(
            chunk=chunk_item , 
            task_type="query" 
        )
        
        total_processing_time_ms = 0 
        
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/embed",
                json=payload.model_dump(),
                headers={'Content-Type': 'application/json'}
            ) as response:
                        if response.status == 200:
                            embedding_response = await response.json()
                            query_embedding = embedding_response["embedding_data"]["embedding"]
                            total_processing_time_ms += embedding_response["total_processing_time_ms"]
                            logger.info(f"Successfully Embedded The Query. Server processing time: {embedding_response.get('total_processing_time_ms', 'N/A')}ms.")
                            return query_embedding , total_processing_time_ms 
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientResponseError(
                                response.request_info, 
                                response.history, 
                                status=response.status, 
                                message=error_text
                            )
                            
                    
        except asyncio.TimeoutError:
            raise Exception("Request timed out - service might be cold starting (this is normal for prototypes)")
        except Exception as e:
            raise Exception(f"Embedding request failed: {e}")
            
    async def embed_batch(self, chunk_items: List[ChunkItem]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    
        all_results = []
        total_processing_time_ms: float = 0.0
        process_chunks_count: int = 0
        total_chunks = len(chunk_items)
        total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
        logger.info(f"Starting batch embedding for {total_chunks} chunks, divided into {total_batches} batches of size {self.batch_size}.")
        
        batch_embedding_stats = {
            "total_processing_time_ms": 0.0,  # Initialize as float, not type
            "count": 0,  # Initialize as int, not type  
            "task_type": EmbeddingConfig.DEFAULT_TASK_TYPE
        }
        
        for i in range(0, len(chunk_items), self.batch_size):
            batch_chunks = chunk_items[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} (chunks {i}-{i + len(batch_chunks) - 1})...")
            
            for attempt in range(self.max_retries):
                try:
                    payload = BatchEmbedRequest(
                        chunks=batch_chunks,
                        truncate_dim=EmbeddingConfig.DEFAULT_DIMENSIONS,
                        title="TESTING",
                        task_type=EmbeddingConfig.DEFAULT_TASK_TYPE
                    )
                    logger.info(f"Attempt {attempt + 1}: Sending POST request to '{self.base_url}/v1/embed/batch' with {len(batch_chunks)} chunks.")
                    # Fixed endpoint URL - added /v1 prefix
                    async with self.session.post(
                        f"{self.base_url}/v1/embed/batch", 
                        json=payload.model_dump(),
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        if response.status == 200:
                            embedding_response = await response.json()
                            process_chunks_count += embedding_response["count"]
                            total_processing_time_ms += embedding_response["total_processing_time_ms"]
                            all_results.extend(embedding_response["embeddings_data"])
                            logger.info(f"Successfully processed batch {batch_num}/{total_batches}. Server processing time: {embedding_response.get('total_processing_time_ms', 'N/A')}ms.")
                            # Break from retry loop on success
                            break
                            
                        else:
                            error_text = await response.text()
                            logger.error(f"Batch {batch_num} failed with status {response.status}: {error_text}")
                            raise aiohttp.ClientResponseError(
                                response.request_info, 
                                response.history, 
                                status=response.status, 
                                message=error_text
                            )
                            
                except asyncio.TimeoutError as e:
        
                    retry_delay = 2 ** attempt + 5
                    logger.warning(f"Timeout on attempt {attempt + 1} for batch {batch_num}. Likely a cold start. Retrying in {retry_delay} seconds...")
                    if attempt == self.max_retries - 1:
                        # LOGGING: Final failure
                        logger.error(f"Request for batch {batch_num} timed out after {self.max_retries} attempts. Raising exception.")
                        raise Exception(f"Request timed out after {self.max_retries} attempts. Original error: {e}")
                    await asyncio.sleep(retry_delay)    
                    
                except aiohttp.ClientResponseError as e:
         
                  # LOGGING: Enhanced retry logging for general exceptions
                    retry_delay = 2 ** attempt
                    logger.warning(f"General error on attempt {attempt + 1} for batch {batch_num}: {e}. Retrying in {retry_delay} seconds...")
                    if attempt == self.max_retries - 1:
                        # LOGGING: Final failure
                        logger.error(f"Batch {batch_num} failed after {self.max_retries} attempts. Raising final exception: {e}")
                        raise e
                    await asyncio.sleep(retry_delay)
                    
                except Exception as e:
            
                    if attempt == self.max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        

        batch_embedding_stats["count"] = process_chunks_count
        batch_embedding_stats["total_processing_time_ms"] = total_processing_time_ms
        
        return all_results, batch_embedding_stats
        
    




 
# if __name__ == "__main__":
    
     
    
