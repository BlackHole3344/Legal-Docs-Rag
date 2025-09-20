"""
Qdrant Service Adapter - Single file RAG integration service
Provides client pooling, collection management, and batch operations
"""

import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from .embedding_service import  ChunkItem 
from .qdrant_client_pool import QdrantClientPool 
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

import dotenv 

dotenv.load_dotenv() 




@dataclass
class UploadChunk:
    """Data structure for upload chunks"""
    id: str
    vector: List[float]
    payload: Dict[str, Any]




class QdrantServiceAdapter:

    
    def __init__(self ):
        self.client_pool = QdrantClientPool()
        self.config = self.client_pool.config  
        self.collections: Dict[str, str] = {}  
        self.logger = self._setup_logger()
        ### always get called when class is initialized 
        self._test_connection()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the service"""
        logger = logging.getLogger("QdrantServiceAdapter")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _test_connection(self):
        try:
            with self.client_pool.get_client() as client:
                collections = client.get_collections()
                self.logger.info(f"✅ Connected to Qdrant. Found {len(collections.collections)} existing collections")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant: {e}")
    
    
    
    def generate_collection_name_simple(self,
                               base_name: str = "embeddings",
                               file_count: Optional[int] = None,
                               include_timestamp: bool = True,
                               include_uuid: bool = False) -> str:

        parts = [base_name]
        
        if file_count and file_count > 0:
            parts.append(f"{file_count}files")
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(timestamp)
        
        if include_uuid:
            short_uuid = str(uuid.uuid4())[:8]
            parts.append(short_uuid)
        
        collection_name = "_".join(parts)
        self.logger.info(f"📝 Generated collection name: {collection_name}")
        return collection_name
    
    
    
    def generate_collection_name(self, 
                                base_name: str = "embeddings",
                                filename: Optional[str] = None,
                                include_timestamp: bool = True,
                                include_uuid: bool = False) -> str:
        parts = [base_name]
        

        if filename:
            clean_filename = self._clean_filename_for_collection(filename)
            if clean_filename:
                parts.append(clean_filename)
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(timestamp)
        
        if include_uuid:
            short_uuid = str(uuid.uuid4())[:8]
            parts.append(short_uuid)
        
        collection_name = "_".join(parts)
        self.logger.info(f"📝 Generated collection name: {collection_name}")
        return collection_name
    
    def _clean_filename_for_collection(self, filename: str) -> str:
        
        import re
        

        name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
        

        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name_without_ext)

        cleaned = re.sub(r'_+', '_', cleaned)
        
   
        cleaned = cleaned.strip('_')

        max_length = 30
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length].rstrip('_')
   
        cleaned = cleaned.lower()
        
        return cleaned
    
    def create_collection(self, 
                         collection_name: Optional[str] = None,
                         filename: Optional[str] = None,
                         vector_size: Optional[int] = 768,
                         distance: Optional[str] = None,
                         force_recreate: bool = False) -> str:

        if collection_name is None:
            collection_name = self.generate_collection_name(filename=filename)
        
        vector_size = vector_size or self.config.vector_size
        distance = distance or self.config.distance_metric
        
        # Convert string distance to enum
        distance_enum = getattr(models.Distance, distance.upper())
        
        try:
            with self.client_pool.get_client() as client:
                # Check if collection exists
                collections = client.get_collections()
                existing_names = [col.name for col in collections.collections]
                
                if collection_name in existing_names:
                    if force_recreate:
                        self.logger.info(f"🔄 Recreating existing collection: {collection_name}")
                        client.delete_collection(collection_name)
                    else:
                        self.logger.info(f"📂 Collection {collection_name} already exists, using existing")
                        self.collections[collection_name] = collection_name
                        return collection_name
                
                # Create new collection
                self.logger.info(f"🆕 Creating collection: {collection_name} (size: {vector_size}, distance: {distance})")
                
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance_enum,
                        hnsw_config=models.HnswConfigDiff(
                            m=16,
                            ef_construct=100,
                        )
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=20000,
                    ),
                )
                
                # Store in collections registry
                self.collections[collection_name] = collection_name
                self.logger.info(f"✅ Collection created successfully: {collection_name}")
                return collection_name
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create collection {collection_name}: {e}")
            raise RuntimeError(f"Collection creation failed: {e}")
    
    #### method for qdrant batch upload 
    def batch_upload_chunks(self, 
                           collection_name: str,
                           chunks: List[UploadChunk],
                           batch_size: Optional[int] = None,
                           parallel_batches: int = 2) -> Dict[str, Any]:

   
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found. Create it first using create_collection()")
        
        batch_size = batch_size or self.config.batch_size
        total_chunks = len(chunks)
        
        self.logger.info(f"📤 Starting batch upload: {total_chunks} chunks, batch_size: {batch_size}")
        
        # Split chunks into batches
        batches = [chunks[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
        
        upload_stats = {
            "total_chunks": total_chunks,
            "total_batches": len(batches),
            "batch_size": batch_size,
            "success_count": 0,
            "error_count": 0,
            "start_time": time.time(),
            "errors": []
        }
        
        def upload_batch(batch_idx: int, batch_chunks: List[UploadChunk]) -> Tuple[int , bool , str]:
            """Upload a single batch"""
            try:
                with self.client_pool.get_client() as client:
                    points = [
                        models.PointStruct(
                            id=chunk.id,
                            vector=chunk.vector,
                            payload=chunk.payload
                        )
                        for chunk in batch_chunks
                    ]
                    
                    # Retry logic
                    for attempt in range(self.config.retry_count):
                        try:
                            client.upsert(
                                collection_name=collection_name,
                                points=points,
                                wait=True
                            )
                            return batch_idx, True, f"Batch {batch_idx + 1} uploaded successfully"
                        
                        except ResponseHandlingException as e:
                            if attempt == self.config.retry_count - 1:
                                raise e
                            self.logger.warning(f"⚠️ Batch {batch_idx + 1} attempt {attempt + 1} failed, retrying...")
                            time.sleep(1 * (attempt + 1))  # Exponential backoff
            
            except Exception as e:
                error_msg = f"Batch {batch_idx + 1} failed: {str(e)}"
                return batch_idx, False, error_msg
        
        # Upload batches in parallel
        with ThreadPoolExecutor(max_workers=parallel_batches) as executor:
            future_to_batch = {
                executor.submit(upload_batch, idx, batch): idx 
                for idx, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_idx, success, message = future.result()
                
                if success:
                    upload_stats["success_count"] += len(batches[batch_idx])
                    self.logger.info(f"✅ {message} ({upload_stats['success_count']}/{total_chunks})")
                else:
                    upload_stats["error_count"] += len(batches[batch_idx])
                    upload_stats["errors"].append(message)
                    self.logger.error(f"❌ {message}")
        
        # Calculate final statistics
        upload_stats["end_time"] = time.time()
        upload_stats["duration"] = upload_stats["end_time"] - upload_stats["start_time"]
        upload_stats["upload_rate"] = upload_stats["success_count"] / upload_stats["duration"] if upload_stats["duration"] > 0 else 0
        
        self.logger.info(f"📊 Upload completed: {upload_stats['success_count']}/{total_chunks} chunks "
                        f"in {upload_stats['duration']:.2f}s "
                        f"({upload_stats['upload_rate']:.1f} chunks/sec)")
        
        if upload_stats["error_count"] > 0:
            self.logger.warning(f"⚠️ {upload_stats['error_count']} chunks failed to upload")
        
        return upload_stats

    #### entry method 
    async def embed_and_upload_documents(self,
                                        collection_name: str,
                                        documents: List[Any],  # Langchain Documents
                                        embedding_client: Any,  # Your embedding service client
                                        batch_size: Optional[int] = None) -> Dict[str, Any]:

        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found. Create it first using create_collection()")
        
        if not documents:
            raise ValueError("No documents provided")
        
        self.logger.info(f"🔄 Starting embed and upload for {len(documents)} documents")
        
        start_time = time.time()
        
  
        chunk_items : List[ChunkItem] = []
        doc_metadata_map = {}  
        
        for doc in documents:
     
            enhanced_text = doc.page_content
            
            
            
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                chunk_id = str(uuid.uuid4())
                self.logger.warning(f"Generated chunk_id for document without one: {chunk_id}")
            

            chunk_item = ChunkItem(
                chunk_id = chunk_id,
                text =  enhanced_text
            )
            chunk_items.append(chunk_item)
            
            # Store original document metadata for later use
            doc_metadata_map[chunk_id] = doc.metadata
        
        # Create batch embed request

        
        self.logger.info(f"📤 Sending {len(chunk_items)} chunks to embedding service")
        
        try:
            # Step 2: Call embedding service
            embed_start = time.time()
            ### will handle batching internally 
            embedding_results , embedding_stats = await embedding_client.embed_batch(chunk_items)
            embed_time = time.time() - embed_start
            
            self.logger.info(f"✅ Received embeddings in {embed_time:.2f}s "
                           f"{embedding_stats['count']} embeddings")
            
            # Step 3: Create UploadChunk objects for Qdrant
            upload_chunks = []
            
            # Create mapping from response back to documents
            for embed_obj in embedding_results:
                chunk_id = embed_obj["chunk_id"]
                embedding_vector = embed_obj["embedding"]
                
                # Get original document metadata
                original_metadata = doc_metadata_map.get(chunk_id, {})
                
                # Create enhanced payload combining original metadata with embedding info
                payload = {
                    # Original document metadata
                    **original_metadata,
                    # Index for tracking
                    "embed_index": embed_obj.get("index", -1)
                }
                
                # Create UploadChunk
                upload_chunk = UploadChunk(
                    id=chunk_id,
                    vector=embedding_vector,
                    payload=payload
                )
                upload_chunks.append(upload_chunk)
            
            self.logger.info(f"🔄 Created {len(upload_chunks)} upload chunks for Qdrant")
            
            # Step 4: Upload to Qdrant using existing batch upload
            upload_stats = self.batch_upload_chunks(
                collection_name=collection_name,
                chunks=upload_chunks,
                batch_size=batch_size
            )
            
            # Step 5: Combine statistics
            total_time = time.time() - start_time
            
            combined_stats = {
                "operation": "embed_and_upload",
                "total_documents": len(documents),
                "total_time_seconds": total_time,
                
                # Embedding stats
                "embedding_stats": {
                    "processing_time_ms": embedding_stats["total_processing_time_ms"],
                    "task_type":  embedding_stats["task_type"],
                    "embed_count":  embedding_stats["count"],
                    "service_call_time": embed_time
                },
                
                # Upload stats
                "upload_stats": upload_stats,
                
                # Performance metrics
                "overall_rate": len(documents) / total_time if total_time > 0 else 0,
                "success": upload_stats["success_count"] == len(documents)
            }
            
            if combined_stats["success"]:
                self.logger.info(f"🎉 Successfully embedded and uploaded {len(documents)} documents "
                               f"in {total_time:.2f}s ({combined_stats['overall_rate']:.1f} docs/sec)")
            else:
                failed_count = upload_stats["error_count"]
                self.logger.warning(f"⚠️ Completed with {failed_count} failures out of {len(documents)} documents")
            
            return combined_stats
            
        except Exception as e:
            self.logger.error(f"❌ Embed and upload failed: {e}")
            raise RuntimeError(f"Embed and upload operation failed: {e}")
    
  