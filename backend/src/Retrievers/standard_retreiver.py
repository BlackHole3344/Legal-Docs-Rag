
from ..qdrant_client_pool import QdrantClientPool
from ..redis_cache import RedisImageCache 
from ..embedding_service import EmbeddingClient , ChunkItem  
from .gemini_augumentator import GeminiQADataAugmentor 
from google import genai
from dotenv import load_dotenv 
from google.genai.types import HttpOptions
from google.auth import load_credentials_from_file 
from google.genai.types import HttpOptions, Content, Part, GenerateContentConfig, ThinkingConfig
from google.auth import default 
from ..image_processor import ImageProcessor 


import os 
import logging 

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
print(os.path.abspath("key.json"))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("key.json")

load_dotenv() 
_ , PROJECT_ID = default() 
LOCATION = "us-central1" 
logger = logging.getLogger(__name__)



class RetrieverAndChat:
    """Main orchestrator for retrieval and chat functionality"""
    
    def __init__(self):
        logger.info("Initializing RetrieverAndChat")
        
        self.qdrant_pool = QdrantClientPool()
        self._gemini_client = None
        self.embedding_client = EmbeddingClient()
        self.gemini_generation = GeminiQADataAugmentor() 
        
        logger.info("RetrieverAndChat initialized successfully")
    
    def test_connection(self):
        """Test Qdrant connection"""
        try:
            logger.info("Testing Qdrant connection")
            
            with self.qdrant_pool.get_client() as client:
                collections = client.get_collections()
                logger.info(f"✅ Connected to Qdrant. Found {len(collections.collections)} existing collections")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant: {e}")
    
    async def process_qdrant_retrieval(self, query: str, collection_name: str, top_k: int = 3):
        """Process Qdrant retrieval and generate response"""
        logger.info(f"Processing retrieval for query: '{query[:50]}...' in collection: {collection_name}")
        
        try:
            # Step 1: Embed the query
            logger.info("Step 1: Embedding the query")
            async with self.embedding_client as client:
                query_embedding, total_processing_time = await client.embed_single(text=query)
                logger.info(f"Query embedded successfully (processing time: {total_processing_time:.2f}s)")
            
            # Step 2: Query Qdrant
            logger.info(f"Step 2: Querying Qdrant for top {top_k} results")
            with self.qdrant_pool.get_client() as client:
                points = client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    limit=top_k,
                    with_payload=True,  # Include metadata/payload
                    with_vectors=False  # Set to True if you need the vectors back, needed later for caching
                )
                
                if hasattr(points, 'points'):
                    actual_points = points.points
                    logger.info(f"Retrieved {len(actual_points)} points from Qdrant")
                else:
                    actual_points = points
                    logger.info(f"Retrieved {len(actual_points)} points from Qdrant")

            ##forbidden
            # # Step 3: Initialize Gemini if needed
            # if not self.gemini_generation:
            #     logger.info("Step 3: Initializing Gemini generation client")
            #     await self.get_generation_client()
            
            # Step 4: Generate response
            logger.info("Step 4: Generating comprehensive response")
            generated_response , citation_data = await self.gemini_generation.generate_answer(
                user_query=query, 
                points=actual_points 
            )
            
            logger.info("Response generated successfully")
            return {
                "status": "success",
                "response": generated_response,
                "metadata": {
                    "query": query,
                    "collection": collection_name,
                    "retrieved_points": len(actual_points),
                    "embedding_time": total_processing_time , 
                    "citation_data" : citation_data 
                }
            }
            
        except Exception as e:
            logger.error(f"Error in process_qdrant_retrieval: {e}")
            return {
                "status": "error",
                "error": str(e),
                "response": "Sorry, I encountered an error while processing your request."
            }
    
    async def process_input(self, query : str, collection_name: str = "default_collection", top_k: int = 5):
        """Process user input and return response"""
        logger.info(f"Processing input: '{query[:50]}...'")
        
        try:
            self.test_connection()
            
            
            
            
            result = await self.process_qdrant_retrieval(
                query=query,
                collection_name=collection_name,
                top_k=top_k
            )
            
            logger.info("Input processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in process_input: {e}")
            return {
                "status": "error",
                "error": str(e),
                "response": "I apologize, but I'm having trouble processing your request right now."
            }
                    
                                
    async def process_image_chat(self, common_id: str, user_query: str):

        logger.info(f"Processing image chat for common_id: {common_id}")
        
        try:
   

            # Initialize image processor for cache access
            from ..image_processor import ImageProcessor
            image_processor = ImageProcessor()
            
            # Generate response using cached images
            
            
            response = await self.gemini_generation.generate_image_chat_response(
                common_id=common_id,
                user_query=user_query,
                cache_client=image_processor.cache
            )
            
            logger.info("Image chat processing completed successfully")
            return {
                "status": "success",
                "response": response,
                "metadata": {
                    "common_id": common_id,
                    "query": user_query,
                    "type": "image_chat"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in process_image_chat: {e}")
            return {
                "status": "error",
                "error": str(e),
                "response": "I apologize, but I'm having trouble processing your image chat request right now."
            }

                            
                
                
                
                
                
                
                
                
                
                
          
        
    

        
        