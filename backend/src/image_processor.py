import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional , List 
from dotenv import load_dotenv 
from google import genai
from google.genai.types import HttpOptions
from google.auth import load_credentials_from_file 
from google.genai.types import HttpOptions, Content, Part, GenerateContentConfig, ThinkingConfig
from google.auth import default 
from .redis_cache import RedisImageCache 
from .gemini_client import GeminiClient 
# Configuration

# PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
# print(os.path.abspath("key.json"))

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("key.json")

load_dotenv() 
# _ , PROJECT_ID = default() 
# LOCATION = "us-central1" 
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")  # Changed for local testing
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

class ImageProcessor:

    
    def __init__(self):
        self._gemini_client = GeminiClient()
        # self.cache_dir = Path(CACHE_DIR)
        # self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = RedisImageCache(host=REDIS_HOST, port=REDIS_PORT)
        
  
    
    def _get_image_hash(self, image_data: bytes) -> str:
        return hashlib.sha256(image_data).hexdigest()

    

            
    
    async def process_image(self, images_data: List[bytes], filenames: List[str]) -> Dict[str, Any]:
   
        try:
     
            individual_hashes = []
            logging.info(f"Generating individual hashes for images")
            for i ,  image_data in enumerate(images_data):
                image_hash = self._get_image_hash(image_data)
                logging.info(f"Hash generated for {i} : Image")
                individual_hashes.append(image_hash)
            
            logger.info(f"Generating Common_ID Hash for {len(individual_hashes)} individual hashes")
            combined_hash_string = "|".join(sorted(individual_hashes))
            common_id = hashlib.sha256(combined_hash_string.encode()).hexdigest()
            
            logger.info(f"Processing {len(images_data)} image(s) with common_id: {common_id}")
            
            cached_result = self.cache.get(common_id)
            if cached_result:
                logger.info(f"Cache hit for common_id: {common_id}")
                return {
                    "status": "success",
                    "filenames": filenames,
                    "summary": cached_result["summary"],
                    "processed_at": cached_result["processed_at"],
                    "source": "cache",
                    "common_id": common_id,
                    "image_count": len(images_data)
                }
            
    

            

            if len(images_data) == 1:
                prompt = """
                Analyze this image and provide a brief summary. If the image contains:
                - Tables: Quote key data points and reference them as "Table 1", "Table 2", etc.
                - Charts/Graphs: Describe the data and reference as "Chart 1", "Figure 1", etc.  
                - Text content: Extract important information and organize it clearly
                
                Format your response as:
                1. Brief overview of document type and purpose
                2. Key information extracted with proper references
                3. Important data points or findings
                
                Keep the summary concise but comprehensive for search and retrieval purposes.
                """
            else:
                prompt = f"""
                Analyze these {len(images_data)} images and provide a combined summary. For each image, if it contains:
                - Tables: Quote key data points and reference them as "Image 1 Table 1", "Image 2 Table 1", etc.
                - Charts/Graphs: Describe the data and reference as "Image 1 Chart 1", "Image 2 Figure 1", etc.  
                - Text content: Extract important information and organize it clearly with image references
                
                Format your response as:
                1. Brief overview of all documents and their relationship
                2. Key information extracted from each image with proper references (Image 1, Image 2, etc.)
                3. Important data points or findings across all images
                4. Any connections or patterns between the images
                
                Keep the combined summary comprehensive but organized for search and retrieval purposes.
                """

            message_parts = [Part.from_text(text=prompt)]
            
  
            for i, image_data in enumerate(images_data):
                filename = filenames[i] if i < len(filenames) else f"image_{i+1}"
                mime_type = self._detect_mime_type(image_data, filename)
                message_parts.append(Part.from_bytes(data=image_data, mime_type=mime_type))
                logger.debug(f"Added image {i+1} ({filename}) to message parts")
            

            contents = [Content(role="user", parts=message_parts)]
 
            response = await self._gemini_client.generate_response(
                contents=contents,
                temperature=0.1 
            )
  

            result = {
                "status": "success",
                "filenames": filenames,
                "summary": response,
                "processed_at": datetime.now().isoformat(),
                "source": "processed",
                "common_id": common_id,
                "image_count": len(images_data)
            }
            
            ####A combined payload for single and multiple images
            cache_data = {
                **result,
                "images_base64": [base64.b64encode(img_data).decode('utf-8') for img_data in images_data],
                "individual_hashes": individual_hashes
            }
            

            ttl_seconds = 604800 * 2 if len(images_data) > 1 else 604800  
            self.cache.set(common_id, cache_data, ttl_seconds)
            
            logger.info(f"Successfully processed {len(images_data)} image(s): {filenames}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing images {filenames}: {e}")
            return {
                "status": "error",
                "filenames": filenames,
                "error": str(e),
                "processed_at": datetime.now().isoformat(),
                "image_count": len(images_data)
            }
        
    
    def _detect_mime_type(self, image_data: bytes, filename: str) -> str:

        ext = Path(filename).suffix.lower()
        mime_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif',
            '.bmp': 'image/bmp', '.webp': 'image/webp'
        }
        
        if ext in mime_map:
            return mime_map[ext]
        
   
        if image_data.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg'
        elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'image/png'
        elif image_data.startswith(b'GIF'):
            return 'image/gif'
        
        return 'image/jpeg'  

