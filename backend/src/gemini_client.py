import os
import logging
from typing import Optional , Union , List , Dict 
from google import genai
from google.genai.types import HttpOptions, Content, Part, GenerateContentConfig, ThinkingConfig
from google.auth import default

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.gemini_client = None
    #     self._setup_credentials()
    
    # # def _setup_credentials(self):
    # #     key_path = os.path.abspath("key.json")
    # #     if os.path.exists(key_path):
    # #         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    # #         logger.info(f"Using credentials from: {key_path}")
    
    async def get_client(self) -> genai.Client:
        """Get or create Gemini client instance"""
        try:
            if self.gemini_client is None:
                credentials, project_id = default()
                
                if not project_id:
                    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                    if not project_id:
                        raise ValueError("Could not determine project ID. Set GOOGLE_CLOUD_PROJECT environment variable.")
                
                self.gemini_client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location="us-central1",
                    http_options=HttpOptions(api_version="v1")
                )
                
                logger.info(f"Gemini client initialized successfully for project: {project_id}")
            
            return self.gemini_client
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    
    
    
    
    async def generate_response(
        self,
        contents: Union[str, List[Content], List[Dict]],
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        thinking_budget: int = 0,
        top_p: float = 0.8,
        top_k: int = 40,
        **kwargs
    ) -> str:
        """Generate response from Gemini model"""
        try:
            client = await self.get_client()
            

            if isinstance(contents, str):
                contents = [Content(role="user", parts=[Part(text=contents)])]
            elif isinstance(contents, list) and contents and isinstance(contents[0], dict):

                formatted_contents = []
                for content in contents:
                    if isinstance(content, dict):
                        role = content.get("role", "user")
                        text = content.get("content", content.get("text", ""))
                        formatted_contents.append(Content(role=role, parts=[Part(text=text)]))
                contents = formatted_contents

            config = GenerateContentConfig(
                thinking_config=ThinkingConfig(thinking_budget=thinking_budget),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                top_k=top_k,
                **kwargs
            )
            
    
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            
               
            if response and hasattr(response, 'text') and response.text:
               answer = response.text.strip()
            else:
               raise ValueError("Invalid or empty response from Gemini API")
            
            logger.info(f"Generated response using model: {model}")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise

    def get_sync_client(self) -> genai.Client:
        """Get synchronous client (if you need non-async version)"""
        try:
            if self.gemini_client is None:
                credentials, project_id = default()
                
                if not project_id:
                    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                    if not project_id:
                        raise ValueError("Could not determine project ID. Set GOOGLE_CLOUD_PROJECT environment variable.")
                
                self.gemini_client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location="us-central1",
                    http_options=HttpOptions(api_version="v1")
                )
                
                logger.info(f"Gemini client initialized successfully for project: {project_id}")
            
            return self.gemini_client
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise