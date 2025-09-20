import logging
from typing import List, Dict, Any , Tuple 
from google import genai
from dotenv import load_dotenv
from google.genai.types import HttpOptions
from google.auth import load_credentials_from_file
from google.genai.types import HttpOptions, Content, Part, GenerateContentConfig, ThinkingConfig
from google.auth import default
from ..gemini_client import GeminiClient 
import os
from dotenv import load_dotenv
import json 

print(os.path.abspath("key.json"))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("key.json")

load_dotenv()


logger = logging.getLogger(__name__)

class GeminiQADataAugmentor:
    """
    Augmentor for creating comprehensive QA responses using extracted Qdrant data with Gemini
    """
    
    def __init__(self):
        self._gemini_client = GeminiClient()
        logger.info("GeminiQADataAugmentor initialized")
    
    # def extract_augment_data(self, points: List[Any]) -> Dict[str, List[Any]]:
    #     """Extract and augment data from Qdrant points"""
    #     logger.info(f"Starting data extraction from {len(points)} points")
        
    #     raw_texts = []
    #     images = []
    #     tables = []
        
    #     for i, point in enumerate(points):
    #         try:
    #             payload = point.payload
    #             types = payload.get('types', [])
                
    #             logger.debug(f"Processing point {i+1}/{len(points)} with types: {types}")
                
    #             # Parse original content
    #             try:
    #                 original_content = json.loads(payload.get('original_content', '{}'))
    #             except json.JSONDecodeError as e:
    #                 logger.warning(f"Failed to parse original_content for point {i+1}: {e}")
    #                 continue
                
    #             # Extract raw text
    #             if 'raw_text' in original_content:
    #                 raw_text = original_content['raw_text']
    #                 if raw_text and raw_text.strip():
    #                     raw_texts.append(raw_text)
    #                     logger.debug(f"Extracted text from point {i+1} ({len(raw_text)} chars)")
                
    #             # Extract images if image type is present
    #             if 'image' in [t.lower() for t in types] and 'images_base64' in original_content:
    #                 point_images = original_content['images_base64']
    #                 if point_images:
    #                     images.extend(point_images)
    #                     logger.debug(f"Extracted {len(point_images)} images from point {i+1}")
                
    #             # Extract tables if table type is present
    #             if 'table' in [t.lower() for t in types] and 'tables_html' in original_content:
    #                 point_tables = original_content['tables_html']
    #                 if point_tables:
    #                     tables.extend(point_tables)
    #                     logger.debug(f"Extracted {len(point_tables)} tables from point {i+1}")
                        
    #         except Exception as e:
    #             logger.error(f"Error processing point {i+1}: {e}")
    #             continue
        
    #     result = {
    #         'raw_texts': raw_texts,
    #         'images': images,
    #         'tables': tables
    #     }
        
    #     logger.info(f"Data extraction completed: {len(raw_texts)} texts, {len(images)} images, {len(tables)} tables")
    #     return result
    
    
    def extract_augment_data(self, points: List[Any]) -> Dict[str, Any]:

        import json
        logger.info(f"Starting data extraction from {len(points)} points")
        
        raw_texts = []
        image_mappings = {}
        table_mappings = {}
        
        for i, point in enumerate(points):
            try:
                payload = point.payload
                types = payload.get('types', [])
                logger.info(f"Processing point {i+1}/{len(points)} with types: {types}")
                
    
                try:
                    original_content = json.loads(payload.get('original_content', '{}'))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse original_content for point {i+1}: {e}")
                    continue
                
         
                if 'raw_text' in original_content:
                    raw_text = original_content['raw_text']
                    if raw_text and raw_text.strip():
                        raw_texts.append(raw_text)
                        logger.info(f"Extracted text from point {i+1} ({len(raw_text)} chars)")
                
   
                if 'image' in [t.lower() for t in types] and 'images_base64' in original_content:
                    point_images = original_content['images_base64']
                    if point_images:
                        for img in point_images:
                            image_id = f"img_{len(image_mappings) + 1:03d}"
                            image_mappings[image_id] = img
                        logger.info(f"Extracted {len(point_images)} images from point {i+1}")
                
         
                if 'table' in [t.lower() for t in types] and 'tables_html' in original_content:
                    point_tables = original_content['tables_html']
                    if point_tables:
                        for table in point_tables:
                            table_id = f"table_{len(table_mappings) + 1:03d}"
                            table_mappings[table_id] = table
                        logger.info(f"Extracted {len(point_tables)} tables from point {i+1}")
                        
            except Exception as e:
                logger.error(f"Error processing point {i+1}: {e}")
                continue
        
   
        self.image_id_mapping = image_mappings
        self.table_id_mapping = table_mappings
        
        result = {
            'raw_texts': raw_texts,
            'image_mappings': image_mappings,
            'table_mappings': table_mappings
        }
        
        print(f"result are : {result}") 
        
        logger.info(f"Data extraction completed: {len(raw_texts)} texts, {len(image_mappings)} images, {len(table_mappings)} tables")
        return result






    
#   
# 
# tables that were actually referenced

# Usage for Frontend Integration

    def _create_retrieval_system_prompt(self, user_query: str, augmented_data: Dict[str, Any]) -> str:
    
        logger.info("Creating retrieval system prompt")
        try : 
            prompt = f"""You are a Legal Document Demystification Assistant. Your mission is to make complex legal language accessible and understandable to everyone, regardless of their legal background or language proficiency.

        USER QUESTION: {user_query}

        CORE PRINCIPLES:
        - Transform complex legal jargon into clear, everyday language
        - Respond in the user's language or requested language
        - Explain rights, obligations, consequences, and practical implications
        - Use simple, everyday words instead of legal jargon
        - Provide practical implications and real-life context
        - Clearly identify what the person CAN do, MUST do, and CANNOT do
        - Highlight potential penalties, deadlines, or negative outcomes
        - Define important legal terms in simple language

        CITATION REQUIREMENTS:
        When referencing any tables or images in your response, you MUST cite them using these formats:
        - For tables: [TABLE:table_001], [TABLE:table_002], etc.
        - For images: [IMAGE:img_001], [IMAGE:img_002], etc.
        Only cite media that directly supports your answer or provides essential context.

        CONTENT SOURCES:
        """
            
            # Add text content
            
            logger.info(f"Adding content into prompt....(texts , tables , images)")
            if augmented_data.get('raw_texts'):
                prompt += "\n=== DOCUMENT TEXT CONTENT ===\n"
                for i, text_item in enumerate(augmented_data['raw_texts'], 1):
                    prompt += f"\nSource {i}:\n{text_item}\n"
                logger.info(f"Added {len(augmented_data['raw_texts'])} text sources to prompt")
            
            # Add table content with unique IDs
            if augmented_data.get('table_mappings'):
                prompt += "\n=== LEGAL TABLES AND STRUCTURED DATA ===\n"
                prompt += "IMPORTANT: These tables contain critical legal terms, rates, penalties, or procedures. Reference them using [TABLE:table_ID] format when citing.\n\n"
                
                for i, (table_id  , table )  in enumerate(augmented_data['table_mappings'].items(), 1):
                    prompt += f"Table ID: {table_id}\n{table}\n\n"
                logger.info(f"Added {len(augmented_data['table_mappings'].items())} tables to prompt")
            
            # Add image instructions with unique IDs
            if augmented_data.get('image_mappings'):
                prompt += f"\n=== LEGAL DOCUMENT IMAGES ===\n"
                prompt += f"Number of images provided: {len(augmented_data['image_mappings'].items())}\n"
                prompt += "Image IDs: " + ", ".join(augmented_data['image_mappings'].keys())+ "\n"
                prompt += "ANALYZE FOR: Signatures, seals, legal stamps, diagrams, flowcharts, legal forms, or visual elements affecting legal interpretation.\n"
                prompt += "CITATION: Reference specific images using [IMAGE:img_ID] format when they support your answer.\n"
                logger.info(f"Added instructions for {len(augmented_data['image_mappings'].items())} images")
            
            prompt += """

        RESPONSE FORMAT:

        ## Quick Answer
        [Direct answer to the user's question in simple terms]

        ## Document Overview
        **Document Type**: [What kind of legal document this is]
        **Main Purpose**: [Why this document exists and what it accomplishes]

        ## Key Parties and Responsibilities
        **Your Role**: [User's role, rights, and responsibilities]
        **Other Party**: [Counterparty's role and obligations]

        ## Financial Implications
        **Costs and Fees**: [Any payments, premiums, or deposits required]
        **Penalties**: [Fines or charges for violations]
        **Benefits**: [Money, refunds, or financial protections available]

        ## Critical Requirements and Deadlines
        **Mandatory Actions**: [Essential obligations and requirements]
        **Prohibited Actions**: [What cannot be done]
        **Important Deadlines**: [Time-sensitive requirements]

        ## Rights and Protections
        [Legal rights, protections, and entitlements]

        ## Limitations and Exclusions
        [What is not covered, exceptions, and restrictions]

        ## Referenced Content Summary
        **Tables Used**: [List table IDs and brief explanation of why each was referenced]
        **Images Used**: [List image IDs and brief explanation of their relevance]

        IMPORTANT: Only include the "Referenced Content Summary" section if you actually cited tables or images in your response.
        """
            
            logger.debug(f"System prompt created ({len(prompt)} characters)")
            return prompt
        except Exception as e : 
            logger.error(f"Error Somewhere creating System Prompt : {e}")
            raise e 
    def _create_image_analysis_prompt(self, num_images: int) -> str:
        """Create image analysis prompt with citation instructions"""
        logger.debug(f"Creating image analysis prompt for {num_images} images")
        return f"""
    LEGAL DOCUMENT IMAGE ANALYSIS FOR {num_images} IMAGES:

    Image IDs available for citation: {', '.join([f'img_{i:03d}' for i in range(1, num_images + 1)])}

    ANALYSIS FOCUS:
    - Legal signatures, stamps, or official seals
    - Contract terms, clauses, or legal text within images
    - Charts, tables, or diagrams with legal rates, fees, or penalties
    - Legal forms, checkboxes, or required fields
    - Any visual elements that affect legal interpretation

    CITATION REMINDER: Reference specific images using [IMAGE:img_ID] format only when they directly support your answer.
    """

    async def generate_answer(self, user_query: str, points: List[Any]) -> Tuple[str , Dict[str , Any]]:
        """Generate comprehensive answer using Gemini with citation tracking"""
        logger.info(f"Starting answer generation for query: '{user_query[:50]}...'")
        
        try:
            # Step 1: Extract and augment data
            logger.info("Step 1: Extracting and augmenting data")
            augmented_data = self.extract_augment_data(points)
            
            # Step 2: Create system prompt
            
            print(f"Augumented data Tables : {augmented_data.get('tables_mappings')}")
            
            logger.info("Step 2: Creating system prompt")
            system_prompt = self._create_retrieval_system_prompt(user_query, augmented_data)
            
            # Step 3: Prepare message parts
            logger.info("Step 3: Preparing message parts")
            message_parts = [Part.from_text(text=system_prompt)]
            
            # Add images if present
            if augmented_data.get('image_mappings'):
                logger.info(f"Adding {len(augmented_data['image_mappings'].items())} images to analysis")
                image_prompt = self._create_image_analysis_prompt(len(augmented_data['image_mappings'].items()))
                message_parts.append(Part.from_text(text=image_prompt))
                
                for i, (image_id , image) in enumerate(augmented_data['image_mappings'].items()):
                    try:
                        image_data = image 
                        if image_data.startswith('data:'):
                            image_data = image_data.split(',')[1]
                        message_parts.append(
                            Part.from_bytes(
                                data=image_data,
                                mime_type='image/jpeg'
                            )
                        )
                        logger.debug(f"Added image {i+1}/{len(augmented_data['image_mappings'].items())} to analysis")
                    except Exception as e:
                        logger.warning(f"Failed to process image {i+1}: {e}")
            
            # Step 4: Create content for Gemini
            logger.info("Step 4: Creating Gemini content")
            contents = [
                Content(
                    role="user",
                    parts=message_parts
                )
            ]
            
            # Step 5: Generate response
            logger.info("Step 5: Generating response with Gemini")
            response = await self._gemini_client.generate_response(
                contents=contents,
                max_output_tokens=4096 , 
                temperature=0.1 
            )
            
            if response :
                enhanced_answer = response
                logger.info(f"Successfully generated answer ({len(enhanced_answer)} characters)")
                
                # Extract cited images and tables for filtering
                cited_content = self._extract_cited_content(enhanced_answer)
                
                # Store cited content for frontend filtering
                cited_images = cited_content.get('images', [])
                cited_tables = cited_content.get('tables', [])
                
                logger.info(f"Cited images: {cited_images}")
                logger.info(f"Cited tables: {cited_tables}")
                
                return enhanced_answer , {"cited_images" : cited_images ,
                                          "cited_tables" : cited_tables}
            else:
                logger.error("Invalid or empty response from Gemini API")
                raise ValueError("Invalid or empty response from Gemini API")
                
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise e

    def _extract_cited_content(self, response_text: str) -> Dict[str, Dict[str, Any]]:
        """Extract cited image and table content with IDs for frontend hyperlinks"""
        import re
        
        cited_content = {'images': {}, 'tables': {}}
        
        # Extract image citations and get actual content with IDs
        image_pattern = r'\[IMAGE:([^\]]+)\]'
        image_matches = re.findall(image_pattern, response_text)
        cited_image_ids = list(set(image_matches))  # Remove duplicates
        
        # Get actual image content using mapping
        for image_id in cited_image_ids:
            if image_id in self.image_id_mapping:
                cited_content['images'][image_id] = self.image_id_mapping[image_id]
                logger.debug(f"Found cited image: {image_id}")
            else:
                logger.warning(f"Cited image ID {image_id} not found in mappings")
        
        # Extract table citations and get actual content with IDs
        table_pattern = r'\[TABLE:([^\]]+)\]'
        table_matches = re.findall(table_pattern, response_text)
        cited_table_ids = list(set(table_matches))  # Remove duplicates
        
        # Get actual table content using mapping
        for table_id in cited_table_ids:
            if table_id in self.table_id_mapping:
                cited_content['tables'][table_id] = self.table_id_mapping[table_id]
                logger.debug(f"Found cited table: {table_id}")
            else:
                logger.warning(f"Cited table ID {table_id} not found in mappings")
        
        logger.debug(f"Extracted citations - {len(cited_content['images'])} images, {len(cited_content['tables'])} tables")
        return cited_content

    async def generate_image_chat_response(self, common_id: str, user_query: str, cache_client) -> str:

        logger.info(f"Generating image chat response for common_id: {common_id}")
        
        try:
     
            logger.info("Step 1: Retrieving cached images")
            cached_data = cache_client.get(common_id)
            
            if not cached_data:
                logger.error(f"No cached data found for common_id: {common_id}")
                return "I couldn't find the images you're referring to. Please upload them again."
            
            images_base64 = cached_data.get('images_base64', [])
            filenames = cached_data.get('filenames', [])
            previous_summary = cached_data.get('summary', '')
            
            if not images_base64:
                logger.error(f"No images found in cached data for common_id: {common_id}")
                return "No images found in the cached data."
            
            logger.info(f"Retrieved {len(images_base64)} images from cache")
            
            # Step 2: Create system prompt for image chat
            logger.info("Step 2: Creating image chat prompt")
            
            image_prompt = f"""You are an AI assistant helping users analyze and discuss images they've uploaded. 

    CONTEXT:
    - User has uploaded {len(images_base64)} image(s): {', '.join(filenames)}
    - Previous analysis summary: {previous_summary}

    USER QUERY: {user_query}

    Please provide a helpful response based on the images and the user's question. You can:
    - Answer questions about specific details in the images
    - Compare information across multiple images if applicable  
    - Provide additional analysis or insights
    - Reference tables, charts, or text content as "Image 1 Table 1", "Image 2 Chart 1", etc.

    Be conversational and helpful while being accurate about what you can see in the images."""
            
            # Step 3: Build message parts
            logger.info("Step 3: Building message parts")
            message_parts = [Part.from_text(text=image_prompt)]
            
            # Add images to message parts
            for i, image_base64 in enumerate(images_base64):
                try:
                    image_data = image_base64
                    
                    # Handle data URL format if present
                    if image_data.startswith('data:'):
                        image_data = image_data.split(',')[1]
                    
                    # Decode base64 to bytes
                    import base64
                    image_bytes = base64.b64decode(image_data)
                    
                    message_parts.append(
                        Part.from_bytes(
                            data=image_bytes,
                            mime_type='image/jpeg'
                        )
                    )
                    logger.debug(f"Added image {i+1}/{len(images_base64)} to analysis")
                    
                except Exception as e:
                    logger.warning(f"Failed to process image {i+1}: {e}")
                    continue
            
            # Step 4: Create content for Gemini
            logger.info("Step 4: Creating Gemini content")
            contents = [
                Content(
                    role="user",
                    parts=message_parts
                )
            ]
            
            # Step 5: Generate response
            logger.info("Step 5: Generating response with Gemini")
            response = await self._gemini_client.generate_response(
                contents=contents,
                temperature=0.2,
                max_output_tokens=4096
            )
            
            
            return response 
            

        except Exception as e:
            logger.error(f"Error generating image chat response: {e}")
            return f"I encountered an error while processing your request: {str(e)}"

        

