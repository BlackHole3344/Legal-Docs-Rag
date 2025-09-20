import unstructured_client 
from unstructured_client.models import operations , shared 
# from unstructured_client.staging.base import elements_from_base64_gzipped_json
# from unstructured_client.general.
from dotenv import load_dotenv 
import os 
import logging 
from typing import Dict , List , Any  , Optional 
# from google import genai
# from google.genai.types import HttpOptions
# from google.auth import load_credentials_from_file 
from google.genai.types import  Content, Part
# from google.auth import default 
from langchain_core.documents import Document 
from .gemini_client import GeminiClient 

import json 
from pathlib import Path 
import base64
import zlib


logging.basicConfig(level =logging.INFO ) 
 
logger = logging.getLogger(__name__) 

# print(os.path.abspath("key.json"))

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("key.json")
# KEY_PATH = os.path.abspath("key.json")  
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH 
load_dotenv()
UNSTRUCTURED_API_KEY = os.environ["UNSTRUCTURED_API_KEY"]
# _ , PROJECT_ID = default() 
LOCATION = "us-central1" 

def elements_from_base64_gzipped_json(b64_encoded_elements: str):
    """Manually decode base64 gzipped JSON elements"""
    # Base64 decode
    decoded_bytes = base64.b64decode(b64_encoded_elements)
    # Decompress gzip
    decompressed_bytes = zlib.decompress(decoded_bytes)
    # Decode to string and parse JSON
    elements_json = decompressed_bytes.decode('utf-8')
    element_dicts = json.loads(elements_json)
    return element_dicts



class UIOPROCESSOR :
    def __init__(self ) : 
        self.uio_client = unstructured_client.UnstructuredClient(
          api_key_auth=UNSTRUCTURED_API_KEY
         )
        self._gemini_client = GeminiClient() 
        self.chunk_size = 2000
        
        
    def get_request(self, file : bytes , filename : str ,  **kwargs):
  

    
        defaults = {
            "strategy": "hi_res",
            "chunking_strategy": "by_title",
            "max_characters": 3000, ### for testing 
            "new_after_n_chars": 2400,
            "unique_element_ids": True,
            "extract_image_block_types": ["Image", "Table"],
            "pdf_infer_table_structure": True,
            "include_page_breaks": False,
            "split_pdf_page": True,
            "split_pdf_concurrency_level": 15,
            "include_orig_elements": True
        }
        

        params = {**defaults, **kwargs}
        
        request = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=file,
                    file_name=filename,
                ),
                **params
            )
        )
        
        return request 
    

 
     
    async def process_pdf(self, file : bytes, filename : str ,  **kwargs) -> List[Document]:
 
        # if not os.path.exists(pdf_path):
        #     logger.error(f"PDF file not found: {pdf_path}")
        #     raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Validate file is actually a PDF
        # if not pdf_path.lower().endswith('.pdf'):
        #     logger.error(f"File is not a PDF: {pdf_path}")
        #     raise ValueError(f"File must be a PDF: {pdf_path}")
            
        try:
            logger.info(f"Starting PDF processing: {filename}")
            
            langchain_documents = []
            
            # Get request with error handling
            try:
                request = self.get_request(file=file,filename=filename, **kwargs)
                logger.info("Got Partition Request")
            except Exception as e:
                logger.error(f"Failed to create request for {filename}: {e}")
                raise Exception(f"Failed to create processing request: {e}")
            
            # Process document with unstructured API
            try:
                response = await self.uio_client.general.partition_async(request=request)
                
                if not response or not hasattr(response, 'elements'):
                    logger.error(f"Invalid response from unstructured API for {filename}")
                    return []
                    
                chunks : List[Dict[str , Any]] = response.elements
                
                logger.info(f"Pulled out {len(chunks)} from elements")
                
                if not chunks:
                    logger.warning(f"No content extracted from PDF: {filename}")
                    return []
                    
            except Exception as e:
                logger.error(f"Failed to process PDF with unstructured API: {e}")
                raise Exception(f"PDF processing failed: {e}")
            
            # Process each chunk
            processed_count = 0
            failed_count = 0
            logger.info(f"Looping Through {len(chunks)} chunks.....")
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
                    
                     
                    chunk_data = self.process_document_with_chunking_and_atomic_access(chunk)
                    
                    
                    if not chunk_data:
                        logger.warning(f"Chunk {chunk_idx + 1} returned no data, skipping")
                        failed_count += 1
                        
                        continue
                    
        
                    if not isinstance(chunk_data, dict) or 'text' not in chunk_data:
                        logger.warning(f"Invalid chunk data structure for chunk {chunk_idx + 1}, skipping")
                        failed_count += 1
                        continue
                    

                    logger.info(f"Types Found: {chunk_data.get('types', [])}")
                    logger.info(f"Tables: {len(chunk_data.get('tables', []))}, Images: {len(chunk_data.get('images', []))}")
                    
              
                    enhanced_text = chunk_data.get('text', '')
                    
                    if chunk_data.get("tables") or chunk_data.get("images"):
                        try:
                            logger.info(f"Generating enhanced text for chunk {chunk_idx + 1}")
                            enhanced_text = await self.create_enhanced_text(
                                text=chunk_data["text"],
                                tables=chunk_data["tables"],
                                images=chunk_data["images"]
                            )
                            logger.info(f"Enhanced text generated successfully for chunk {chunk_idx + 1}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to generate enhanced text for chunk {chunk_idx + 1}: {e}")
                            enhanced_text = chunk_data["text"]
                    
 
                    if not enhanced_text or not enhanced_text.strip():
                        logger.warning(f"Empty enhanced text for chunk {chunk_idx + 1}, using original text")
                        enhanced_text = chunk_data.get("text", "No content available")
            
                    metadata = {
                            "chunk_id": chunk_data.get("chunk_id"),
                            "page_number": chunk_data.get("page_number"),
                            "languages": chunk_data.get("languages", []),
                            "types": chunk_data.get("types", ["text"]),
                           }

                    
                    try:
                        metadata["original_content"] = json.dumps({
                            "raw_text": chunk_data["text"],
                            "images_base64": [image.get("image_base64") for image in chunk_data["images"]],
                            "tables_html": [table.get("table_html") for table in chunk_data["tables"]] 
                        })
                        
            
                        content_size = len(metadata["original_content"].encode('utf-8')) / (1024 * 1024)
                        if content_size > 5:  # Log if larger than 5MB
                            logger.info(f"Large metadata for chunk {chunk_idx + 1}: {content_size:.2f}MB")
                        
                    except Exception as e:
                        logger.error(f"Failed to serialize original content for chunk {chunk_idx + 1}: {e}")
                        # Even in error case, try to save what we can
                        metadata["original_content"] = json.dumps({
                            "raw_text": chunk_data.get("text", ""),
                            "error": f"Serialization failed: {str(e)}"
                        })

                    doc = Document(
                        page_content=enhanced_text,
                        metadata=metadata
                    )

                    langchain_documents.append(doc)
                    processed_count += 1
                
                except Exception as e : 
                    print(f"Error in Processing Chunk : {chunk_idx}") 
                    continue    
        
            return langchain_documents       
        except FileNotFoundError:
            # Re-raise file not found errors
            raise
        except ValueError:
            # Re-raise validation errors  
            raise
        except Exception as e:
            logger.error(f"Critical error processing PDF {filename}: {e}")
            raise Exception(f"Failed to process PDF {filename}: {e}")
                
                     
    async def process_multiple_pdfs(self, files : List[bytes], filenames : List[str] ,  **kwargs) -> List[Document]:
        
        logger.info(f"Processing files : {len(files)}")
        

        try :
            combined_documents = []        
            for file , filename in zip(files , filenames) : 
                logger.info(f"Processing file : {len(filename)}")
                docs = await self.process_pdf(file , filename) 
                logger.info(f"Processed file : {len(filename)}")
                combined_documents.extend(docs)
            return combined_documents    
        except Exception as e : 
            logger.error(f"error in processing files : {e}") 
            raise Exception(f"Failed to process PDFs : {e}")
               
             
                        
            
    def process_document_with_chunking_and_atomic_access(self , chunk : Dict[str , Any]) -> Optional[Dict[str , Any]]:

        try:
            logger.info(f"Starting Atomic processing of chunk")
            
            # print("Raw chunk : " , chunk)
            chunk_metadata = chunk.get('metadata', {})
            chunk_data = {
                'chunk_id': chunk.get('element_id'),
                'text': chunk.get('text', ''),
                "page_number": chunk_metadata.get("page_number"),
                "languages": chunk_metadata.get("languages", []),
                'images': [],
                'tables': [],
                'types': ["text"]
            }
            
            
            # Access original atomic elements from this chunk
            if 'orig_elements' in chunk_metadata:
                try:
                    orig_elements_compressed = chunk_metadata['orig_elements']
                    logger.info(f"Found orig_elements in chunk ")
                    
                    # Decompress the original elements
                    atomic_elements_objects = elements_from_base64_gzipped_json(orig_elements_compressed)
                    logger.info(f"Decompressed {len(atomic_elements_objects)} atomic elements")
                    
                    # atomic_elements_dicts: List[Dict] = []
                
                    # ###convert element objects to dicts 
                    # for elem_obj in atomic_elements_objects:
                    #     try:
                    #         atomic_elements_dicts.append(elem_obj.to_dict())
                    #     except Exception as e:
                    #         logger.error(f"Failed to convert element to dict: {e}")
                    #         continue
                    
                    # separating different types of elements
                    for element_idx, element in enumerate(atomic_elements_objects):
                        try:
                            element_type = element.get("type", "Unknown")
                            logger.info(f"Processing atomic element {element_idx + 1}: {element_type}")
                            
                            if element_type == 'Image':
                                try:
                                    element_metadata = element.get("metadata", {})
                                    image_base64 = element_metadata.get("image_base64")
                                    
                                    if image_base64:
                                        image_data = {
                                            "page_number": element_metadata.get("page_number"),
                                            "image_base64": image_base64,
                                            "image_mime_type": element_metadata.get("image_mime_type", "image/jpeg")
                                        }
                                        
                                        logger.info(f"Generated image_data")
                                         
                                        if "image" not in chunk_data['types']:
                                            chunk_data['types'].append("image")
                                        chunk_data["images"].append(image_data)
                                        logger.info(f"Added image from page {image_data['page_number']}")
                                    else:
                                        logger.info("Image element found but no image_base64 data")
                                except Exception as e:
                                    logger.error(f"Error processing image element: {e}")
                                    continue
                            
                            elif element_type == 'Table':
                                try:
                                    element_metadata = element.get("metadata", {})
                                    table_data = {}
                                    
                                    # Get HTML representation if available
                                    table_html = element_metadata.get("text_as_html")
                                    if table_html:
                                        table_data['table_html'] = table_html
                                        table_data["page_number"] = element_metadata.get("page_number")
                                        
                                        logger.info("Generated table_data")
                                        if "table" not in chunk_data['types']:
                                            chunk_data["types"].append("table")
                                        chunk_data['tables'].append(table_data)
                                        logger.info(f"Added table from page {table_data['page_number']}")
                                    else:
                                        logger.error("Table element found but no HTML data")
                                except Exception as e:
                                    logger.error(f"Error processing table element: {e}")
                                    continue
                                    
                        except Exception as e:
                            logger.error(f"Error processing atomic element {element_idx + 1}: {e}")
                            continue
                    return chunk_data 
                        
                except Exception as e:
                    logger.error(f"Error decompressing orig_elements for chunk")
            
            else:
                logger.debug(f"No orig_elements found in chunk ")
                
                return chunk_data 
            
        
        except Exception as e:
            logger.error(f"Critical error in document processing: {e}")
            return None                 
    
    
    async def create_enhanced_text(self,  text: str, tables: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> str:

        try:
    

            prompt = f"""You are creating a searchable description for document content retrieval.
    CONTENT TO ANALYZE:
    TEXT
    {text}

    """
            if tables:
                prompt += "TABLES:\n"
                for i, table in enumerate(tables):
                    table_html = table.get('table_html', '')
                    page_num = table.get('page_number', 'unknown')
                    prompt += f"Table {i+1} (Page {page_num}):\n{table_html}\n\n"
            
        
            prompt += """TASK: Generate a comprehensive, searchable summary that includes:
    1. Main topics and key information from the text
    2. Important data, numbers, and facts from tables (if present)
    3. Description of visual content from images (if present)
    4. Searchable keywords and phrases users might query
    5. Document type and purpose

    Keep the summary concise but comprehensive for retrieval purposes."""

            # Create message content starting with text
            message_parts = [Part.from_text(text=prompt)]
            
            # Add images if present
            if images:
                prompt_with_images = prompt + f"\n\nIMAGES: {len(images)} image(s) to analyze for visual content, charts, diagrams, or important visual information."
                message_parts = [Part.from_text(text=prompt_with_images)]
                for i, image in enumerate(images):
                    if 'image_base64' in image:
                        message_parts.append(
                            Part.from_bytes(
                                data=image['image_base64'],
                                mime_type=image.get('image_mime_type', 'image/jpeg')
                            )
                        )
                        logger.debug(f"Added image {i+1} to analysis")
            
            # Create content for Gemini
            contents = [
                Content(
                    role="user",
                    parts=message_parts
                )
            ]
            
            # Generate enhanced content using Gemini 2.5 Flash
            enhanced_text = await self._gemini_client.generate_response(
                contents=contents , 
                temperature=0.2
            )
            
            logger.info(f"Successfully generated enhanced text ({len(enhanced_text)} characters)")
            return enhanced_text
            
        except Exception as e:
            logger.error(f"AI text generation failed: {e}")
            return self._create_fallback_summary(text, tables, images)

    def _create_fallback_summary(self , text: str, tables: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> str:

        summary_parts = []

        if text:
            summary_parts.append(f"CONTENT: {text[:300]}")

        if tables:
            summary_parts.append(f"TABLES: Document contains {len(tables)} table(s) with structured data")
            for i, table in enumerate(tables):
                if 'table_html' in table:
    
                    import re
                    table_text = re.sub(r'<[^>]+>', ' ', table['table_html'])
                    table_text = re.sub(r'\s+', ' ', table_text).strip()
                    summary_parts.append(f"Table {i+1}: {table_text[:100]}...")

        if images:
            summary_parts.append(f"IMAGES: Document contains {len(images)} image(s) for visual analysis")
        
        return "\n\n".join(summary_parts)     