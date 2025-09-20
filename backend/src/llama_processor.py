# import asyncio
# from typing import Dict, Any, Optional
# import pymupdf4llm
# from llama_cloud_services import LlamaParse 
# # from docx import Document
# from PIL import Image
# import io
# import dotenv 
# import os 
# import logging 
# from .models import FileProcessorResponse 


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# dotenv.load_dotenv()    
# LLAMAPARSE_KEY = os.getenv("LLAMAPARSE_KEY")

# class DocumentProcessor:
#     def __init__(self, default_parser_mode: str = "pymupdf", auto_switch_threshold: int = 5_000_000):
 
#         self.llama_parser = None
#         self.filetype = None 
#         self.UIO_client = None 
        
#         # Validate default mode
#         if default_parser_mode not in ["llamaparse", "pymupdf"]:
#             logger.error(f"Unsupported default parser mode: {default_parser_mode}")
#             raise ValueError(f"Unsupported default parser mode: {default_parser_mode}")
        
#         # Initialize LlamaParse only if needed and available
#         if default_parser_mode == "llamaparse":
#             self._initialize_llamaparse()
        
#         logger.info(f"Initialized DocumentProcessor with default mode: {self.default_parser_mode}")
    
#     def _initialize_llamaparse(self) -> bool:
#         """Initialize LlamaParse if not already done"""
#         if self.llama_parser is not None:
#             return True
            
#         if LLAMAPARSE_KEY is None:
#             logger.warning("LLAMAPARSE_KEY environment variable is not set. LlamaParse unavailable.")
#             return False
        
#         try:
#             self.llama_parser = LlamaParse(
#                 api_key=LLAMAPARSE_KEY,
#                 verbose=True,
#             )
#             logger.info("LlamaParse initialized successfully.")
#             return True
#         except Exception as e:
#             logger.error(f"Failed to initialize LlamaParse: {str(e)}")
#             return False
    
    
#     def _initialize_uio_client(self) -> bool : 
        
#     def set_parser_mode(self, mode: str, persist: bool = True):  
#         if mode == "auto":
#             self.current_parser_mode = "auto"
#             if persist:
#                 self.default_parser_mode = "auto"
#             logger.info("Parser mode set to AUTO (will choose based on file size)")
#             return
            
#         if mode not in ["llamaparse", "pymupdf"]:
#             logger.error(f"Unsupported parser mode: {mode}")
#             raise ValueError(f"Unsupported parser mode: {mode}")
        
#         self.current_parser_mode = mode
#         if persist:
#             self.default_parser_mode = mode
            
#         # Initialize LlamaParse if switching to it
#         if mode == "llamaparse":
#             self._initialize_llamaparse()
        
#         logger.info(f"Parser mode set to: {self.current_parser_mode}" + 
#                    (" (persistent)" if persist else " (temporary)"))
    
   
    
#     async def process_document(self, content: bytes, file_type: str, filename: str) -> Dict[str, Any]:
#         """Route document to appropriate processor based on type"""
#         logger.info(f"Processing document: {filename} (type: {file_type}, size: {len(content):,} bytes)")
        
#         processors = {
#             'application/pdf': self.process_pdf,
#             # 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self.process_docx,
#             # 'application/msword': self.process_doc,
#             'text/plain': self.process_text
#         }

#         processor = processors.get(file_type)
#         if not processor:
#             logger.error(f"No processor for file type: {file_type}")
#             raise ValueError(f"No processor for file type: {file_type}")
#         self.filetype = file_type  
#         # Determine optimal parser mode for this specific fil       
#         return await processor(content, filename)

#     async def process_pdf(self, content: bytes, filename: str  ) -> FileProcessorResponse:
#         """Process PDF files with specified or optimal parser mode"""

        
#         logger.info(f"Processing PDF: {filename} (size: {len(content):,} bytes) with {self.current_parser_mode}")
        
#         try:
#             if self.current_parser_mode == "llamaparse":
#                 if self.llama_parser is None and not self._initialize_llamaparse():
#                     logger.warning("LlamaParse requested but unavailable. Falling back to PyMuPDF.")
#                     parser_mode = "pymupdf"
#                 else:
#                     logger.info("Using LlamaParse for PDF extraction.")
#                     temp_path = f"/tmp/{filename}"
#                     with open(temp_path, "wb") as f:
#                         f.write(content)
#                     documents = self.llama_parser.load_data(temp_path)
#                     text = documents[0].text if documents else ""
#                     method = "llamaparse"
            
#             if self.current_parser_mode == "pymupdf":
#                 logger.info("Using PyMuPDF for PDF extraction.")
#                 temp_path = f"/tmp/{filename}"
#                 # Temporary workaround for pymupdf4llm bug  
#                 with open(temp_path, "wb") as f:
#                     f.write(content)
#                 try:
#                     text = pymupdf4llm.to_markdown(temp_path)
#                     method = "pymupdf"
#                 finally:
#                     try:
#                         os.remove(temp_path)
#                         logger.info(f"Temporary file {temp_path} deleted.")
#                     except Exception as cleanup_err:
#                         logger.warning(f"Failed to delete temp file {temp_path}: {cleanup_err}")

#             self._write_text_to_temp(text, filename)
#             logger.info(f"PDF processed successfully with method: {method}")

#             return FileProcessorResponse(
#                 status="success",
#                 method=method,
#                 pages=self._count_pages(content), 
#                 data={"text": text},
#                 file_type=self.filetype,
#                 filename=filename
#             )
    
            
#         except Exception as e:
#             logger.exception(f"PDF processing failed: {str(e)}")
#             raise Exception(f"PDF processing failed: {str(e)}")

#     def _write_text_to_temp(self, text: str, filename: str):
#         """Write extracted text to a .txt file in a temp folder"""
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         temp_dir = os.path.join(os.path.dirname(base_dir), "temp")
#         os.makedirs(temp_dir, exist_ok=True)
#         txt_filename = os.path.splitext(filename)[0] + ".txt"
#         txt_path = os.path.join(temp_dir, txt_filename)

#         with open(txt_path, "w", encoding="utf-8") as f:
#             f.write(text)
#         logger.info(f"Extracted text written to {txt_path}")

    

#     async def process_text(self, content: bytes, filename: str, parser_mode: Optional[str] = None) -> Dict[str, Any]:
#         """Process plain text files"""
#         logger.info(f"Processing text file: {filename} (size: {len(content):,} bytes)")
#         try:
#             text = content.decode('utf-8')
#             logger.info("Text file decoded with utf-8.")
#             return {
#                 "text": text,
#                 "method": "direct_text",
#                 "parser_mode": "utf-8",
#                 "length": len(text)
#             }
#         except UnicodeDecodeError:
#             logger.warning("utf-8 decoding failed, trying fallback encodings.")
#             for encoding in ['latin-1', 'cp1252']:
#                 try:
#                     text = content.decode(encoding)
#                     logger.info(f"Text file decoded with {encoding}.")
#                     return {
#                         "text": text,
#                         "method": f"direct_text_{encoding}",
#                         "parser_mode": encoding,
#                         "length": len(text)
#                     }
#                 except Exception:
#                     logger.warning(f"Decoding with {encoding} failed.")
#                     continue
#             logger.error("Could not decode text file with any supported encoding.")
#             raise Exception("Could not decode text file")

#     def _count_pages(self, pdf_content: bytes) -> int:
#         """Count pages in PDF"""
#         try:
#             import fitz  # PyMuPDF
#             doc = fitz.open(stream=pdf_content, filetype="pdf")
#             page_count = len(doc)
#             doc.close()
#             logger.info(f"PDF page count: {page_count}")
#             return page_count
#         except Exception as e:
#             logger.warning(f"Failed to count PDF pages: {str(e)}")
#             return 0
    
#     def get_status(self) -> Dict[str, Any]:
#         """Get current processor status"""
#         return {
#             "default_parser_mode": self.default_parser_mode,
#             "current_parser_mode": self.current_parser_mode,
#             "llamaparse_available": self.llama_parser is not None,
#             "llamaparse_key_set": LLAMAPARSE_KEY is not None
#         }

