from .uio_processor import UIOPROCESSOR  
from .embedding_service import EmbeddingClient 
from .vector_database_service import QdrantServiceAdapter  
from langchain_core.documents import Document 
from typing import List , Optional , Dict , Any 
from .image_processor import ImageProcessor  
import logging 

logging.basicConfig(level=logging.INFO) 
logging = logging.getLogger(__name__)

#### Document Processing Orchestrator 
class DocumentProcessor:
    def __init__(self, mode: str = "pdf"):
        self.mode = mode
        self.uio_processor = UIOPROCESSOR()
        self.embedding_client = EmbeddingClient()
        self.vector_db_client = QdrantServiceAdapter() 
        self.image_processor = ImageProcessor( )
    
    async def process_pdfs(self, files: List[bytes], filenames: List[str]) -> Dict[str, Any]:
        try:
            logging.info(f"Processing PDFs: {len(filenames)}")
            
            # Step 1: Extract documents using UIO processor
            
            documents = [] 
            multiple_docs = False 
            
            if files and len(files) == 1 :
                logging.info(f"calling UIO PROCESSOR on  {filenames[0]}")
                documents: List[Document] = await self.uio_processor.process_pdf(
                    file=files[0], 
                    filename=filenames[0]
                )
                logging.info(f"Extracted {len(documents)} documents from {filenames[0]}")
                
                if not documents or len(documents) == 0:
                    logging.warning(f"No documents were extracted from {filenames[0]}. Halting processing for this file.")
                    # Return a specific response indicating failure
                    return {
                        "status": "failure",
                        "message": "No documents or content could be extracted from the file.",
                        "filename": filenames[0],
                        "document_count": 0
                    } 
            else : 
                logging.info(f"Starting multi-PDF processing for {len(files)} files")
                # Process all files
                documents = await self.uio_processor.process_multiple_pdfs(files=files , filenames=filenames)
                multiple_docs = True   
                
                if not documents or len(documents) == 0:
                    logging.warning(f"No documents were extracted from {filenames[0]}. Halting processing for this file.")
                    # Return a specific response indicating failure
                    return {
                        "status": "failure",
                        "message": "No documents or content could be extracted from the file.",
                        "filename": filenames[0],
                        "document_count": 0
                    }       
                    
                
                
            
            # Step 2: Create collection (with proper error handling)
            try:
                if not multiple_docs : 
                     collection_name = self.vector_db_client.create_collection(filename=filenames[0])
                     logging.info(f"Created collection: {collection_name}")
                
                
                combined_collection_name = self.vector_db_client.generate_collection_name_simple(file_count=len(filenames)  , include_uuid=True , include_timestamp=True)      
                collection_name = self.vector_db_client.create_collection(collection_name=combined_collection_name)
                logging.info(f"Created collection: {collection_name} for {len(filenames)} files")
            except Exception as e:
                logging.error(f"Failed to create collection: {e}")
                raise
            
            # Step 3: Embed and upload documents
            async with self.embedding_client as client:
                try:
                    upload_stats = await self.vector_db_client.embed_and_upload_documents(
                        collection_name=collection_name,
                        documents=documents,
                        embedding_client=client,
                    )
                    
                    logging.info(f"Upload completed: {upload_stats['success']}")
                    
                    # Enhanced response with collection info
                    response = {
                        **upload_stats,
                        "collection_name": collection_name,
                        "filenames": filenames,
                        "document_count": len(documents)
                    }
                    
                    return response
                    
                except Exception as e:
                    logging.error(f"Failed to embed and upload: {e}")
                    raise
                    
        except Exception as e:
            logging.error(f"Error processing PDFs {filenames}: {e}")
            raise

 
            
            
                    
    async def process_image(self, files_data: List[bytes], filenames: List[str]) -> Dict[str, Any]:

        try:
            logging.info(f"Processing {len(files_data)} image(s): {filenames}")
            
       
            result = await self.image_processor.process_image(files_data, filenames)
            
            if result["status"] == "success":
                logging.info(f"Successfully processed {len(files_data)} image(s)")
                return {
                    "status": "success",
                    "filenames": result["filenames"],
                    "summary": result["summary"],
                    "processed_at": result["processed_at"],
                    "source": result["source"],
                    "common_id": result["common_id"],
                    "document_count": result["image_count"]
                }
            else:
                logging.error(f"Failed to process images: {filenames}")
                return {
                    "status": "failure", 
                    "message": result.get("error", "Unknown error occurred"),
                    "filenames": filenames,
                    "document_count": 0
                }
                
        except Exception as e:
            logging.error(f"Error processing images {filenames}: {e}")
            raise



                    
                
            
            
            
            
            