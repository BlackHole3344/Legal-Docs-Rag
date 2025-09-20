from qdrant_client import QdrantClient 
from contextlib import contextmanager 
import threading 
from queue import Queue  
import os 
from dotenv import load_dotenv 
from dataclasses import dataclass 


load_dotenv() 
QDRANT_URL=os.environ["QDRANT_URL"]
QDRANT_API_KEY=os.environ["QDRANT_API_KEY"]



@dataclass
class QdrantConfig:

    url : str = QDRANT_URL
    api_key:str = QDRANT_API_KEY
    timeout: int = 30
    pool_size: int = 5
    retry_count: int = 3
    batch_size: int = 100
    vector_size: int = 768
    distance_metric: str = "COSINE" 
    

class QdrantClientPool:
    def __init__(self):
        self.config = QdrantConfig
        self.pool = Queue(maxsize=self.config.pool_size)
        self.lock = threading.Lock()
        self._initialize_pool()
        

    
    def _initialize_pool(self):
        for _ in range(self.config.pool_size):
            client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            self.pool.put(client)
            
            
    @contextmanager 
    def get_client(self):
        client = self.pool.get() 
        try : 
            yield client 
        finally : 
            self.pool.put(client) 
    