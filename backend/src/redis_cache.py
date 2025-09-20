import redis
import json
import base64
import logging
from typing import Optional, Dict, Any
import os 



REDIS_HOST = os.getenv("REDIS_HOST")
logger = logging.getLogger(__name__)

class RedisImageCache:    
    def __init__(self, host = None , port=6379, db=0):  # Changed default to localhost for testing
        try:
            print(REDIS_HOST)
            if host is None : 
                host = os.getenv("REDIS_HOST" , "localhost")
                
                
            self.redis_client = redis.Redis(
                host=host, 
                port=port, 
                db=db, 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis cache connected successfully to {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis at {host}:{port}: {e}")
            logger.info("Make sure Redis is running: redis-server or docker run -p 6379:6379 redis:alpine")
            raise
    
    def get(self, common_id: str) -> Optional[Dict[str, Any]]:
        """Get cached data by common_id"""
        try:
            data = self.redis_client.get(f"img:{common_id}")
            if data:
                logger.info(f"Cache hit for common_id: {common_id}")
                return json.loads(data)
            logger.info(f"Cache miss for common_id: {common_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    def set(self, common_id: str, data: Dict[str, Any], ttl_seconds: int = 604800):
        try:
            self.redis_client.setex(
                f"img:{common_id}", 
                ttl_seconds, 
                json.dumps(data)
            )
            logger.info(f"Cached data with common_id: {common_id}")
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
    
    def delete(self, common_id: str):
        try:
            self.redis_client.delete(f"img:{common_id}")
            logger.info(f"Deleted cached data: {common_id}")
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            keys = self.redis_client.keys("img:*")
            total_memory = self.redis_client.memory_usage("img:*") or 0
            
            return {
                "total_cached_entries": len(keys),
                "total_memory_bytes": total_memory,
                "total_memory_mb": round(total_memory / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    def clear_expired(self):
        try:
            keys = self.redis_client.keys("img:*")
            expired_count = 0
            for key in keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -2:  
                    expired_count += 1
            
            logger.info(f"Found {expired_count} expired keys (auto-cleaned by Redis)")
            return expired_count
        except Exception as e:
            logger.error(f"Error checking expired keys: {e}")
            return 0