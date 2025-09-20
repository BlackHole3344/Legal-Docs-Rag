"""
Demo mode functions for testing the Streamlit interface without FastAPI backend
Set DEMO_MODE = True in config.py to use these functions
"""

import time
import random
from typing import Dict, List

def demo_upload_files(files) -> Dict:
    """Simulate file upload for demo mode"""
    # Simulate processing delay
    time.sleep(2)
    
    return {
        "status": "success", 
        "data": {
            "message": "Files processed successfully",
            "files_processed": len(files),
            "session_id": "demo_session_123"
        }
    }

def demo_chat_response(message: str) -> Dict:
    """Generate demo chat responses with citations"""
    # Simulate thinking delay
    time.sleep(1)
    
    # Sample responses based on message content
    responses = {
        "default": {
            "message": "Based on your uploaded documents, I can help you with various questions. What would you like to know?",
            "citations": []
        },
        "summary": {
            "message": "Here's a summary of your documents: The uploaded files contain information about document processing and AI chat systems. Key topics include file handling, user interface design, and API integration. [1] [2]",
            "citations": [
                {
                    "source": "document.pdf",
                    "page": 1,
                    "context": "Document processing systems enable users to upload and analyze various file formats...",
                    "type": "document"
                },
                {
                    "source": "interface.png",
                    "page": None,
                    "context": "User interface mockup showing chat interface with file upload capabilities",
                    "type": "image"
                }
            ]
        },
        "features": {
            "message": "The main features mentioned in your documents include: 1) File upload with drag-and-drop support [1], 2) Real-time chat interface [2], 3) Citation system for sources [3], and 4) Responsive design for mobile devices [1].",
            "citations": [
                {
                    "source": "requirements.pdf",
                    "page": 2,
                    "context": "The system shall support drag-and-drop file upload functionality with real-time progress indicators...",
                    "type": "document"
                },
                {
                    "source": "chat_interface.jpg",
                    "page": None,
                    "context": "Screenshot of the chat interface showing message bubbles and typing indicators",
                    "type": "image"
                },
                {
                    "source": "citations.pdf",
                    "page": 5,
                    "context": "Citation system allows users to trace AI responses back to source documents with page-level precision...",
                    "type": "document"
                }
            ]
        }
    }
    
    # Simple keyword matching for demo responses
    message_lower = message.lower()
    if any(word in message_lower for word in ["summary", "summarize", "overview"]):
        response_data = responses["summary"]
    elif any(word in message_lower for word in ["features", "functionality", "capabilities"]):
        response_data = responses["features"]
    else:
        response_data = responses["default"]
    
    # Add some randomness to make it feel more real
    if random.random() > 0.7:  # 30% chance to add extra citation
        response_data["citations"].append({
            "source": "additional_doc.pdf",
            "page": random.randint(1, 10),
            "context": "Additional context from the documents that supports this response...",
            "type": "document"
        })
    
    return {
        "status": "success",
        "data": response_data
    }

# Sample demo data for testing
DEMO_FILES = [
    {"name": "requirements.pdf", "size": 245760, "type": "application/pdf"},
    {"name": "interface_mockup.png", "size": 102400, "type": "image/png"},
    {"name": "user_guide.pdf", "size": 512000, "type": "application/pdf"}
]

DEMO_CHAT_HISTORY = [
    {
        "role": "user",
        "content": "What are the main features mentioned in the documents?",
        "timestamp": "2024-01-15 10:30:00"
    },
    {
        "role": "assistant", 
        "content": "Based on your documents, the main features include: 1) File upload with drag-and-drop [1], 2) Real-time chat interface [2], and 3) Citation system [1] [3].",
        "citations": [
            {
                "source": "requirements.pdf",
                "page": 2,
                "context": "File upload functionality with progress indicators...",
                "type": "document"
            }
        ],
        "timestamp": "2024-01-15 10:30:15"
    }
]