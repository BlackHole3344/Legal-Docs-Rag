# Configuration settings for the Streamlit app

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your FastAPI backend URL
UPLOAD_ENDPOINT = "/upload-files/"
CHAT_ENDPOINT = "/chat/"
CITATIONS_ENDPOINT = "/citations/"

# File upload settings
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
ALLOWED_FILE_TYPES = ['pdf', 'jpg', 'jpeg', 'png', 'gif', 'webp']

# UI Configuration
APP_TITLE = "Document Chat Assistant"
APP_ICON = "📄"
PRIMARY_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"

# Chat settings
MAX_CHAT_HISTORY = 100  # Maximum number of messages to keep in history
CHAT_TIMEOUT = 60  # Timeout for chat API calls in seconds
UPLOAD_TIMEOUT = 300  # Timeout for file upload in seconds

# Demo mode (set to False for production)
DEMO_MODE = True  # If True, shows sample data instead of calling API