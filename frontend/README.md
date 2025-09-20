# Document Chat Assistant - Streamlit Frontend

A modern Streamlit interface for your FastAPI document processing backend.

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API URLs
Open `main.py` and update line 97-99:
```python
API_BASE_URL = "http://localhost:8000"  # Change to your FastAPI URL
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload-files/"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat/"
```

### 3. Run the Application
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## Features

✅ **File Upload Interface**
- Drag-and-drop support
- Multiple file types (PDF, images)
- File previews and metadata
- Progress indicators

✅ **Chat Interface** 
- Real-time messaging
- Message history
- Professional styling
- Error handling

✅ **Citations System**
- Citation badges in responses
- Expandable citation panels
- Image display and zoom
- Source information

✅ **Professional UI**
- Modern gradient design
- Responsive layout
- Mobile-friendly
- Custom CSS styling

## FastAPI Backend Requirements

Your FastAPI backend should have these endpoints:

### 1. File Upload Endpoint
```python
@app.post("/upload-files/")
async def upload_files(files: List[UploadFile], session_id: str):
    # Process files and return success response
    return {"status": "success", "message": "Files processed"}
```

### 2. Chat Endpoint  
```python
@app.post("/chat/")
async def chat(request: ChatRequest):
    # Process chat message and return response with citations
    return {
        "message": "AI response text here",
        "citations": [
            {
                "source": "document.pdf",
                "page": 1,
                "context": "relevant text",
                "type": "image",
                "image_data": "base64_encoded_image"
            }
        ]
    }
```

## Customization

### Update Styling
Modify the CSS in `main.py` lines 20-95 to match your brand colors.

### Change API Response Format
Update the `parse_citations_from_response()` function to match your backend response format.

### Add New Features
- File management (delete uploaded files)
- Export chat history  
- User authentication
- Multiple document sessions

## Troubleshooting

**Connection Error**: Check your API_BASE_URL in main.py
**File Upload Issues**: Verify your FastAPI accepts multipart/form-data
**Citations Not Showing**: Check the response format from your chat endpoint

## Production Deployment

### Using Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables for API URLs

### Using Docker
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Support

If you encounter issues:
1. Check your FastAPI backend is running
2. Verify API endpoint URLs are correct
3. Check browser console for JavaScript errors
4. Test API endpoints directly with curl/Postman first