import streamlit as st
import time
import random
import base64
import requests
import re
from io import BytesIO
from datetime import datetime
import os 
# Page config
st.set_page_config(page_title="Legal Doc Assistant", page_icon="⚖️", layout="wide")

# API Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
UPLOAD_API_URL = f"{BACKEND_URL}/upload/pdf"  # Single file
UPLOAD_MULTIPLE_API_URL = f"{BACKEND_URL}/upload/pdfs"  # Multiple files
UPLOAD_MULTIPLE_IMAGES_API_URL = f"{BACKEND_URL}/upload/multiple/images"  # Multiple images
CHAT_API_URL = f"{BACKEND_URL}/chat"  # Base URL for chat
DEMO_MODE = False  # Set to True to use demo responses


# Demo responses with citations including images
DEMO_RESPONSES = [
    {
        "answer": "Based on the legal documents, a contract is a legally binding agreement between two or more parties that creates mutual obligations enforceable by law. The essential elements include offer, acceptance, consideration, and mutual intent to be bound.",
        "sources": [
            {
                "content": "A contract is an agreement which the law will enforce. It is a legally binding arrangement between two or more parties...",
                "metadata": {"source": "contract_law.pdf - Page 12", "type": "text"}
            },
            {
                "content": "Essential elements of contracts diagram showing the four pillars",
                "metadata": {"source": "legal_diagrams.pdf - Page 5", "type": "image", "image_desc": "Contract Elements Diagram"}
            }
        ]
    },
    {
        "answer": "According to the documents, tort law deals with civil wrongs that cause harm to individuals. It includes negligence, intentional torts, and strict liability. The primary purpose is to provide compensation to victims and deter wrongful conduct.",
        "sources": [
            {
                "content": "Tort law is a body of law that addresses and provides remedies for civil wrongs not arising out of contractual obligations...",
                "metadata": {"source": "tort_law_basics.pdf - Page 8", "type": "text"}
            },
            {
                "content": "Flowchart showing the classification of torts into three main categories",
                "metadata": {"source": "tort_classifications.pdf - Page 15", "type": "image", "image_desc": "Tort Classification Flowchart"}
            }
        ]
    }
]

def chat_with_document(query, collection_name, top_k=5):
    """Chat with document using real API"""
    if DEMO_MODE:
        # Use demo responses
        return {"success": True, "data": get_demo_response(query)}
    
    try:
        # Prepare request payload
        payload = {
            "query": query,
            "top_k": top_k
        }
        
        # Call real chat API with collection_id in path
        chat_url = f"{CHAT_API_URL}/{collection_name}"
        
        # Using GET as per your endpoint definition
        response = requests.get(chat_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False, 
                "error": f"Chat failed: {response.status_code} - {response.text}"
            }
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def chat_with_images(query, common_id, top_k=5):
    """Chat with images using common_id"""
    if DEMO_MODE:
        # Use demo responses for images
        return {"success": True, "data": get_demo_response(query)}
    
    try:
        # Prepare request payload for image chat
        payload = {
            "common_id": common_id,
            "query": query
        }
        
        # Call image chat API endpoint
        chat_url = f"{CHAT_API_URL.replace('/chat', '')}/chat_image"
        
        response = requests.get(chat_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False, 
                "error": f"Image chat failed: {response.status_code} - {response.text}"
            }
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def simulate_detailed_progress(mode="pdf"):
    """Simulate detailed processing stages for visual feedback"""
    if mode == "image":
        stages = [
            ("📤 Uploading Images", 0, 20, 2),
            ("🔍 Image Analysis", 20, 50, 4),
            ("🧠 AI Processing", 50, 80, 5),
            ("📝 Generating Summary", 80, 100, 3)
        ]
    else:  # PDF mode
        stages = [
            ("📤 Uploading Document", 0, 10, 2),
            ("📊 Document Partitioning", 10, 25, 3),
            ("✂️ Text Chunking", 25, 45, 4),
            ("🔍 Atomic Element Extraction", 45, 70, 5),
            ("🧠 Generating Embeddings", 70, 90, 4),
            ("💾 Uploading to Vector Database", 90, 100, 2)
        ]
    
    return stages

def upload_files_to_api(uploaded_files):
    """Upload PDF files to FastAPI backend"""
    try:
        # Prepare files for API - FIXED FORMAT for FastAPI multiple files
        files = []
        for uploaded_file in uploaded_files:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            files.append(
                ("files", (uploaded_file.name, uploaded_file, uploaded_file.type))
            )
            
        # Call real API
        response = requests.post(UPLOAD_MULTIPLE_API_URL, files=files, timeout=300)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False, 
                "error": f"Upload failed: {response.status_code} - {response.text}"
            }
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def upload_images_to_api(uploaded_files):
    """Upload images to FastAPI backend"""
    try:
        # Prepare files for API
        files = []
        for uploaded_file in uploaded_files:
            files.append(
                ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
            )
            
        # Call real API
        response = requests.post(UPLOAD_MULTIPLE_IMAGES_API_URL, files=files, timeout=300)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False, 
                "error": f"Upload failed: {response.status_code} - {response.text}"
            }
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def upload_file_to_api(uploaded_file):
    """Upload single PDF file to FastAPI backend"""
    if DEMO_MODE:
        # Demo response
        time.sleep(2)
        return {
            "success": True,
            "data": {
                "total_time_seconds": 12.5,
                "upload_stats": {"total_chunks": 15},
                "collection_name": f"demo_collection_{int(time.time())}",
                "filename": uploaded_file.name
            }
        }
    
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Prepare file for API - FIXED FORMAT for FastAPI
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        
        # Call real API
        response = requests.post(UPLOAD_API_URL, files=files, timeout=300)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False, 
                "error": f"Upload failed: {response.status_code} - {response.text}"
            }
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def get_demo_response(user_input):
    """Generate demo response with citations including images"""
    response = random.choice(DEMO_RESPONSES)
    
    # Customize response based on legal keywords
    if any(word in user_input.lower() for word in ["contract", "agreement", "deal"]):
        response = DEMO_RESPONSES[0]
    elif any(word in user_input.lower() for word in ["tort", "negligence", "liability"]):
        response = DEMO_RESPONSES[1]
    
    return response

def create_demo_image():
    """Create a simple demo image for citations"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple diagram
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple legal diagram
        draw.rectangle([50, 50, 350, 250], outline='black', width=2)
        draw.text((200, 60), "LEGAL CONCEPT", fill='black', anchor="mm")
        draw.rectangle([70, 100, 170, 140], outline='blue', width=2)
        draw.text((120, 120), "Element 1", fill='blue', anchor="mm")
        draw.rectangle([230, 100, 330, 140], outline='blue', width=2)
        draw.text((280, 120), "Element 2", fill='blue', anchor="mm")
        draw.rectangle([70, 180, 170, 220], outline='green', width=2)
        draw.text((120, 200), "Requirement", fill='green', anchor="mm")
        draw.rectangle([230, 180, 330, 220], outline='green', width=2)
        draw.text((280, 200), "Application", fill='green', anchor="mm")
        
        return img
    except ImportError:
        return None

def display_image_citation(image_desc):
    """Display a demo image for citation"""
    img = create_demo_image()
    if img:
        st.image(img, caption=image_desc, width=300)
    else:
        st.info(f"📊 {image_desc}")

def parse_and_display_citations(citation_data):
    """Parse and display citation data from API response"""
    citations_displayed = {}
    
    if not citation_data:
        return citations_displayed
    
    cited_images = citation_data.get("cited_images", {})
    cited_tables = citation_data.get("cited_tables", {})
    
    # Display cited tables
    if cited_tables:
        st.markdown("#### 📊 Referenced Tables")
        for table_id, table_html in cited_tables.items():
            with st.expander(f"📋 {table_id.upper()}", expanded=False):
                # Clean and display HTML table
                st.markdown(table_html, unsafe_allow_html=True)
                citations_displayed[f"TABLE:{table_id}"] = "table"
    
    # Display cited images
    if cited_images:
        st.markdown("#### 🖼️ Referenced Images")
        for image_id, image_data in cited_images.items():
            with st.expander(f"🖼️ {image_id.upper()}", expanded=False):
                try:
                    # Handle base64 image data
                    if isinstance(image_data, str) and image_data.startswith('data:image'):
                        # Remove data:image/jpeg;base64, prefix if present
                        base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                        image_bytes = base64.b64decode(base64_data)
                        st.image(image_bytes, caption=f"Referenced Image: {image_id}")
                    elif isinstance(image_data, str):
                        # Assume it's raw base64
                        image_bytes = base64.b64decode(image_data)
                        st.image(image_bytes, caption=f"Referenced Image: {image_id}")
                    else:
                        st.info(f"📷 Image reference: {image_id}")
                except Exception as e:
                    st.error(f"Error displaying image {image_id}: {str(e)}")
                
                citations_displayed[f"IMAGE:{image_id}"] = "image"
    
    return citations_displayed

def create_citation_links(response_text, citations_displayed):
    """Create clickable citation links in response text"""
    if not citations_displayed:
        return response_text
    
    # Pattern to match [TABLE:table_id] or [IMAGE:image_id]
    citation_pattern = r'\[([A-Z]+):([a-zA-Z0-9_]+)\]'
    
    def replace_citation(match):
        citation_type = match.group(1)  # TABLE or IMAGE
        citation_id = match.group(2)   # table_001, image_001, etc.
        full_citation = f"{citation_type}:{citation_id}"
        
        if full_citation in citations_displayed:
            # Create a highlighted citation reference
            icon = "📋" if citation_type == "TABLE" else "🖼️"
            return f"**{icon} [{citation_type}:{citation_id}]**"
        else:
            return match.group(0)  # Return original if not found
    
    # Replace all citation patterns
    enhanced_text = re.sub(citation_pattern, replace_citation, response_text)
    
    return enhanced_text

def display_image_analysis_result(result_data):
    """Display the direct AI analysis result for images"""
    st.markdown("### 🤖 AI Analysis Results")
    
    # Main summary
    summary = result_data.get("summary", "No summary available")
    st.markdown("#### 📝 Analysis Summary")
    st.markdown(summary)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📷 Images Processed", result_data.get("document_count", 0))
    
    with col2:
        processed_at = result_data.get("processed_at", "")
        if processed_at:
            # Parse and format the datetime
            try:
                dt = datetime.fromisoformat(processed_at.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%H:%M:%S")
                st.metric("⏰ Processed At", formatted_time)
            except:
                st.metric("⏰ Processed At", processed_at[:10])
        else:
            st.metric("⏰ Processed At", "N/A")
    
    with col3:
        st.metric("🆔 Common ID", result_data.get("common_id", "N/A")[:8] + "...")
    
    with col4:
        status = result_data.get("status", "unknown")
        status_emoji = "✅" if status == "success" else "❌"
        st.metric("📊 Status", f"{status_emoji} {status.title()}")
    
    # File details
    filenames = result_data.get("filenames", [])
    if filenames:
        st.markdown("#### 📁 Processed Files")
        for i, filename in enumerate(filenames):
            st.write(f"{i+1}. **{filename}**")
    
    # Validation errors if any
    validation_errors = result_data.get("validation_errors", [])
    if validation_errors:
        st.markdown("#### ⚠️ Validation Issues")
        for error in validation_errors:
            st.warning(f"• {error}")
        
        processed_count = result_data.get("processed_count", 0)
        total_submitted = result_data.get("total_submitted", 0)
        if total_submitted > processed_count:
            st.info(f"Successfully processed {processed_count} out of {total_submitted} submitted files.")

def render_uploader():
    """Render file uploader in sidebar with mode selection"""
    st.sidebar.header("📄 Upload Documents")
    
    # Mode selection
    upload_mode = st.sidebar.radio(
        "Select upload mode:",
        options=["PDF Documents", "Images"],
        index=0,
        help="Choose between PDF document analysis or image analysis"
    )
    
    # Update file types and help text based on mode
    if upload_mode == "Images":
        file_types = ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]
        help_text = "Upload images for AI analysis. Max 10 images, 50MB per file."
        mode_icon = "🖼️"
        mode_text = "*Supported: Images (Max 50MB per file, Max 10 files)*"
    else:
        file_types = ["pdf"]
        help_text = "Upload contracts, legal briefs, case law, statutes, etc. Multiple files will be processed together."
        mode_icon = "📄"
        mode_text = "*Supported: PDF (Max 50MB per file)*"
    
    st.sidebar.markdown(mode_text)
    
    uploaded_files = st.sidebar.file_uploader(
        f"Upload {upload_mode.lower()}", 
        type=file_types, 
        accept_multiple_files=True,
        help=help_text
    )
    
    # Show file count and total size
    if uploaded_files:
        total_size = sum(file.size for file in uploaded_files)
        file_count = len(uploaded_files)
        
        # Check limits for images
        if upload_mode == "Images" and file_count > 10:
            st.sidebar.error("⚠️ Maximum 10 images allowed. Please select fewer files.")
            return
        
        st.sidebar.info(f"{mode_icon} **{file_count} file(s) selected**\n📏 **Total size:** {total_size / (1024*1024):.1f} MB")
        
        # Show individual files
        with st.sidebar.expander("📋 Selected Files"):
            for i, file in enumerate(uploaded_files):
                st.write(f"{i+1}. **{file.name}** ({file.size / (1024*1024):.1f} MB)")
    
    # Process button
    button_text = f"{mode_icon} Process {upload_mode}"
    if st.sidebar.button(button_text) and uploaded_files:
        # Check file sizes (50MB limit per file)
        oversized_files = [f.name for f in uploaded_files if f.size > 50 * 1024 * 1024]
        if oversized_files:
            st.sidebar.error(f"⚠️ Files too large (Max 50MB per file):\n" + "\n".join(f"• {name}" for name in oversized_files))
            return
        
        # Enhanced processing with detailed progress
        with st.sidebar:
            # Determine processing type
            file_count = len(uploaded_files)
            is_multiple = file_count > 1
            is_image_mode = upload_mode == "Images"
            
            if is_image_mode:
                processing_title = f"🔄 Analyzing {file_count} Image{'s' if is_multiple else ''}"
                endpoint_text = "/upload/multiple/images"
            else:
                processing_title = f"🔄 Processing {file_count} Document{'s' if is_multiple else ''}"
                endpoint_text = f"/upload/{'pdfs' if is_multiple else 'pdf'}"
            
            # Main status container
            with st.status(processing_title, expanded=True) as status:
                # Show file information
                st.write(f"📋 **Files:** {file_count} {upload_mode.lower()}")
                st.write(f"📏 **Total Size:** {sum(f.size for f in uploaded_files) / (1024*1024):.1f} MB")
                st.write(f"🔗 **Endpoint:** `{endpoint_text}`")
                
                # Overall progress bar
                progress_bar = st.progress(0)
                stage_info = st.empty()
                
                # Processing stages with detailed updates
                mode_key = "image" if is_image_mode else "pdf"
                stages = simulate_detailed_progress(mode_key)
                
                # Adjust stage names for multiple files
                if is_multiple and not is_image_mode:
                    stages = [
                        (f"📤 Uploading {file_count} Documents", 0, 15, 3),
                        ("📊 Document Partitioning", 15, 30, 4),
                        ("✂️ Text Chunking", 30, 50, 5),
                        ("🔍 Atomic Element Extraction", 50, 75, 6),
                        ("🧠 Generating Embeddings", 75, 92, 5),
                        ("💾 Uploading to Vector Database", 92, 100, 3)
                    ]
                elif is_multiple and is_image_mode:
                    stages = [
                        (f"📤 Uploading {file_count} Images", 0, 20, 3),
                        ("🔍 Multi-Image Analysis", 20, 50, 4),
                        ("🧠 AI Processing", 50, 80, 5),
                        ("📝 Generating Combined Summary", 80, 100, 3)
                    ]
                
                try:
                    for stage_name, start_pct, end_pct, duration in stages:
                        status.update(label=f"🔄 {stage_name}", state="running")
                        stage_info.info(f"**Current Stage:** {stage_name}")
                        
                        # Show sub-progress for each stage
                        if DEMO_MODE:
                            # Simulate detailed progress for demo
                            for i in range(start_pct, end_pct + 1, 2):
                                progress_bar.progress(i, text=f"{stage_name}... {i}%")
                                time.sleep(duration / ((end_pct - start_pct) / 2))
                        else:
                            # For real API, show stage start and let it complete
                            progress_bar.progress(start_pct, text=f"{stage_name}... Starting")
                    
                    # Call appropriate API based on mode
                    if not DEMO_MODE:
                        stage_info.info("**Current Stage:** 🌐 Calling API...")
                        if is_image_mode:
                            result = upload_images_to_api(uploaded_files)
                        else:
                            result = upload_files_to_api(uploaded_files) if is_multiple else upload_file_to_api(uploaded_files[0])
                    else:
                        # Demo mode
                        if is_image_mode:
                            result = {
                                "success": True,
                                "data": {
                                    "status": "success",
                                    "filenames": [f.name for f in uploaded_files],
                                    "summary": "This is a demo analysis of the uploaded images. The AI has successfully processed all images and generated insights.",
                                    "processed_at": datetime.now().isoformat(),
                                    "source": "demo",
                                    "common_id": "demo_" + str(int(time.time())),
                                    "document_count": len(uploaded_files)
                                }
                            }
                        else:
                            result = upload_files_to_api(uploaded_files) if is_multiple else upload_file_to_api(uploaded_files[0])
                    
                    if result["success"]:
                        # Success state
                        progress_bar.progress(100, text="✅ Processing Complete!")
                        
                        if is_image_mode:
                            success_msg = f"✅ Images Analyzed Successfully"
                        else:
                            success_msg = f"✅ Documents Processed Successfully" if is_multiple else "✅ Document Processed Successfully"
                        
                        status.update(label=success_msg, state="complete")
                        
                        data = result["data"]
                        
                        # Handle different response types
                        if is_image_mode:
                            # Display image analysis results immediately
                            st.success(f"🎉 **{file_count} Image{'s' if is_multiple else ''} Analyzed!**")
                            
                            # Show the AI analysis result
                            display_image_analysis_result(data)
                            
                            # Store in session state for history
                            if "image_analyses" not in st.session_state:
                                st.session_state.image_analyses = []
                            
                            st.session_state.image_analyses.append({
                                "filenames": data.get("filenames", []),
                                "summary": data.get("summary", ""),
                                "processed_at": data.get("processed_at", ""),
                                "common_id": data.get("common_id", ""),
                                "document_count": data.get("document_count", 0),
                                "status": data.get("status", "unknown"),
                                "validation_errors": data.get("validation_errors", []),
                                "timestamp": time.time()  # Add timestamp for comparison
                            })
                            
                            # Store current image context for chat
                            st.session_state.current_image_context = {
                                "common_id": data.get("common_id", ""),
                                "document_count": data.get("document_count", 0),
                                "filenames": data.get("filenames", [])
                            }
                            
                        else:
                            # PDF processing - existing logic
                            # Display processing results
                            if is_multiple:
                                st.success(f"🎉 **{file_count} Documents Ready for Analysis!**")
                            else:
                                st.success("🎉 **Document Ready for Analysis!**")
                            
                            # Processing statistics in expandable section
                            with st.expander("📊 Detailed Processing Statistics", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("⏱️ Processing Time", f"{data.get('total_time_seconds', 0):.1f}s")
                                    st.metric("📄 Total Chunks", data.get('upload_stats', {}).get('total_chunks', 0))
                                with col2:
                                    collection_name = data.get('collection_name', 'N/A')
                                    st.metric("📁 Collection", collection_name[:20] + ("..." if len(collection_name) > 20 else ""))
                                    doc_count = data.get('total_documents', file_count)
                                    st.metric("📋 Documents", doc_count)
                            
                            # Store for chat functionality
                            if "processed_documents" not in st.session_state:
                                st.session_state.processed_documents = []
                            
                            # Create file list for display
                            file_names = [f.name for f in uploaded_files]
                            combined_name = file_names[0] if len(file_names) == 1 else f"{file_names[0]} + {len(file_names)-1} more"
                            
                            st.session_state.processed_documents.append({
                                "filename": combined_name,
                                "file_list": file_names,
                                "collection_name": data.get("collection_name"),
                                "total_chunks": data.get('upload_stats', {}).get('total_chunks', 0),
                                "processing_time": data.get('total_time_seconds', 0),
                                "file_count": file_count,
                                "is_multiple": is_multiple,
                                "timestamp": time.time()  # Add timestamp for comparison
                            })
                            
                            # Store current collection for chat
                            if DEMO_MODE:
                                st.session_state.current_collection = "embeddings_sbi_vehicle_insurrance_20250918_145033"
                            else:
                                st.session_state.current_collection = data.get("collection_name")
                    
                    else:
                        # Error state
                        progress_bar.progress(0, text="❌ Processing Failed")
                        status.update(label="❌ Processing Failed", state="error")
                        st.error(f"**Error:** {result['error']}")
                        
                        # Error details
                        with st.expander("🔍 Error Details"):
                            st.code(result['error'])
                            endpoint_info = f"`{endpoint_text}`"
                            st.info(f"""💡 **Troubleshooting:**
- Check if FastAPI server is running on port 8080
- Verify endpoint {endpoint_info} is available
- Ensure files are not corrupted
- Try smaller file sizes
- Check server logs for detailed error information""")
                
                except Exception as e:
                    progress_bar.progress(0, text="❌ Unexpected Error")
                    status.update(label="❌ Unexpected Error", state="error")
                    st.error(f"**Unexpected Error:** {str(e)}")
    
    # Show processed documents (PDF mode only)
    if upload_mode == "PDF Documents" and st.session_state.get("processed_documents"):
        st.sidebar.markdown("### 📁 Processed Document Sets")
        for i, doc in enumerate(st.session_state.processed_documents):
            is_current = i == len(st.session_state.processed_documents) - 1
            icon = "📄" if doc['file_count'] == 1 else "📚"
            title = f"{icon} {doc['filename']}"
            
            with st.sidebar.expander(title, expanded=is_current):
                if doc.get('is_multiple'):
                    st.write(f"**📋 {doc['file_count']} Files:**")
                    for j, fname in enumerate(doc.get('file_list', [])[:3]):  # Show first 3
                        st.write(f"  {j+1}. {fname}")
                    if len(doc.get('file_list', [])) > 3:
                        st.write(f"  ... and {len(doc['file_list']) - 3} more")
                
                st.metric("📄 Total Chunks", doc['total_chunks'])
                st.metric("⏱️ Processing Time", f"{doc['processing_time']:.1f}s")
                st.caption(f"Collection: `{doc['collection_name']}`")
    
    # Show image analyses history
    if upload_mode == "Images" and st.session_state.get("image_analyses"):
        st.sidebar.markdown("### 🖼️ Image Analysis History")
        for i, analysis in enumerate(st.session_state.image_analyses[-3:]):  # Show last 3
            timestamp = datetime.fromtimestamp(analysis['timestamp']).strftime("%H:%M")
            files_text = f"{analysis['document_count']} image{'s' if analysis['document_count'] > 1 else ''}"
            
            with st.sidebar.expander(f"🖼️ {files_text} - {timestamp}", expanded=i == len(st.session_state.image_analyses[-3:]) - 1):
                st.write(f"**📷 Files:** {analysis['document_count']}")
                st.write(f"**🆔 ID:** {analysis['common_id'][:12]}...")
                st.caption(f"Status: {analysis['status']}")
                
                # Show first few filenames
                for j, fname in enumerate(analysis['filenames'][:2]):
                    st.write(f"  {j+1}. {fname}")
                if len(analysis['filenames']) > 2:
                    st.write(f"  ... +{len(analysis['filenames']) - 2} more")
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚖️ About")
    st.sidebar.markdown(
        "Legal Document Assistant supports both PDF document analysis and image analysis "
        "with AI-powered insights and interactive citations."
    )
    
    with st.sidebar.expander("✨ Features"):
        st.markdown(f"""
        **📄 PDF Mode:**
        - Single & Multiple PDF Upload (Max 50MB per file)
        - AI Legal Analysis with Chat
        - Interactive Tables & Citations
        - Vector Database Storage
        
        **🖼️ Image Mode:**
        - Multiple Image Upload (Max 10 files, 50MB each)
        - Direct AI Analysis & Summary
        - Instant Results (No Chat Required)
        - Support for PNG, JPG, JPEG, BMP, GIF, TIFF
        """)
    
    # Show connection status
    if DEMO_MODE:
        st.sidebar.warning("🧪 **Demo Mode Active**")
    else:
        st.sidebar.info(f"🔗 **PDF API:** `{UPLOAD_MULTIPLE_API_URL}`")
        st.sidebar.info(f"🔗 **Image API:** `{UPLOAD_MULTIPLE_IMAGES_API_URL}`")
        st.sidebar.info(f"💬 **Chat API:** `{CHAT_API_URL}/{{collection_id}}`")

def render_chat():
    """Render chat interface for both PDF and Image modes"""
    # Check if we have either processed documents (PDFs) or image analyses
    has_pdfs = st.session_state.get("processed_documents")
    has_images = st.session_state.get("image_analyses")
    
    if not has_pdfs and not has_images:
        return
    
    # Determine current mode based on what was processed most recently
    current_mode = "pdf"
    current_context = None
    
    if has_images and has_pdfs:
        # Compare timestamps to see which was more recent
        latest_pdf_time = st.session_state.processed_documents[-1].get('timestamp', 0) if has_pdfs else 0
        latest_image_time = st.session_state.image_analyses[-1].get('timestamp', 0) if has_images else 0
        
        if latest_image_time > latest_pdf_time:
            current_mode = "image"
            current_context = st.session_state.image_analyses[-1]
        else:
            current_context = st.session_state.processed_documents[-1]
    elif has_images:
        current_mode = "image"
        current_context = st.session_state.image_analyses[-1]
    else:
        current_mode = "pdf"
        current_context = st.session_state.processed_documents[-1]
    
    # Show current context
    if current_mode == "image":
        file_count = current_context['document_count']
        common_id = current_context['common_id']
        st.info(f"🎯 **Currently analyzing:** {file_count} image{'s' if file_count > 1 else ''} | 🆔 ID: {common_id[:12]}... | Mode: Image Analysis")
        
        # Show image files
        with st.expander("📷 Current Image Set", expanded=False):
            for i, filename in enumerate(current_context.get('filenames', [])):
                st.write(f"{i+1}. **{filename}**")
    else:
        # PDF mode - existing logic
        if current_context.get('is_multiple'):
            file_count = current_context['file_count']
            st.info(f"🎯 **Currently analyzing:** {file_count} documents | 📄 {current_context['total_chunks']} chunks | Collection: `{current_context['collection_name']}`")
        else:
            st.info(f"🎯 **Currently analyzing:** `{current_context['filename']}` | 📄 {current_context['total_chunks']} chunks | Collection: `{current_context['collection_name']}`")
    
    st.subheader(f"⚖️ Ask Questions About Your {current_mode.upper()} Analysis")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Query form
    with st.form(key="legal_qa_form"):
        # Update placeholder text based on current mode
        if current_mode == "image":
            placeholder_text = f"e.g., What details can you see in these images? Are there any important numbers or dates?"
        else:
            if current_context.get('is_multiple'):
                placeholder_text = f"e.g., Compare the terms across these {current_context['file_count']} documents"
            else:
                placeholder_text = "e.g., What are the key terms in this document?"
        
        query = st.text_area(
            "Ask a question",
            placeholder=placeholder_text,
            height=100
        )
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col4:
            submit = st.form_submit_button("🔍 Ask", type="primary")
        with col3:
            show_images = st.checkbox("Show visual references", value=True)
        with col2:
            top_k = st.selectbox("Retrieved points", [3, 5, 10, 15], index=1, help="Number of chunks to retrieve")
        with col1:
            if current_mode == "image":
                st.caption(f"💡 Querying {current_context['document_count']} cached image{'s' if current_context['document_count'] > 1 else ''}")
            else:
                if current_context.get('is_multiple'):
                    st.caption(f"💡 Querying across {current_context['file_count']} documents")
                else:
                    st.caption("💡 Searching document with AI analysis")
    
    # Display previous conversations
    if st.session_state.messages:
        st.markdown("### 💭 Chat History")
        for i, msg_pair in enumerate(st.session_state.messages):
            # Create title with citation count and mode indicator
            citation_count = ""
            if msg_pair.get('citation_data'):
                cited_tables = len(msg_pair['citation_data'].get('cited_tables', {}))
                cited_images = len(msg_pair['citation_data'].get('cited_images', {}))
                if cited_tables or cited_images:
                    citation_count = f" (📋{cited_tables} 🖼️{cited_images})"
            
            mode_indicator = "🖼️" if msg_pair.get('mode') == 'image' else "📄"
            
            with st.expander(f"{mode_indicator} Q{i+1}: {msg_pair['question'][:60]}... ({msg_pair.get('processing_time', 0):.1f}s){citation_count}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Answer:**")
                    if msg_pair.get('status') == 'success':
                        # Use enhanced answer if available (with citation links)
                        answer_text = msg_pair.get('enhanced_answer', msg_pair.get('answer', ''))
                        st.markdown(answer_text)
                        
                        # Display citations if available
                        if msg_pair.get('citation_data'):
                            st.markdown("---")
                            citation_data = msg_pair['citation_data']
                            parse_and_display_citations(citation_data)
                            
                    else:
                        st.error(f"❌ {msg_pair.get('answer', 'Error occurred')}")
                
                with col2:
                    st.markdown("**Query Details:**")
                    
                    # Show mode
                    query_mode = msg_pair.get('mode', 'pdf')
                    mode_emoji = "🖼️" if query_mode == 'image' else "📄"
                    st.metric("🔧 Mode", f"{mode_emoji} {query_mode.upper()}")
                    
                    # Show metadata if available (real API response)
                    if 'metadata' in msg_pair and msg_pair['metadata']:
                        metadata = msg_pair['metadata']
                        st.metric("📄 Retrieved Points", metadata.get('retrieved_points', 0))
                        st.metric("🔍 Embedding Time", f"{metadata.get('embedding_time', 0):.3f}s")
                        
                        if query_mode == 'image':
                            st.caption(f"Image ID: `{metadata.get('common_id', 'Unknown')[:12]}...`")
                        else:
                            st.caption(f"Collection: `{metadata.get('collection', 'Unknown')}`")
                        
                        # Citation summary
                        citation_data = metadata.get('citation_data', {})
                        if citation_data:
                            cited_images = citation_data.get("cited_images", {})
                            cited_tables = citation_data.get("cited_tables", {})
                            if cited_tables:
                                st.metric("📋 Tables", len(cited_tables))
                            if cited_images:
                                st.metric("🖼️ Images", len(cited_images))
                    
                    # Show processing time
                    if msg_pair.get('processing_time'):
                        st.metric("⏱️ Total Time", f"{msg_pair['processing_time']:.2f}s")
                    
                    # Show context info
                    if query_mode == 'image':
                        img_context = msg_pair.get('image_context', {})
                        if img_context:
                            st.markdown(f"**Images:** {img_context.get('document_count', 'Unknown')}")
                            st.caption("Image analysis mode")
                    else:
                        doc_set = msg_pair.get('document_set', {})
                        if doc_set.get('is_multiple'):
                            st.markdown(f"**Documents:** {doc_set['file_count']} files")
                            st.caption("Multi-document analysis")
                        else:
                            st.markdown(f"**Document:** `{msg_pair.get('document', 'Unknown')}`")
                            st.caption("Single document analysis")
    
    # Process new query
    if submit and query:
        # Determine which API to call based on current mode
        if current_mode == "image":
            common_id = current_context['common_id']
            
            # Call image chat API
            with st.spinner("🤖 Analyzing your images..."):
                result = chat_with_images(query, common_id, top_k)
        else:
            current_collection = st.session_state.get("current_collection")
            
            # Call document chat API
            with st.spinner("🤖 Analyzing your legal document..."):
                result = chat_with_document(query, current_collection, top_k)
        
        if result["success"]:
            api_response = result["data"]
            
            # Display results in columns
            answer_col, sources_col = st.columns([2, 1])
            
            with answer_col:
                st.markdown("#### 📝 Answer")
                
                if api_response.get("status") == "success":
                    response_text = api_response.get("response", "No response received")
                    
                    # Extract citation data
                    metadata = api_response.get("metadata", {})
                    citation_data = metadata.get("citation_data", {})
                    
                    # Parse and display citations first
                    citations_displayed = parse_and_display_citations(citation_data)
                    
                    # Enhance response text with citation links
                    enhanced_response = create_citation_links(response_text, citations_displayed)
                    
                    # Display the enhanced response
                    st.markdown(enhanced_response)
                    
                else:
                    st.error(f"❌ API Error: {api_response.get('error', 'Unknown error')}")
            
            with sources_col:
                st.markdown("#### 📊 Query Information")
                
                # Display mode
                mode_emoji = "🖼️" if current_mode == 'image' else "📄"
                st.metric("🔧 Mode", f"{mode_emoji} {current_mode.upper()}")
                
                # Display metadata
                metadata = api_response.get("metadata", {})
                if metadata:
                    st.metric("⏱️ Processing Time", f"{api_response.get('processing_time', 0):.2f}s")
                    st.metric("📄 Retrieved Points", metadata.get("retrieved_points", 0))
                    st.metric("🔍 Embedding Time", f"{metadata.get('embedding_time', 0):.3f}s")
                    
                    if current_mode == 'image':
                        st.info(f"**Image ID:** `{current_context['common_id'][:12]}...`")
                    else:
                        st.info(f"**Collection:** `{metadata.get('collection', 'Unknown')}`")
                    
                    # Show citation summary
                    citation_data = metadata.get("citation_data", {})
                    if citation_data:
                        cited_images = citation_data.get("cited_images", {})
                        cited_tables = citation_data.get("cited_tables", {})
                        
                        if cited_tables or cited_images:
                            st.markdown("#### 📚 Citations Summary")
                            if cited_tables:
                                st.metric("📋 Tables Referenced", len(cited_tables))
                            if cited_images:
                                st.metric("🖼️ Images Referenced", len(cited_images))
                
                # Add demo visual references if enabled and no real citations
                if show_images and not citation_data.get("cited_images") and not citation_data.get("cited_tables"):
                    st.markdown("#### 📊 Demo Visual References")
                    st.caption("*Visual aids for demo...*")
                    display_image_citation("Analysis Diagram")
            
            # Save to session state
            chat_entry = {
                "question": query,
                "answer": api_response.get("response", "No response"),
                "enhanced_answer": enhanced_response if 'enhanced_response' in locals() else api_response.get("response", "No response"),
                "metadata": metadata,
                "citation_data": citation_data,
                "citations_displayed": citations_displayed if 'citations_displayed' in locals() else {},
                "processing_time": api_response.get("processing_time", 0),
                "status": api_response.get("status", "unknown"),
                "mode": current_mode
            }
            
            # Add mode-specific context
            if current_mode == 'image':
                chat_entry.update({
                    "common_id": current_context['common_id'],
                    "image_context": current_context
                })
            else:
                chat_entry.update({
                    "collection_name": st.session_state.get("current_collection"),
                    "document": current_context['filename'],
                    "document_set": current_context
                })
            
            st.session_state.messages.append(chat_entry)
            
        else:
            # Handle API error
            st.error(f"❌ Chat Error: {result['error']}")
            if current_mode == 'image':
                st.info("Please check that your image chat endpoint is available")
            else:
                st.info("Please check that your FastAPI server is running on port 8080")

def render_history_download():
    """Render chat history download - only when there are messages"""
    if st.session_state.get("messages"):
        # Create comprehensive chat history
        full_history = "LEGAL DOCUMENT ASSISTANT - CHAT HISTORY\n"
        full_history += "=" * 50 + "\n\n"
        
        for i, msg in enumerate(st.session_state.messages):
            full_history += f"QUESTION {i+1}: {msg['question']}\n\n"
            full_history += f"ANSWER: {msg['answer']}\n\n"
            
            # Add document context
            doc_set = msg.get('document_set', {})
            if doc_set.get('is_multiple'):
                full_history += f"DOCUMENT CONTEXT: Multi-document analysis ({doc_set['file_count']} files)\n"
                full_history += "FILES ANALYZED:\n"
                for j, fname in enumerate(doc_set.get('file_list', [])[:10]):
                    full_history += f"  {j+1}. {fname}\n"
                if len(doc_set.get('file_list', [])) > 10:
                    full_history += f"  ... and {len(doc_set['file_list']) - 10} more files\n"
                full_history += "\n"
            else:
                full_history += f"DOCUMENT CONTEXT: Single document analysis\n"
                full_history += f"FILE: {msg.get('document', 'Unknown')}\n\n"
            
            full_history += "\n" + "="*50 + "\n\n"
        
        col1, col2 = st.columns(2)
        with col1:
            # Determine filename suffix - safe access with get()
            processed_docs = st.session_state.get("processed_documents", [])
            suffix = "multi_docs" if processed_docs and processed_docs[-1].get('is_multiple') else "single_doc"
            filename = f"legal_chat_history_{suffix}_{int(time.time())}.txt"
            
            st.download_button(
                "📥 Download Chat History",
                full_history,
                file_name=filename,
                mime="text/plain",
                help="Download your legal research session"
            )
        with col2:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

def initialize_session_state():
    """Initialize all session state variables"""
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []
    
    if "image_analyses" not in st.session_state:
        st.session_state.image_analyses = []
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_collection" not in st.session_state:
        st.session_state.current_collection = None
    
    if "current_image_context" not in st.session_state:
        st.session_state.current_image_context = None

def main():
    """Main application"""
    # Initialize session state first
    initialize_session_state()
    
    st.title("⚖️ Legal Document Assistant")
    st.markdown("*AI-Powered PDF Analysis & Image Intelligence*")
    
    # Render components
    render_uploader()
    
    # Main content area - show chat for both PDF and Image modes
    if st.session_state.get("processed_documents") or st.session_state.get("image_analyses"):
        render_chat()
        
        # Add spacing
        st.markdown("---")
        
        # Download section (only show for PDF mode with chat history)
        if st.session_state.get("messages"):
            render_history_download()
    
    # Welcome message for new users
    if (not st.session_state.get("messages") and 
        not st.session_state.get("processed_documents") and 
        not st.session_state.get("image_analyses")):
        
        st.info("""
        ### 👋 Welcome to Legal Document Assistant!
        
        **🔄 Dual Mode Support:**
        
        **📄 PDF Mode - Document Analysis & Chat:**
        1. Upload PDF legal documents (Max 50MB per file)
        2. Watch real-time processing through 6 stages
        3. Chat and ask questions about your documents
        4. Get interactive citations with tables and images
        5. Download your research session
        
        **🖼️ Image Mode - Direct AI Analysis:**
        1. Upload images (Max 10 files, 50MB each)
        2. Get instant AI analysis and summary
        3. View processing statistics and results
        4. No chat required - immediate insights
        
        **✨ Features:**
        - **Smart Processing**: Automatic endpoint selection
        - **Real-time Progress**: Detailed stage monitoring
        - **Multi-format Support**: PDFs, PNG, JPG, JPEG, BMP, GIF, TIFF
        - **Error Recovery**: Comprehensive error handling
        - **Validation**: File size and type checking
        
        *Choose your mode in the sidebar and start uploading!*
        """)
    
    # Show API status
    if not DEMO_MODE:
        with st.expander("🔧 API Connection Status"):
            st.code(f"PDF Upload: {UPLOAD_MULTIPLE_API_URL}")
            st.code(f"Image Upload: {UPLOAD_MULTIPLE_IMAGES_API_URL}")
            st.code(f"Chat API: {CHAT_API_URL}/{{collection_id}}")
            st.caption("✅ Make sure your FastAPI server is running on port 8080")

if __name__ == "__main__":
    main()