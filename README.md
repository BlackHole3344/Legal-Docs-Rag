# ⚖️ Legal Document RAG System

## 📋 Overview

This project is a sophisticated legal document processing and question-answering system developed for the Google GenAI Hack2Skill hackathon 2025. Instead of building a traditional open-source RAG system that requires extensive API key management and setup, we've created a managed service that abstracts the complexity away from end users.

## 🏗️ Architecture

The system employs a microservices architecture with the following components:

- **🌐 Frontend**: Flask-based web application for user interaction
- **⚡ Backend**: FastAPI service handling document processing and AI interactions
- **🔀 Nginx**: Reverse proxy for routing and load balancing
- **💾 Redis**: Caching layer for image processing and session management
- **☁️ Cloudflared**: Tunnel service for secure external access

## ✨ Key Features

### 📄 Advanced Document Processing
- **🧩 Semantic Chunking**: Uses Unstructured.io API for sophisticated document parsing
- **🎭 Multi-modal Processing**: Handles text, images, tables, and infographics in legal documents
- **🔍 Enhanced Text Generation**: Expands chunks with contextual information from images and tables
- **⚛️ Atomic Element Processing**: Breaks documents into headers, images, tables, texts, titles, and narrative texts

### 🧠 Intelligent Embeddings
- **🔧 Custom Embedding Pipeline**: Uses EmbeddingGemma 2025 (300M model) running on ONNX runtime
- **📦 Batch Processing**: Optimized for performance with grouped embedding generation
- **🗂️ Vector Storage**: Utilizes Qdrant for efficient vector storage and retrieval

### 🖼️ Image Processing Strategy
- **📏 Large Context Window**: Leverages Gemini's extensive context capabilities
- **⚡ Redis Caching**: Intelligent caching system using image byte hashes
- **🖼️➕ Multi-image Support**: Handles multiple images with combined hash-based identification

## 🚀 Quick Start

### 📋 Prerequisites
- 🐳 Docker and Docker Compose
- ☁️ Google Cloud Platform account with Vertex AI access
- 🔑 GCP service account key file

### ⚙️ Setup Instructions

1. **📥 Clone the repository**
   ```bash
   git clone <repository-url>
   cd legal-document-rag
   ```

2. **🔧 Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration values
   ```

3. **🔐 Add GCP credentials**
   ```bash
   # Place your GCP service account key file
   cp /path/to/your/service-account-key.json ./backend/key.json
   ```

4. **🚀 Launch the application**
   ```bash
   docker compose up -d --build
   ```

5. **🌐 Access the application**
   - 🏠 Main application: `http://localhost:8087`
   - 📖 Backend API docs: `http://localhost:8087/docs` (via Nginx proxy)
   - 💾 Redis: `localhost:6379`

## ⚙️ Environment Configuration

Create a `.env` file based on `.env.example` with the following required variables:

```env
# ☁️ Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp_key
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=your-preferred-location

# 🔑 API Keys
UNSTRUCTURED_API_KEY=your-unstructured-api-key
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_URL=your-qdrant-cluster-url

# 🔧 Application Configuration
BACKEND_URL=http://backend:3000
REDIS_URL=redis://redis:6379

# 🌐 Optional: Cloudflared Configuration
CLOUDFLARED_TUNNEL_TOKEN=your-tunnel-token
```

## 🏛️ Service Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Nginx     │    │   Frontend   │    │   Backend   │
│ (Port 8087) │────│    (Flask)   │────│  (FastAPI)  │
└─────────────┘    └──────────────┘    └─────────────┘
                           │                    │
                           │                    │
                   ┌───────────────┐    ┌─────────────┐
                   │     Redis     │    │   Qdrant    │
                   │   (Cache)     │    │ (Vectors)   │
                   └───────────────┘    └─────────────┘
```

## API Endpoints

### Document Processing
- `POST /upload/pdf` - Upload and process PDF documents
- `POST /upload/image` - Upload and process images
- `GET /collections/{collection_id}` - Retrieve document collection

### Chat Interface
- `POST /chat/{collection_id}` - Chat with processed documents
- `POST /chat/image/{image_id}` - Chat about uploaded images

### Health & Monitoring
- `GET /health` - Service health check
- `GET /docs` - API documentation

## Development

### Project Structure
```
.
├── backend/           # FastAPI backend service
├── frontend/          # Flask frontend application
├── nginx/            # Nginx configuration
├── cloudflared/      # Cloudflared tunnel setup
├── uploads/          # Document upload directory
├── docker-compose.yml
├── .env.example
└── README.md
```

### Running in Development Mode

For development, you can run services individually:

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 3000

# Frontend
cd frontend
pip install -r requirements.txt
flask run --port 5000
```

## Monitoring and Logging

- **Health Checks**: Automated health monitoring for all services
- **Container Logs**: Access via `docker compose logs <service-name>`
- **Redis Monitoring**: Connect to Redis CLI for cache inspection

## Performance Optimizations

- **Batch Embedding Processing**: Reduces API calls and improves throughput
- **Intelligent Caching**: Redis-based caching for image processing
- **Semantic Chunking**: Preserves document context and meaning
- **ONNX Runtime**: Optimized model inference for embeddings

## Troubleshooting

### Common Issues

1. **GCP Authentication Errors**
   - Verify `key.json` is placed in `./backend/` directory
   - Check GCP service account permissions

2. **Port Conflicts**
   - Ensure ports 8087 and 6379 are available
   - Modify docker-compose.yml if needed

3. **Memory Issues**
   - Increase Docker memory allocation for large document processing
   - Monitor Redis memory usage

### Logs
```bash
# View all service logs
docker compose logs

# View specific service logs
docker compose logs backend
docker compose logs frontend
docker compose logs nginx
```

## Contributing

This project was developed for the Google GenAI Hack2Skill hackathon. While not currently configured for open-source contributions, we plan to add comprehensive contribution guidelines in future releases.

## License

[Add your license information here]

## Contact

[Add contact information for the development team]
