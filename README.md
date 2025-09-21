Legal Document RAG System
Overview
This project is a sophisticated legal document processing and question-answering system developed for the Google GenAI Hack2Skill hackathon 2025. Instead of building a traditional open-source RAG system that requires extensive API key management and setup, we've created a managed service that abstracts the complexity away from end users.
Architecture
The system employs a microservices architecture with the following components:

Frontend: Flask-based web application for user interaction
Backend: FastAPI service handling document processing and AI interactions
Nginx: Reverse proxy for routing and load balancing
Redis: Caching layer for image processing and session management
Cloudflared: Tunnel service for secure external access

Key Features
Advanced Document Processing

Semantic Chunking: Uses Unstructured.io API for sophisticated document parsing
Multi-modal Processing: Handles text, images, tables, and infographics in legal documents
Enhanced Text Generation: Expands chunks with contextual information from images and tables
Atomic Element Processing: Breaks documents into headers, images, tables, texts, titles, and narrative texts

Intelligent Embeddings

Custom Embedding Pipeline: Uses EmbeddingGemma 2025 (300M model) running on ONNX runtime
Batch Processing: Optimized for performance with grouped embedding generation
Vector Storage: Utilizes Qdrant for efficient vector storage and retrieval

Image Processing Strategy

Large Context Window: Leverages Gemini's extensive context capabilities
Redis Caching: Intelligent caching system using image byte hashes
Multi-image Support: Handles multiple images with combined hash-based identification

Quick Start
Prerequisites

Docker and Docker Compose
Google Cloud Platform account with Vertex AI access
GCP service account key file

Setup Instructions

Clone the repository

