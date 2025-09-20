#!/usr/bin/env python3
"""
Simple script to run the Streamlit app with proper configuration
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import requests
        import PIL
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def run_app():
    """Run the Streamlit application"""
    if not check_dependencies():
        return
    
    print("🚀 Starting Streamlit app...")
    print("📝 Make sure to update your API URL in config.py")
    print("🌐 App will open at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped")

if __name__ == "__main__":
    run_app()