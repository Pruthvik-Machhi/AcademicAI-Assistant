#!/usr/bin/env python
"""
Run script for Academic AI Assistant

This script first ensures the database is initialized, then launches the Streamlit application.
"""

import os
import subprocess
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_postgres():
    """Check if PostgreSQL is running"""
    try:
        # Get PostgreSQL connection settings from environment
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        
        # Simple check if the PostgreSQL port is open
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex((db_host, int(db_port)))
        s.close()
        
        if result == 0:
            print("✓ PostgreSQL server is running.")
            return True
        else:
            print("✗ PostgreSQL server is not running. Please start it before continuing.")
            return False
            
    except Exception as e:
        print(f"Error checking PostgreSQL: {e}")
        return False

def check_groq_api_key():
    """Check if Groq API key is set"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key":
        print("✗ Groq API key not set in .env file.")
        print("  This application requires a Groq API key to function properly.")
        print("  Please set the GROQ_API_KEY value in your .env file.")
        return False
    else:
        print("✓ Groq API key found in environment variables.")
        return True

def initialize_database():
    """Initialize the database using the init_db.py script"""
    print("Initializing database...")
    try:
        result = subprocess.run([sys.executable, "init_db.py"], check=True)
        if result.returncode == 0:
            print("✓ Database initialized successfully.")
            return True
        else:
            print("✗ Database initialization failed.")
            return False
    except subprocess.CalledProcessError:
        print("✗ Database initialization script failed.")
        return False
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    print("Starting Academic AI Assistant...")
    subprocess.run([
        "streamlit", "run", "academic_ai_assistant.py", 
        "--browser.serverAddress=localhost",
        "--browser.gatherUsageStats=false",
        "--server.enableCORS=false"
    ])

def main():
    """Main function to run the application"""
    print("=" * 60)
    print("Academic AI Assistant - Startup")
    print("=" * 60)
    
    # Check if PostgreSQL is running
    if not check_postgres():
        print("Please start your PostgreSQL server and try again.")
        sys.exit(1)
    
    # Check if Groq API key is set
    check_groq_api_key()
    
    # Initialize database
    if not initialize_database():
        print("Database initialization failed. Please check your PostgreSQL settings in .env file.")
        sys.exit(1)
    
    # Run Streamlit application
    run_streamlit()

if __name__ == "__main__":
    main() 