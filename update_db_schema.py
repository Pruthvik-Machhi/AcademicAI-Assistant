"""
Database schema update script for Academic AI Assistant

This script updates the existing database schema by adding missing columns to tables as needed.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

# Load environment variables from .env file
load_dotenv()

# PostgreSQL Connection Settings 
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "academic_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

def update_notes_table():
    """Add source_type and source_url columns to the notes table if they don't exist"""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        
        # Check if source_type column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'notes' AND column_name = 'source_type'
        """)
        
        if not cursor.fetchone():
            print("Adding 'source_type' column to notes table...")
            cursor.execute("""
                ALTER TABLE notes
                ADD COLUMN source_type VARCHAR(50)
            """)
            print("Column 'source_type' added successfully.")
        else:
            print("Column 'source_type' already exists in notes table.")
        
        # Check if source_url column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'notes' AND column_name = 'source_url'
        """)
        
        if not cursor.fetchone():
            print("Adding 'source_url' column to notes table...")
            cursor.execute("""
                ALTER TABLE notes
                ADD COLUMN source_url TEXT
            """)
            print("Column 'source_url' added successfully.")
        else:
            print("Column 'source_url' already exists in notes table.")
        
        # Commit the changes
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error updating notes table: {e}")
        return False

def main():
    """Main function to update the database schema"""
    print("Updating Academic AI Assistant database schema...")
    
    # Update notes table
    if not update_notes_table():
        print("Failed to update notes table. Exiting.")
        sys.exit(1)
    
    print("Database schema update completed successfully.")

if __name__ == "__main__":
    main() 