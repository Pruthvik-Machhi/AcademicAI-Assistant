import streamlit as st
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
from pathlib import Path
import sys
import asyncio
from src.LLM import GroqLLaMa  # Import the LLM class
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import nest_asyncio
import uuid
import validators
import json
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from src.agents.notewriter import get_notewriter
from src.agents.planner import get_planner
from src.agents.advisor import get_advisor
from src.extractors import extract_youtube_id

# RAG components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_qa_with_sources_chain, create_history_aware_retriever, RetrievalQA
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Add source directory to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables from .env file
load_dotenv()

# PostgreSQL Connection Settings 
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "academic_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Apply nest_asyncio to allow asyncio to work in Streamlit 
# This enables asynchronous content extraction in the app
nest_asyncio.apply()

def init_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.OperationalError:
        # Database might not exist yet, connect to default postgres db
        conn = psycopg2.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if our DB exists, if not create it
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,))
        if not cursor.fetchone():
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
        
        cursor.close()
        conn.close()
        
        # Now connect to our database
        return psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )

def init_db():
    """Initialize database tables if they don't exist"""
    conn = init_connection()
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        email VARCHAR(255) UNIQUE,
        learning_style VARCHAR(50),
        study_hours INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id SERIAL PRIMARY KEY,
        student_id INTEGER REFERENCES students(id),
        title VARCHAR(255) NOT NULL,
        description TEXT,
        due_date TIMESTAMP,
        priority VARCHAR(50),
        status VARCHAR(50) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS notes (
        id SERIAL PRIMARY KEY,
        student_id INTEGER REFERENCES students(id),
        title VARCHAR(255) NOT NULL,
        content TEXT,
        subject VARCHAR(100),
        tags TEXT[],
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        content TEXT,
        embedding_vector BYTEA,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()

def check_db_schema():
    """Check if the database schema is up to date"""
    conn = init_connection()
    cursor = conn.cursor()
    
    schema_issues = []
    
    # Check if source_type and source_url columns exist in notes table
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'notes' AND column_name IN ('source_type', 'source_url')
    """)
    
    existing_columns = [row[0] for row in cursor.fetchall()]
    
    if 'source_type' not in existing_columns or 'source_url' not in existing_columns:
        schema_issues.append("The notes table is missing the source_type or source_url columns")
    
    cursor.close()
    conn.close()
    
    return schema_issues

def main():
    st.set_page_config(
        page_title="Academic AI Assistant", 
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize database
    init_db()
    
    # Check if database schema is up to date
    schema_issues = check_db_schema()
    if schema_issues:
        st.warning("""
        ⚠️ Database schema needs to be updated to enable all features.
        
        Please run the following command to update your database:
        ```
        python update_db_schema.py
        ```
        
        Issues detected:
        - """ + "\n- ".join(schema_issues))
    
    # Sidebar for navigation
    st.sidebar.title("Academic AI Assistant")
    st.sidebar.image("https://img.icons8.com/color/96/000000/student-male--v1.png", width=100)
    
    # Navigation options
    pages = {
        "Home": "🏠",
        "Notewriter": "📝",
        "Planner": "📅",
        "Advisor": "🧠",
        "PDF Chat": "💬"
    }
    
    # Create selection box with icons
    selection = st.sidebar.radio(
        "Navigate to",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )
    
    # Display appropriate page based on selection
    if selection == "Home":
        home_page()
    elif selection == "Notewriter":
        notewriter_page()
    elif selection == "Planner":
        planner_page()
    elif selection == "Advisor":
        advisor_page()
    elif selection == "PDF Chat":
        pdf_chat_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 Academic AI Assistant")
    
def home_page():
    st.title("🏠 Academic AI Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Your Personal Academic Companion
        
        Academic AI Assistant is your all-in-one solution for managing your academic life. 
        Our intelligent agents work together to provide you with personalized support:
        
        - **📝 Notewriter**: Generate study materials and summarize lectures
        - **📅 Planner**: Optimize your schedule and manage your academic calendar
        - **🧠 Advisor**: Get personalized learning and time management advice
        
        Get started by exploring the different features using the sidebar navigation.
        """)
        
        st.info("💡 To begin, set up your student profile using the form on the right.")
    
    with col2:
        st.subheader("Quick Profile Setup")
        with st.form("profile_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Email")
            learning_style = st.selectbox(
                "Learning Style",
                ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"]
            )
            study_hours = st.slider("Daily Study Hours", 1, 12, 3)
            
            submit = st.form_submit_button("Save Profile")
            
            if submit and name and email:
                conn = init_connection()
                cursor = conn.cursor()
                
                # Check if email already exists
                cursor.execute("SELECT id FROM students WHERE email = %s", (email,))
                existing_user = cursor.fetchone()
                
                if existing_user:
                    # Update existing user
                    cursor.execute("""
                        UPDATE students 
                        SET name = %s, learning_style = %s, study_hours = %s
                        WHERE email = %s
                    """, (name, learning_style, study_hours, email))
                    user_id = existing_user[0]
                    st.success("Profile updated successfully!")
                else:
                    # Insert new user
                    cursor.execute("""
                        INSERT INTO students (name, email, learning_style, study_hours)
                        VALUES (%s, %s, %s, %s) RETURNING id
                    """, (name, email, learning_style, study_hours))
                    user_id = cursor.fetchone()[0]
                    st.success("Profile created successfully!")
                
                conn.commit()
                cursor.close()
                conn.close()
                
                # Store user_id in session state
                st.session_state['user_id'] = user_id
                st.session_state['user_name'] = name
    
    # Display system overview
    st.markdown("---")
    st.subheader("🔍 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📝 Notewriter")
        st.markdown("""
        - Generate comprehensive notes
        - Summarize lecture content
        - Create flashcards for study
        """)
        st.button("Go to Notewriter", key="home_to_notewriter", on_click=lambda: st.session_state.update({"navigation": "Notewriter"}))
    
    with col2:
        st.markdown("### 📅 Planner")
        st.markdown("""
        - Optimize your study schedule
        - Manage deadlines and tasks
        - Sync with your calendar
        """)
        st.button("Go to Planner", key="home_to_planner", on_click=lambda: st.session_state.update({"navigation": "Planner"}))
    
    with col3:
        st.markdown("### 🧠 Advisor")
        st.markdown("""
        - Get personalized learning advice
        - Optimize study techniques
        - Receive time management tips
        """)
        st.button("Go to Advisor", key="home_to_advisor", on_click=lambda: st.session_state.update({"navigation": "Advisor"}))

def notewriter_page():
    st.title("📝 Notewriter")
    
    st.markdown("""
    The Notewriter agent helps you process academic content and generate study materials 
    tailored to your learning style.
    """)
    
    # Get API key from .env or session state
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key":
        if "api_key" in st.session_state:
            api_key = st.session_state.api_key
        else:
            api_key = st.text_input("Enter Groq API Key (required for AI processing):", type="password")
            if api_key:
                st.session_state.api_key = api_key
    
    # Source type selection
    st.subheader("Select Content Source")
    source_type = st.radio(
        "Choose your source type:",
        ["Text Input", "Web Page", "YouTube Video"],
        horizontal=True
    )
    
    # Create a form for note creation
    with st.form("notewriter_form"):
        title = st.text_input("Note Title")
        subject = st.text_input("Subject")
        
        # Source content based on selected type
        if source_type == "Text Input":
            content = st.text_area("Enter lecture content, readings, or notes to process", height=300)
            source_url = None
            uploaded_file = None
        elif source_type == "Web Page":
            content = None
            source_url = st.text_input("Enter webpage URL:")
            st.info("The Notewriter will extract and process content from the webpage.")
            uploaded_file = None
        elif source_type == "YouTube Video":
            content = None
            source_url = st.text_input("Enter YouTube video URL:")
            if source_url:
                try:
                    video_id = extract_youtube_id(source_url)
                    if video_id:
                        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading video preview: {str(e)}")
            
            st.info("""
            The Notewriter will extract the transcript and process content from the video.
            
            **Note:** This feature requires that the YouTube video has captions/transcripts available.
            Some videos, especially newer ones or those in certain languages, may not have transcripts available.
            """)
            uploaded_file = None
        
        # Additional options
        col1, col2, col3 = st.columns(3)
        with col1:
            output_format = st.selectbox(
                "Output Format",
                ["Comprehensive Notes", "Brief Summary", "Flashcards", "Mind Map"]
            )
        with col2:
            focus_area = st.text_input("Focus Area (optional)", 
                                       placeholder="E.g., historical context, methodology")
        with col3:
            tags = st.text_input("Tags (comma separated)", 
                                 placeholder="E.g., biology, cells, mitosis")
        
        submit = st.form_submit_button("Process Content")
        
        if submit:
            if 'user_id' not in st.session_state:
                st.warning("Please set up your profile first on the Home page.")
            elif not api_key:
                st.warning("Please enter your Groq API key to enable AI processing.")
            elif source_type == "Text Input" and not content:
                st.warning("Please enter some content to process.")
            elif source_type == "Web Page" and not source_url:
                st.warning("Please enter a webpage URL.")
            elif source_type == "YouTube Video" and not source_url:
                st.warning("Please enter a YouTube video URL.")
            elif source_type == "PDF Document" and not uploaded_file:
                st.warning("Please upload a PDF document.")
            elif not title or not subject:
                st.warning("Please provide both a title and subject for your notes.")
            else:
                # Get student profile to determine learning style
                conn = init_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT learning_style
                    FROM students
                    WHERE id = %s
                """, (st.session_state['user_id'],))
                
                result = cursor.fetchone()
                cursor.close()
                
                learning_style = result[0] if result else "Visual"
                
                # Import the notewriter agent
                notewriter = get_notewriter()
                
                if not notewriter:
                    st.error("Failed to initialize the Notewriter agent. Please check your Groq API key.")
                    return
                
                # Process content using the appropriate method
                with st.spinner("Processing your content with AI - this may take a few moments..."):
                    try:
                        # Handle different source types
                        if source_type == "Text Input":
                            source_data = content
                            source_type_code = "text"
                        elif source_type == "Web Page":
                            source_data = source_url
                            source_type_code = "web"
                        elif source_type == "YouTube Video":
                            source_data = source_url
                            source_type_code = "youtube"
                        
                        # Try a specific YouTube validation if needed
                        if source_type_code == "youtube":
                            if not validators.url(source_data):
                                st.error(f"Invalid YouTube URL: {source_data}")
                                st.info("Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)")
                                return
                            
                            if "youtube.com" not in source_data and "youtu.be" not in source_data:
                                st.warning(f"URL doesn't appear to be a YouTube link: {source_data}")
                                continue_anyway = st.button("Continue anyway")
                                if not continue_anyway:
                                    return
                        
                        # Process the source
                        result = asyncio.run(notewriter.process_source(
                            student_id=st.session_state['user_id'],
                            source_type=source_type_code,
                            source=source_data,
                            title=title,
                            subject=subject,
                            focus_area=focus_area,
                            tags=tags,
                            learning_style=learning_style
                        ))
                        
                        if result["success"]:
                            st.success(f"Note '{title}' processed and saved successfully!")
                            
                            # Store the note ID in session state for viewing
                            st.session_state['selected_note_id'] = result["note_id"]
                        else:
                            # Special handling for YouTube extraction failures
                            if source_type_code == "youtube":
                                st.error(f"Failed to extract YouTube content: {result['error']}")
                                
                                # Provide helpful information and alternatives
                                st.warning("""
                                ### ⚠️ YouTube Transcript Extraction Failed
                                
                                This could be due to one of the following reasons:
                                
                                1. **The video doesn't have captions/transcripts available**
                                2. The video might have disabled automatic transcription
                                3. The video might be in a language not supported by the transcript API
                                4. The video might be private or age-restricted
                                
                                ### What you can do:
                                
                                1. **Try a different YouTube video** that has captions
                                2. **Check if the video has captions** (look for the CC button in YouTube's player)
                                3. **Copy the transcript manually** from YouTube:
                                   - Click the "..." button under the video
                                   - Select "Show transcript"
                                   - Copy and paste it as Text Input
                                """)
                                
                                # Option to view available videos with transcripts
                                if st.button("Show Examples of Videos with Transcripts"):
                                    st.markdown("""
                                    ### Examples of Educational Videos with Transcripts:
                                    
                                    - [Khan Academy: Introduction to Limits](https://www.youtube.com/watch?v=riXcZT2ICjA)
                                    - [TED-Ed: The benefits of a bilingual brain](https://www.youtube.com/watch?v=MMmOLN5zBLY)
                                    - [Crash Course: Introduction to Psychology](https://www.youtube.com/watch?v=vo4pMVb0R6M)
                                    - [MIT OpenCourseWare](https://www.youtube.com/c/mitocw/videos)
                                    """)
                            else:
                                st.error(f"Error: {result['error']}")
                            
                            # If extraction failed but we have direct content, still try to save it
                            if source_type == "Text Input":
                                # Save original content
                                tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
                                note_id = notewriter.add_note(
                                    st.session_state['user_id'],
                                    {
                                        "title": title,
                                        "content": content,
                                        "subject": subject,
                                        "tags": tag_list
                                    }
                                )
                                
                                if note_id:
                                    st.info("Original content saved without AI processing.")
                                    st.session_state['selected_note_id'] = note_id
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        
                        # Try to save raw content as fallback
                        if source_type == "Text Input":
                            conn = init_connection()
                            cursor = conn.cursor()
                            
                            # Parse tags
                            tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
                            
                            cursor.execute("""
                                INSERT INTO notes (student_id, title, content, subject, tags)
                                VALUES (%s, %s, %s, %s, %s) RETURNING id
                            """, (st.session_state['user_id'], title, content, subject, tag_list))
                            
                            note_id = cursor.fetchone()[0]
                            conn.commit()
                            cursor.close()
                            conn.close()
                            
                            st.info("Original content saved without AI processing.")
                            st.session_state['selected_note_id'] = note_id
    
    # Display saved notes
    st.markdown("---")
    st.subheader("Your Saved Notes")
    
    if 'user_id' in st.session_state:
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, subject, created_at 
            FROM notes 
            WHERE student_id = %s
            ORDER BY created_at DESC
        """, (st.session_state['user_id'],))
        
        notes = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if notes:
            notes_df = pd.DataFrame(notes, columns=["ID", "Title", "Subject", "Created At"])
            notes_df["Created At"] = pd.to_datetime(notes_df["Created At"]).dt.strftime("%Y-%m-%d %H:%M")
            
            # Display as table with view buttons
            for index, row in notes_df.iterrows():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
                with col1:
                    st.write(row["Title"])
                with col2:
                    st.write(row["Subject"])
                with col3:
                    st.write(row["Created At"])
                with col4:
                    if st.button("View", key=f"view_note_{row['ID']}"):
                        st.session_state['selected_note_id'] = row["ID"]
                        # Clear delete confirmation flag if set
                        if 'delete_confirmation' in st.session_state:
                            del st.session_state['delete_confirmation']
                with col5:
                    if st.button("Delete", key=f"delete_note_{row['ID']}"):
                        st.session_state['delete_note_id'] = row["ID"]
                        st.session_state['delete_confirmation'] = False
            
            # Show delete confirmation
            if 'delete_note_id' in st.session_state and 'delete_confirmation' in st.session_state and not st.session_state['delete_confirmation']:
                note_id = st.session_state['delete_note_id']
                st.warning(f"Are you sure you want to delete this note? This action cannot be undone.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Delete", key=f"confirm_delete_{note_id}"):
                        # Import the notewriter agent
                        notewriter = get_notewriter()
                        
                        if notewriter and notewriter.delete_note(note_id, st.session_state['user_id']):
                            st.success("Note deleted successfully!")
                            st.session_state['delete_confirmation'] = True
                            
                            # Remove the selected note if it's the one being deleted
                            if 'selected_note_id' in st.session_state and st.session_state['selected_note_id'] == note_id:
                                del st.session_state['selected_note_id']
                            
                            # Refresh the page
                            st.rerun()
                        else:
                            st.error("Failed to delete note.")
                with col2:
                    if st.button("Cancel", key=f"cancel_delete_{note_id}"):
                        # Clear deletion state
                        del st.session_state['delete_note_id']
                        del st.session_state['delete_confirmation']
                        st.rerun()
            
            # Show selected note
            if 'selected_note_id' in st.session_state:
                conn = init_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT title, content, subject, tags, source_type, source_url, id
                    FROM notes
                    WHERE id = %s AND student_id = %s
                """, (st.session_state['selected_note_id'], st.session_state['user_id']))
                
                note = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if note:
                    st.markdown("---")
                    # Add note actions in a row
                    note_id = note[6]  # ID is at index 6 now
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.subheader(f"📄 {note[0]}")
                    with col2:
                        if st.button("Delete Note", key=f"delete_current_note_{note_id}"):
                            st.session_state['delete_note_id'] = note_id
                            st.session_state['delete_confirmation'] = False
                            st.rerun()
                    
                    st.caption(f"Subject: {note[2]}")
                    
                    # Display source information if available
                    if note[4]:  # source_type
                        source_type_display = note[4].capitalize()
                        if note[5]:  # source_url
                            st.caption(f"Source: {source_type_display} - [{note[5]}]({note[5]})")
                        else:
                            st.caption(f"Source: {source_type_display}")
                    
                    if note[3]:  # tags
                        st.caption(f"Tags: {', '.join(note[3])}")
                    
                    # Display note content
                    st.markdown(note[1])
                    
                    # Add download button
                    note_text = f"# {note[0]}\n\nSubject: {note[2]}\n"
                    if note[4]:  # source_type
                        source_type_display = note[4].capitalize()
                        if note[5]:  # source_url
                            note_text += f"Source: {source_type_display} - {note[5]}\n"
                        else:
                            note_text += f"Source: {source_type_display}\n"
                    if note[3]:  # tags
                        note_text += f"Tags: {', '.join(note[3])}\n"
                    note_text += f"\n{note[1]}"
                    
                    st.download_button(
                        label="Download Note as Markdown",
                        data=note_text,
                        file_name=f"{note[0].replace(' ', '_')}.md",
                        mime="text/markdown",
                    )
        else:
            st.info("You haven't saved any notes yet.")
    else:
        st.warning("Please set up your profile first on the Home page.")

def planner_page():
    st.title("📅 Planner")
    
    st.markdown("""
    The Planner agent helps you manage your academic schedule and optimize your study time.
    """)
    
    # Task management section
    st.subheader("Task Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("task_form"):
            task_title = st.text_input("Task Title")
            task_description = st.text_area("Description", height=100)
            
            task_col1, task_col2 = st.columns(2)
            with task_col1:
                due_date = st.date_input("Due Date")
                due_time = st.time_input("Due Time")
            with task_col2:
                priority = st.selectbox(
                    "Priority",
                    ["High", "Medium", "Low"]
                )
            
            submit_task = st.form_submit_button("Add Task")
            
            if submit_task and task_title:
                if 'user_id' not in st.session_state:
                    st.warning("Please set up your profile first on the Home page.")
                else:
                    # Combine date and time
                    due_datetime = pd.Timestamp.combine(due_date, due_time)
                    
                    # Save task to database
                    conn = init_connection()
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO tasks (student_id, title, description, due_date, priority, status)
                        VALUES (%s, %s, %s, %s, %s, 'pending')
                    """, (st.session_state['user_id'], task_title, task_description, due_datetime, priority))
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    st.success(f"Task '{task_title}' added successfully!")
    
    with col2:
        st.subheader("Quick Statistics")
        
        if 'user_id' in st.session_state:
            conn = init_connection()
            cursor = conn.cursor()
            
            # Get task counts by status
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM tasks 
                WHERE student_id = %s
                GROUP BY status
            """, (st.session_state['user_id'],))
            
            status_counts = dict(cursor.fetchall())
            
            # Get task counts by priority
            cursor.execute("""
                SELECT priority, COUNT(*) 
                FROM tasks 
                WHERE student_id = %s
                GROUP BY priority
            """, (st.session_state['user_id'],))
            
            priority_counts = dict(cursor.fetchall())
            
            cursor.close()
            conn.close()
            
            # Display counts
            pending = status_counts.get('pending', 0)
            completed = status_counts.get('completed', 0)
            
            st.metric("Pending Tasks", pending)
            st.metric("Completed Tasks", completed)
            
            # Display by priority if available
            if priority_counts:
                st.markdown("---")
                st.caption("Tasks by Priority")
                for priority, count in priority_counts.items():
                    st.metric(f"{priority} Priority", count)
        else:
            st.info("Set up your profile to see statistics.")
    
    # Show task list
    st.markdown("---")
    st.subheader("Your Tasks")
    
    if 'user_id' in st.session_state:
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, description, due_date, priority, status
            FROM tasks
            WHERE student_id = %s
            ORDER BY due_date ASC
        """, (st.session_state['user_id'],))
        
        tasks = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if tasks:
            tasks_df = pd.DataFrame(
                tasks, 
                columns=["ID", "Title", "Description", "Due Date", "Priority", "Status"]
            )
            
            tasks_df["Due Date"] = pd.to_datetime(tasks_df["Due Date"]).dt.strftime("%Y-%m-%d %H:%M")
            
            # Display tabs for different task views
            tab1, tab2, tab3 = st.tabs(["All Tasks", "Pending Tasks", "Completed Tasks"])
            
            with tab1:
                display_tasks(tasks_df, tab_id="all")
            
            with tab2:
                display_tasks(tasks_df[tasks_df["Status"] == "pending"], tab_id="pending")
            
            with tab3:
                display_tasks(tasks_df[tasks_df["Status"] == "completed"], tab_id="completed")
        else:
            st.info("You haven't added any tasks yet.")
    else:
        st.warning("Please set up your profile first on the Home page.")

def display_tasks(tasks_df, tab_id="all"):
    """
    Display a list of tasks with action buttons
    
    Args:
        tasks_df: DataFrame of tasks to display
        tab_id: Identifier for the tab (all, pending, completed) to ensure unique keys
    """
    if not tasks_df.empty:
        for index, row in tasks_df.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 2, 1, 1, 1])
            
            with col1:
                st.write(f"**{row['Title']}**")
                if row['Description']:
                    st.caption(row['Description'][:50] + "..." if len(row['Description']) > 50 else row['Description'])
            
            with col2:
                st.write(f"Due: {row['Due Date']}")
            
            with col3:
                priority_color = {
                    "High": "🔴",
                    "Medium": "🟠",
                    "Low": "🟢"
                }
                st.write(f"{priority_color.get(row['Priority'], '⚪')} {row['Priority']}")
            
            with col4:
                status = "✓" if row['Status'] == "completed" else "🕒"
                st.write(f"{status} {row['Status'].capitalize()}")
            
            with col5:
                if row['Status'] == "pending":
                    if st.button("Complete", key=f"complete_task_{tab_id}_{row['ID']}"):
                        conn = init_connection()
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            UPDATE tasks
                            SET status = 'completed'
                            WHERE id = %s AND student_id = %s
                        """, (row['ID'], st.session_state['user_id']))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        st.rerun()
            
            with col6:
                if st.button("Delete", key=f"delete_task_{tab_id}_{row['ID']}"):
                    st.session_state['delete_task_id'] = row['ID']
                    st.session_state['delete_task_confirmation'] = False
        
        # Show delete confirmation
        if 'delete_task_id' in st.session_state and 'delete_task_confirmation' in st.session_state and not st.session_state['delete_task_confirmation']:
            task_id = st.session_state['delete_task_id']
            st.warning(f"Are you sure you want to delete this task? This action cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Delete", key=f"confirm_delete_task_{tab_id}_{task_id}"):
                    # Import the planner agent
                    planner = get_planner()
                    
                    if planner and planner.delete_task(task_id, st.session_state['user_id']):
                        st.success("Task deleted successfully!")
                        st.session_state['delete_task_confirmation'] = True
                        
                        # Refresh the page
                        st.rerun()
                    else:
                        st.error("Failed to delete task.")
            with col2:
                if st.button("Cancel", key=f"cancel_delete_task_{tab_id}_{task_id}"):
                    # Clear deletion state
                    del st.session_state['delete_task_id']
                    del st.session_state['delete_task_confirmation']
                    st.rerun()
    else:
        st.info("No tasks to display in this category.")

def advisor_page():
    st.title("🧠 Advisor")
    
    st.markdown("""
    The Advisor agent provides personalized learning and time management advice based on your 
    student profile, academic data, and course syllabi.
    """)
    
    if 'user_id' not in st.session_state:
        st.warning("Please set up your profile first on the Home page.")
        return
    
    # Get API key from .env or session state
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key":
        if "api_key" in st.session_state:
            api_key = st.session_state.api_key
        else:
            api_key = st.text_input("Enter Groq API Key (required for AI advice):", type="password")
            if api_key:
                st.session_state.api_key = api_key
    
    # Get user profile
    conn = init_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT learning_style, study_hours
        FROM students
        WHERE id = %s
    """, (st.session_state['user_id'],))
    
    profile = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not profile:
        st.warning("Profile data not found. Please update your profile on the Home page.")
        return
    
    learning_style, study_hours = profile
    
    # Display profile summary
    st.subheader("Your Learning Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"Learning Style: **{learning_style}**")
        st.info(f"Daily Study Hours: **{study_hours}**")
    
    with col2:
        learning_style_tips = {
            "Visual": "Use diagrams, charts, and color coding in your notes.",
            "Auditory": "Record lectures and listen to them repeatedly. Discuss topics aloud.",
            "Reading/Writing": "Take detailed notes and rewrite them to reinforce concepts.",
            "Kinesthetic": "Use hands-on activities, experiments, and real-world applications."
        }
        
        st.markdown(f"### Tips for {learning_style} Learners")
        st.markdown(learning_style_tips.get(learning_style, "Customize your study approach."))
    
    # Student Progress Dashboard
    st.markdown("---")
    st.subheader("📊 Academic Progress Dashboard")
    
    # Get user's stats
    conn = init_connection()
    cursor = conn.cursor()
    
    # Notes statistics
    cursor.execute("""
        SELECT COUNT(id), COUNT(DISTINCT subject)
        FROM notes
        WHERE student_id = %s
    """, (st.session_state['user_id'],))
    
    notes_stats = cursor.fetchone()
    total_notes = notes_stats[0] if notes_stats else 0
    unique_subjects = notes_stats[1] if notes_stats else 0
    
    # Task statistics
    cursor.execute("""
        SELECT status, COUNT(id)
        FROM tasks
        WHERE student_id = %s
        GROUP BY status
    """, (st.session_state['user_id'],))
    
    task_stats = dict(cursor.fetchall())
    pending_tasks = task_stats.get('pending', 0)
    completed_tasks = task_stats.get('completed', 0)
    
    # Recent activity - includes notes and chat interactions
    cursor.execute("""
        SELECT 'note' as type, title, created_at 
        FROM notes 
        WHERE student_id = %s
        UNION ALL
        SELECT 'chat' as type, title, created_at
        FROM knowledge_base
        WHERE metadata->>'type' = 'chat_interaction' AND metadata->>'student_id' = %s
        ORDER BY created_at DESC
        LIMIT 5
    """, (st.session_state['user_id'], str(st.session_state['user_id'])))
    
    recent_activity = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    # Display the stats in a dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Notes", total_notes)
    with col2:
        st.metric("Subjects Covered", unique_subjects)
    with col3:
        st.metric("Pending Tasks", pending_tasks)
    with col4:
        st.metric("Completed Tasks", completed_tasks)
        
    # Display recent activity
    st.subheader("Recent Activity")
    if recent_activity:
        for activity in recent_activity:
            activity_type, title, date = activity
            activity_icon = "📝" if activity_type == "note" else "💬"
            st.write(f"{activity_icon} **{title}** - {date.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("No recent activity found. Start creating notes or tasks!")
    
    # Syllabus Upload Section
    st.markdown("---")
    st.subheader("📚 Course Syllabus Analysis")
    
    # Create tabs for Syllabus Upload and Saved Syllabi
    syllabus_tab1, syllabus_tab2 = st.tabs(["Upload Syllabus", "Saved Syllabi"])
    
    with syllabus_tab1:
        st.markdown("""
        Upload your course syllabus to get personalized advice tailored to your specific courses.
        This helps the advisor understand your course requirements, deadlines, and topics.
        """)
        
        uploaded_syllabus = st.file_uploader("Upload Course Syllabus (PDF)", type=["pdf"])
        
        if uploaded_syllabus is not None:
            # Form for syllabus metadata
            with st.form("syllabus_form"):
                course_name = st.text_input("Course Name", placeholder="e.g., Introduction to Computer Science")
                course_code = st.text_input("Course Code", placeholder="e.g., CS101")
                semester = st.text_input("Semester", placeholder="e.g., Fall 2025")
                
                save_syllabus = st.form_submit_button("Save Syllabus")
                
                if save_syllabus and course_name and course_code:
                    try:
                        # Save uploaded file to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_syllabus.getvalue())
                            temp_path = temp_file.name
                        
                        # Load PDF using LangChain
                        loader = PyPDFLoader(temp_path)
                        documents = loader.load()
                        
                        # Extract text content
                        syllabus_content = "\n".join([doc.page_content for doc in documents])
                        
                        # Clean up the temporary file
                        os.unlink(temp_path)
                        
                        # Save to database
                        conn = init_connection()
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            INSERT INTO knowledge_base (title, content, metadata)
                            VALUES (%s, %s, %s) 
                            RETURNING id
                        """, (
                            f"Syllabus: {course_name} ({course_code})",
                            syllabus_content,
                            json.dumps({
                                "type": "syllabus",
                                "course_name": course_name,
                                "course_code": course_code,
                                "semester": semester,
                                "student_id": st.session_state['user_id']
                            })
                        ))
                        
                        syllabus_id = cursor.fetchone()[0]
                        conn.commit()
                        cursor.close()
                        conn.close()
                        
                        st.success(f"Syllabus for {course_name} ({course_code}) saved successfully!")
                        
                        # Set the current syllabus for advice
                        st.session_state['current_syllabus_id'] = syllabus_id
                        st.session_state['current_syllabus_name'] = f"{course_name} ({course_code})"
                        
                    except Exception as e:
                        st.error(f"Error processing syllabus: {str(e)}")
    
    with syllabus_tab2:
        # Display saved syllabi
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, created_at, metadata
            FROM knowledge_base
            WHERE 
                metadata->>'type' = 'syllabus'
                AND metadata->>'student_id' = %s
            ORDER BY created_at DESC
        """, (str(st.session_state['user_id']),))
        
        syllabi = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if syllabi:
            st.markdown("### Your Saved Syllabi")
            
            for syllabus in syllabi:
                syllabus_id, title, created_at, metadata = syllabus
                # Handle metadata - check if it's already a dictionary
                if isinstance(metadata, dict):
                    metadata_dict = metadata
                else:
                    metadata_dict = json.loads(metadata)
                
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{title}**")
                    st.caption(f"Semester: {metadata_dict.get('semester', 'N/A')}")
                
                with col2:
                    created_date = pd.to_datetime(created_at).strftime("%Y-%m-%d")
                    st.write(f"Added: {created_date}")
                
                with col3:
                    if st.button("Select", key=f"select_syllabus_{syllabus_id}"):
                        st.session_state['current_syllabus_id'] = syllabus_id
                        st.session_state['current_syllabus_name'] = title
                        st.success(f"Selected {title} for advice")
        else:
            st.info("You haven't uploaded any syllabi yet. Upload syllabi to get course-specific advice.")
    
    # AI Advisor Section
    st.markdown("---")
    st.subheader("Ask the Advisor")
    
    # Display current selected syllabus if any
    if 'current_syllabus_id' in st.session_state:
        st.info(f"Current syllabus: **{st.session_state.get('current_syllabus_name', 'None')}**")
    
    user_question = st.text_area("What would you like advice on?", 
                              placeholder="e.g., How can I prepare for my upcoming exam? OR What topics should I focus on for CS101?")
    
    if st.button("Get Comprehensive Advice") and user_question:
        if not api_key:
            st.warning("Please enter your Groq API key to enable AI advice.")
        else:
            # Gather comprehensive student data for context
            conn = init_connection()
            cursor = conn.cursor()
            
            # Get task count by status
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM tasks 
                WHERE student_id = %s
                GROUP BY status
            """, (st.session_state['user_id'],))
            
            task_stats = dict(cursor.fetchall())
            
            # Get note count by subject
            cursor.execute("""
                SELECT subject, COUNT(*) 
                FROM notes 
                WHERE student_id = %s AND subject != ''
                GROUP BY subject
            """, (st.session_state['user_id'],))
            
            subject_stats = dict(cursor.fetchall())
            
            # Get upcoming deadlines
            cursor.execute("""
                SELECT title, due_date, priority
                FROM tasks
                WHERE student_id = %s AND status = 'pending' AND due_date >= NOW()
                ORDER BY due_date ASC
                LIMIT 5
            """, (st.session_state['user_id'],))
            
            upcoming_tasks = cursor.fetchall()
            
            # Get recent notes
            cursor.execute("""
                SELECT title, subject, created_at
                FROM notes
                WHERE student_id = %s
                ORDER BY created_at DESC
                LIMIT 5
            """, (st.session_state['user_id'],))
            
            recent_notes = cursor.fetchall()
            
            # Get current syllabus content if selected
            syllabus_content = ""
            if 'current_syllabus_id' in st.session_state:
                cursor.execute("""
                    SELECT content, metadata
                    FROM knowledge_base
                    WHERE id = %s
                """, (st.session_state['current_syllabus_id'],))
                
                syllabus_result = cursor.fetchone()
                if syllabus_result:
                    syllabus_content = syllabus_result[0]
                    syllabus_metadata_raw = syllabus_result[1]
                    
                    # Handle metadata - check if it's already a dictionary
                    if isinstance(syllabus_metadata_raw, dict):
                        syllabus_metadata = syllabus_metadata_raw
                    else:
                        syllabus_metadata = json.loads(syllabus_metadata_raw)
            
            # Get chat interactions for learning patterns
            cursor.execute("""
                SELECT content
                FROM knowledge_base
                WHERE metadata->>'type' = 'chat_interaction' 
                AND metadata->>'student_id' = %s
                ORDER BY created_at DESC
                LIMIT 10
            """, (str(st.session_state['user_id']),))
            
            chat_interactions = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            with st.spinner("Generating comprehensive academic advice..."):
                try:
                    # Initialize LLM
                    llm = GroqLLaMa(api_key)
                    
                    # Create a comprehensive context-rich prompt
                    prompt = f"""
                    As an academic advisor, provide comprehensive personalized advice for a student with the following profile:
                    
                    STUDENT PROFILE:
                    Learning Style: {learning_style}
                    Daily Study Hours: {study_hours}
                    
                    ACADEMIC STATS:
                    - Total Notes: {total_notes}
                    - Subjects Covered: {unique_subjects}
                    - Completed tasks: {task_stats.get('completed', 0)}
                    - Pending tasks: {task_stats.get('pending', 0)}
                    
                    SUBJECT DISTRIBUTION:
                    {', '.join([f"{subject}: {count}" for subject, count in subject_stats.items()])}
                    
                    UPCOMING DEADLINES:
                    """
                    
                    # Add upcoming tasks if available
                    if upcoming_tasks:
                        for task in upcoming_tasks:
                            task_title, due_date, priority = task
                            prompt += f"- {task_title} (Due: {due_date.strftime('%Y-%m-%d')}, Priority: {priority})\n"
                    else:
                        prompt += "No upcoming deadlines.\n"
                    
                    prompt += "\nRECENT NOTES:\n"
                    
                    # Add recent notes if available
                    if recent_notes:
                        for note in recent_notes:
                            note_title, note_subject, created_at = note
                            prompt += f"- {note_title} ({note_subject}, Created: {created_at.strftime('%Y-%m-%d')})\n"
                    else:
                        prompt += "No recent notes.\n"
                    
                    # Add chat interactions for learning patterns if available
                    if chat_interactions:
                        prompt += "\nRECENT LEARNING INTERACTIONS:\n"
                        for i, chat in enumerate(chat_interactions[:5]):  # Limit to 5 to manage token count
                            # Handle chat data - check if it's already a dictionary
                            chat_content = chat[0]
                            if isinstance(chat_content, dict):
                                chat_data = chat_content
                            else:
                                chat_data = json.loads(chat_content)
                            prompt += f"- Question: {chat_data.get('question', 'N/A')}\n"
                    
                    # Add syllabus context if available
                    if syllabus_content:
                        # Truncate syllabus content if too long (to avoid token limits)
                        max_syllabus_length = 2000
                        truncated_syllabus = syllabus_content[:max_syllabus_length]
                        if len(syllabus_content) > max_syllabus_length:
                            truncated_syllabus += "... [content truncated]"
                        
                        prompt += f"""
                        
                        COURSE SYLLABUS INFORMATION:
                        
                        Course: {syllabus_metadata.get('course_name')} ({syllabus_metadata.get('course_code')})
                        Semester: {syllabus_metadata.get('semester')}
                        
                        Syllabus Content:
                        {truncated_syllabus}
                        """
                    
                    prompt += f"""
                    
                    The student is asking: {user_question}
                    
                    PROVIDE COMPREHENSIVE ADVICE THAT:
                    1. Analyzes their current academic situation holistically
                    2. Takes into account their learning style, subject distribution, and time commitments
                    3. Provides specific strategies tailored to their situation
                    4. Includes actionable steps they can take immediately
                    5. Addresses any upcoming deadlines or course requirements
                    6. Recommends specific study techniques based on their learning style
                    7. Suggests how to optimize their study time based on their available hours
                    
                    If the student has provided a course syllabus, analyze it to provide advice specific to:
                    - Important topics and concepts in the course
                    - Upcoming assignments or exams mentioned in the syllabus
                    - Recommended study techniques for the specific course material
                    - How to budget time effectively for this course
                    
                    Your advice should be comprehensive, specific, and actionable.
                    """
                    
                    messages = [{"role": "user", "content": prompt}]
                    
                    # Generate advice
                    advice = llm.generate(messages)
                    
                    # Display advice
                    st.markdown("### 💡 Comprehensive Academic Advice")
                    st.markdown(advice)
                    
                    # Save the advice to knowledge base for future reference
                    try:
                        conn = init_connection()
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            INSERT INTO knowledge_base (title, content, metadata)
                            VALUES (%s, %s, %s)
                        """, (
                            f"Advice: {user_question[:50]}{'...' if len(user_question) > 50 else ''}",
                            advice,
                            json.dumps({
                                "type": "advisor_advice",
                                "question": user_question,
                                "student_id": st.session_state['user_id'],
                                "timestamp": datetime.now().isoformat(),
                                "syllabus_id": st.session_state.get('current_syllabus_id')
                            })
                        ))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                    except Exception as db_error:
                        print(f"Error storing advice: {str(db_error)}")
                    
                except Exception as e:
                    st.error(f"Error generating advice: {str(e)}")
                    st.warning("Please check your API key and try again.")
    
    # Show advice history
    st.markdown("---")
    st.subheader("Previous Advice")
    
    conn = init_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT title, content, created_at
        FROM knowledge_base
        WHERE 
            metadata->>'type' = 'advisor_advice'
            AND metadata->>'student_id' = %s
        ORDER BY created_at DESC
        LIMIT 5
    """, (str(st.session_state['user_id']),))
    
    previous_advice = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if previous_advice:
        for i, advice in enumerate(previous_advice):
            title, content, date = advice
            with st.expander(f"{title} - {date.strftime('%Y-%m-%d %H:%M')}"):
                st.markdown(content)
    else:
        st.info("No previous advice found. Ask the advisor for advice to get started.")

def pdf_chat_page():
    st.title("💬 PDF & Notes Chat")
    
    st.markdown("""
    Chat with your study materials. Ask questions about your saved notes or upload a PDF to get instant answers.
    This feature uses RAG (Retrieval Augmented Generation) technology to provide precise answers from your documents.
    """)
    
    if 'user_id' not in st.session_state:
        st.warning("Please set up your profile first on the Home page.")
        return
    
    # Get API key from .env or session state
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key":
        if "api_key" in st.session_state:
            api_key = st.session_state.api_key
        else:
            api_key = st.text_input("Enter Groq API Key (required for chat):", type="password")
            if api_key:
                st.session_state.api_key = api_key
    
    # Create tabs for different content sources
    source_tab1, source_tab2, source_tab3 = st.tabs(["📄 PDF Upload", "📝 Saved Notes", "🔍 Multi-Source Search"])
    
    # For tracking the current content source
    if 'chat_content' not in st.session_state:
        st.session_state.chat_content = None
    
    if 'chat_content_name' not in st.session_state:
        st.session_state.chat_content_name = None
        
    # For storing RAG pipelines to avoid rebuilding them
    if 'rag_pipelines' not in st.session_state:
        st.session_state.rag_pipelines = {}
        
    # For storing multi-source selections
    if 'multi_sources' not in st.session_state:
        st.session_state.multi_sources = []
    
    with source_tab1:
        st.subheader("Upload a PDF to Chat")
        
        uploaded_pdf = st.file_uploader("Upload PDF document:", type=["pdf"])
        
        if uploaded_pdf is not None:
            # Extract PDF content
            try:
                # Display processing message
                with st.spinner("Processing PDF and building knowledge base..."):
                    # Save uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_pdf.getvalue())
                        temp_path = temp_file.name
                    
                    # Load PDF using LangChain - this creates Document objects with metadata
                    loader = PyPDFLoader(temp_path)
                    documents = loader.load()
                    
                    # Clean up the temporary file
                    os.unlink(temp_path)
                    
                    # Extract text content for preview
                    pdf_content = "\n".join([doc.page_content for doc in documents])
                    
                    # Store in session state
                    pdf_name = uploaded_pdf.name
                    st.session_state.chat_content = documents  # Store document objects
                    st.session_state.chat_content_name = pdf_name
                    
                    # Clear any existing RAG pipeline for this document to rebuild it
                    pipeline_key = f"rag_{hash(pdf_name)}"
                    if pipeline_key in st.session_state.rag_pipelines:
                        del st.session_state.rag_pipelines[pipeline_key]
                    
                    st.success(f"Loaded and indexed PDF: {pdf_name}")
                    
                    # Display a preview of the content
                    with st.expander("PDF Content Preview"):
                        st.text(pdf_content[:500] + "..." if len(pdf_content) > 500 else pdf_content)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    with source_tab2:
        st.subheader("Chat with Your Notes")
        
        # Get user's notes
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, subject, created_at 
            FROM notes 
            WHERE student_id = %s
            ORDER BY created_at DESC
        """, (st.session_state['user_id'],))
        
        notes = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if notes:
            notes_options = ["Select a note..."] + [f"{note[1]} ({note[2]})" for note in notes]
            selected_note_index = st.selectbox("Choose a note to chat with:", 
                                              options=range(len(notes_options)),
                                              format_func=lambda x: notes_options[x])
            
            if selected_note_index > 0:
                # Get the actual note
                note_id = notes[selected_note_index - 1][0]
                
                conn = init_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT title, content
                    FROM notes
                    WHERE id = %s AND student_id = %s
                """, (note_id, st.session_state['user_id']))
                
                note = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if note:
                    # Display processing message
                    with st.spinner("Processing note and building knowledge base..."):
                        note_title, note_content = note
                        
                        # Store in session state
                        st.session_state.chat_content = note_content
                        st.session_state.chat_content_name = note_title
                        
                        # Clear any existing RAG pipeline for this note to rebuild it
                        pipeline_key = f"rag_{hash(note_title)}"
                        if pipeline_key in st.session_state.rag_pipelines:
                            del st.session_state.rag_pipelines[pipeline_key]
                        
                        st.success(f"Loaded and indexed note: {note_title}")
                        
                        # Display a preview of the content
                        with st.expander("Note Content Preview"):
                            st.text(note_content[:500] + "..." if len(note_content) > 500 else note_content)
        else:
            st.info("You haven't saved any notes yet. Create notes using the Notewriter feature.")
    
    with source_tab3:
        st.subheader("Multi-Source Knowledge Base")
        st.markdown("""
        Select multiple notes and PDFs to build a comprehensive knowledge base. 
        This allows you to ask questions across all your study materials at once.
        """)
        
        # Get user's notes for selection
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, subject
            FROM notes 
            WHERE student_id = %s
            ORDER BY subject, title
        """, (st.session_state['user_id'],))
        
        notes = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Get syllabi from knowledge base
        conn = init_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, metadata
            FROM knowledge_base
            WHERE 
                metadata->>'type' = 'syllabus'
                AND metadata->>'student_id' = %s
            ORDER BY title
        """, (str(st.session_state['user_id']),))
        
        syllabi = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Display available sources with checkboxes
        st.subheader("Select Sources")
        
        with st.form("multi_source_form"):
            st.markdown("### Notes")
            selected_notes = []
            if notes:
                for note_id, title, subject in notes:
                    if st.checkbox(f"{title} ({subject})", key=f"note_{note_id}"):
                        selected_notes.append(note_id)
            else:
                st.info("No notes available. Create notes using the Notewriter feature.")
                
            st.markdown("### Syllabi")
            selected_syllabi = []
            if syllabi:
                for syllabus_id, title, _ in syllabi:
                    if st.checkbox(f"{title}", key=f"syllabus_{syllabus_id}"):
                        selected_syllabi.append(syllabus_id)
            else:
                st.info("No syllabi available. Upload syllabi in the Advisor section.")
                
            build_kb_button = st.form_submit_button("Build Multi-Source Knowledge Base")
            
            if build_kb_button:
                if not selected_notes and not selected_syllabi:
                    st.warning("Please select at least one source to build the knowledge base.")
                else:
                    with st.spinner("Building multi-source knowledge base..."):
                        # Get content for all selected sources
                        sources_list = []
                        
                        # Get selected notes
                        if selected_notes:
                            conn = init_connection()
                            cursor = conn.cursor()
                            
                            placeholders = ', '.join(['%s'] * len(selected_notes))
                            query = f"""
                                SELECT id, title, content
                                FROM notes
                                WHERE id IN ({placeholders}) AND student_id = %s
                            """
                            
                            cursor.execute(query, selected_notes + [st.session_state['user_id']])
                            note_data = cursor.fetchall()
                            cursor.close()
                            conn.close()
                            
                            for note_id, title, content in note_data:
                                sources_list.append((content, f"Note: {title}"))
                        
                        # Get selected syllabi
                        if selected_syllabi:
                            conn = init_connection()
                            cursor = conn.cursor()
                            
                            placeholders = ', '.join(['%s'] * len(selected_syllabi))
                            query = f"""
                                SELECT id, title, content
                                FROM knowledge_base
                                WHERE id IN ({placeholders})
                            """
                            
                            cursor.execute(query, selected_syllabi)
                            syllabus_data = cursor.fetchall()
                            cursor.close()
                            conn.close()
                            
                            for syllabus_id, title, content in syllabus_data:
                                sources_list.append((content, f"Syllabus: {title}"))
                        
                        # Combine all sources
                        all_documents = combine_knowledge_sources(sources_list)
                        
                        # Store for chat interface
                        st.session_state.chat_content = all_documents
                        st.session_state.chat_content_name = f"Multi-Source ({len(sources_list)} documents)"
                        
                        # Clear any existing RAG pipeline
                        pipeline_key = f"rag_multi_source_{hash(tuple(selected_notes + selected_syllabi))}"
                        if pipeline_key in st.session_state.rag_pipelines:
                            del st.session_state.rag_pipelines[pipeline_key]
                            
                        # Pre-initialize pipeline if API key is available
                        if api_key:
                            with st.spinner("Building knowledge retrieval system..."):
                                llm = GroqLLaMa(api_key)
                                qa_chain = create_rag_pipeline(
                                    all_documents,
                                    f"Multi-Source ({len(sources_list)} documents)",
                                    llm
                                )
                                st.session_state.rag_pipelines[pipeline_key] = qa_chain
                        
                        st.success(f"Built knowledge base with {len(sources_list)} sources. You can now chat with all selected materials!")
        
    # Chat interface
    st.markdown("---")
    
    if st.session_state.chat_content:
        st.subheader(f"Chat with: {st.session_state.chat_content_name}")
        
        # Initialize chat history for this content if not exists
        chat_history_key = f"chat_history_{hash(st.session_state.chat_content_name)}"
        if chat_history_key not in st.session_state:
            st.session_state[chat_history_key] = []
        
        # Display chat history
        for message in st.session_state[chat_history_key]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    st.caption("**Sources:**")
                    for source in message["sources"]:
                        st.caption(f"- {source}")
        
        # Chat input
        user_question = st.chat_input("Ask a question about this content...")
        
        if user_question and api_key:
            # Add user message to chat history
            st.session_state[chat_history_key].append({
                "role": "user", 
                "content": user_question
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                try:
                    # Initialize LLM
                    llm = GroqLLaMa(api_key)
                    
                    # Get or create RAG pipeline for this content
                    pipeline_key = f"rag_{hash(st.session_state.chat_content_name)}"
                    
                    if pipeline_key not in st.session_state.rag_pipelines:
                        with st.spinner("Building knowledge retrieval system..."):
                            # Create a new RAG pipeline
                            qa_chain = create_rag_pipeline(
                                st.session_state.chat_content, 
                                st.session_state.chat_content_name,
                                llm
                            )
                            # Cache the pipeline for future use
                            st.session_state.rag_pipelines[pipeline_key] = qa_chain
                    else:
                        # Use existing pipeline
                        qa_chain = st.session_state.rag_pipelines[pipeline_key]
                    
                    # Execute the query against the RAG pipeline
                    with st.spinner("Searching document and generating answer..."):
                        result = qa_chain({"query": user_question})
                        
                        # Extract the response and source documents
                        response = result["result"]
                        source_docs = result["source_documents"]
                        
                        # Extract source information
                        sources = []
                        for doc in source_docs:
                            source_info = []
                            if "source" in doc.metadata:
                                source_info.append(doc.metadata["source"])
                            if "chunk_id" in doc.metadata:
                                source_info.append(f"Chunk {doc.metadata['chunk_id']}")
                            sources.append(" - ".join(source_info))
                    
                    # Add assistant response to chat history with sources
                    st.session_state[chat_history_key].append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                    # Store interaction in database for future reference
                    try:
                        conn = init_connection()
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            INSERT INTO knowledge_base (title, content, metadata)
                            VALUES (%s, %s, %s)
                        """, (
                            f"Chat: {st.session_state.chat_content_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            json.dumps({
                                "question": user_question,
                                "answer": response,
                                "sources": sources
                            }),
                            json.dumps({
                                "type": "chat_interaction",
                                "content_name": st.session_state.chat_content_name,
                                "student_id": st.session_state['user_id'],
                                "timestamp": datetime.now().isoformat()
                            })
                        ))
                        
                        conn.commit()
                        cursor.close()
                        conn.close()
                    except Exception as db_error:
                        print(f"Error storing chat interaction: {str(db_error)}")
                    
                    # Update the placeholder with the response
                    # First, clear the placeholder to remove the "Thinking..." message
                    message_placeholder.empty()
                    
                    # Then create content within the assistant message block
                    st.markdown(response)
                    
                    # Display sources if available
                    if sources:
                        st.caption("**Sources:**")
                        for source in sources:
                            st.caption(f"- {source}")
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    # Clear the placeholder and display the error
                    message_placeholder.empty()
                    st.markdown(error_msg)
                    
                    # Add error to chat history
                    st.session_state[chat_history_key].append({
                        "role": "assistant", 
                        "content": error_msg
                    })
        elif not api_key and user_question:
            st.warning("Please enter your Groq API key to enable chat.")
    else:
        st.info("Upload a PDF, select a saved note, or build a multi-source knowledge base to start chatting.")
        
    # Add option to clear chat history
    if st.session_state.chat_content and chat_history_key in st.session_state and st.session_state[chat_history_key]:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Clear Chat History"):
                st.session_state[chat_history_key] = []
                st.rerun()
        with col2:
            if st.button("Clear Knowledge Base (Reset Index)"):
                # Clear the RAG pipeline to force rebuilding it
                pipeline_key = f"rag_{hash(st.session_state.chat_content_name)}"
                if pipeline_key in st.session_state.rag_pipelines:
                    del st.session_state.rag_pipelines[pipeline_key]
                st.success("Knowledge base cleared and will be rebuilt on your next question.")
                st.rerun()

def create_rag_pipeline(content, content_name, llm):
    """
    Create a RAG (Retrieval Augmented Generation) pipeline for document Q&A.
    
    Args:
        content (str): The document content to process
        content_name (str): Name of the document for metadata
        llm: The language model to use for generation
        
    Returns:
        A retrieval QA chain for answering questions about the document
    """
    # Create documents for ingestion
    if isinstance(content, str):
        # Create a Document object with metadata
        document = Document(
            page_content=content,
            metadata={"source": content_name}
        )
        documents = [document]
    else:
        # If it's already a list of Documents
        documents = content
    
    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    
    # Add chunk IDs to document metadata
    for i, split in enumerate(splits):
        split.metadata["chunk_id"] = i
    
    # Initialize embeddings - use a lightweight model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store for similarity search
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 chunks for each query
    )
    
    # Create prompt template
    prompt_template = """
    You are an academic assistant helping a student understand their study materials.
    Use only the following retrieved context to answer the question. If you don't know the 
    answer or if it's not in the context, say that you don't have that information in the material.
    
    Be accurate, helpful, clear, and concise.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    # Create prompt
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # If we have a custom GroqLLaMa instance, use its chat_model attribute
    if hasattr(llm, 'chat_model'):
        llm_for_chain = llm.chat_model
    else:
        llm_for_chain = llm
    
    # Create chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_for_chain,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def combine_knowledge_sources(sources_list):
    """
    Combine multiple knowledge sources (notes, PDFs) into a single collection
    for unified querying.
    
    Args:
        sources_list: List of tuples (content, name) to combine
        
    Returns:
        A RAG pipeline that can query across all sources
    """
    # Create a list of documents
    all_documents = []
    
    for content, name in sources_list:
        # Handle different content types
        if isinstance(content, list) and all(isinstance(item, Document) for item in content):
            # It's already a list of Document objects
            # Add source name to metadata
            for doc in content:
                doc.metadata["source"] = name
            all_documents.extend(content)
        else:
            # It's raw text, create a Document
            document = Document(
                page_content=content,
                metadata={"source": name}
            )
            all_documents.append(document)
    
    return all_documents

if __name__ == "__main__":
    # Check if navigation is set in session state
    if 'navigation' in st.session_state:
        # Set sidebar selection based on navigation
        pass
    main() 