"""
Advisor Agent - Personalized learning and time management advice

This agent is responsible for:
1. Providing personalized study recommendations
2. Analyzing learning patterns and suggesting improvements
3. Offering time management strategies
4. Generating tailored academic advice
"""

from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
import json
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Database connection settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "academic_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

class Advisor:
    """Advisor Agent for personalized learning recommendations"""
    
    def __init__(self):
        """Initialize the advisor agent"""
        self.conn = self._init_connection()
    
    def _init_connection(self):
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
        except psycopg2.OperationalError as e:
            print(f"Database connection error: {e}")
            return None
    
    def get_student_profile(self, student_id: int) -> Optional[Dict[str, Any]]:
        """Get a student's profile"""
        if not self.conn:
            return None
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, name, email, learning_style, study_hours, created_at
            FROM students
            WHERE id = %s
        """, (student_id,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if not row:
            return None
        
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "learning_style": row[3],
            "study_hours": row[4],
            "created_at": row[5]
        }
    
    def get_task_statistics(self, student_id: int) -> Dict[str, Any]:
        """Get statistics about a student's tasks"""
        if not self.conn:
            return {}
        
        cursor = self.conn.cursor()
        
        # Get total task counts by status
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM tasks 
            WHERE student_id = %s
            GROUP BY status
        """, (student_id,))
        
        status_counts = dict(cursor.fetchall())
        
        # Get task counts by priority
        cursor.execute("""
            SELECT priority, COUNT(*) 
            FROM tasks 
            WHERE student_id = %s
            GROUP BY priority
        """, (student_id,))
        
        priority_counts = dict(cursor.fetchall())
        
        # Get tasks completed in the last 14 days
        cursor.execute("""
            SELECT COUNT(*) 
            FROM tasks 
            WHERE student_id = %s 
              AND status = 'completed'
              AND updated_at >= NOW() - INTERVAL '14 days'
        """, (student_id,))
        
        completed_recent = cursor.fetchone()[0]
        
        # Get total tasks due in the last 14 days
        cursor.execute("""
            SELECT COUNT(*) 
            FROM tasks 
            WHERE student_id = %s 
              AND due_date >= NOW() - INTERVAL '14 days'
              AND due_date <= NOW()
        """, (student_id,))
        
        due_recent = cursor.fetchone()[0]
        
        # Calculate completion rate
        completion_rate = (completed_recent / due_recent * 100) if due_recent > 0 else 0
        
        # Get upcoming deadlines
        cursor.execute("""
            SELECT COUNT(*) 
            FROM tasks 
            WHERE student_id = %s 
              AND status != 'completed'
              AND due_date BETWEEN NOW() AND NOW() + INTERVAL '7 days'
        """, (student_id,))
        
        upcoming_week = cursor.fetchone()[0]
        
        cursor.close()
        
        return {
            "total_tasks": sum(status_counts.values()),
            "completed_tasks": status_counts.get("completed", 0),
            "pending_tasks": status_counts.get("pending", 0),
            "priority_distribution": priority_counts,
            "completion_rate_14d": round(completion_rate, 1),
            "upcoming_week_deadlines": upcoming_week
        }
    
    def get_learning_stats(self, student_id: int) -> Dict[str, Any]:
        """Get statistics about a student's notes and learning activities"""
        if not self.conn:
            return {}
        
        cursor = self.conn.cursor()
        
        # Get total notes count
        cursor.execute("""
            SELECT COUNT(*) 
            FROM notes 
            WHERE student_id = %s
        """, (student_id,))
        
        total_notes = cursor.fetchone()[0]
        
        # Get notes by subject
        cursor.execute("""
            SELECT subject, COUNT(*) 
            FROM notes 
            WHERE student_id = %s AND subject != ''
            GROUP BY subject
        """, (student_id,))
        
        subject_counts = dict(cursor.fetchall())
        
        # Get notes created in the last 30 days
        cursor.execute("""
            SELECT COUNT(*) 
            FROM notes 
            WHERE student_id = %s 
              AND created_at >= NOW() - INTERVAL '30 days'
        """, (student_id,))
        
        recent_notes = cursor.fetchone()[0]
        
        # Get average note length (as a measure of depth)
        cursor.execute("""
            SELECT AVG(LENGTH(content)) 
            FROM notes 
            WHERE student_id = %s
        """, (student_id,))
        
        avg_length_result = cursor.fetchone()[0]
        avg_note_length = int(avg_length_result) if avg_length_result else 0
        
        cursor.close()
        
        return {
            "total_notes": total_notes,
            "notes_by_subject": subject_counts,
            "notes_last_30d": recent_notes,
            "avg_note_length": avg_note_length
        }
    
    def generate_advice(self, student_id: int, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate personalized advice for a student
        
        This is a placeholder implementation. In a real system, this would:
        1. Use LLMs to analyze the student's profile, tasks, and notes
        2. Generate personalized advice based on the student's query
        3. Provide tailored recommendations for improving study habits
        """
        # Get student profile
        profile = self.get_student_profile(student_id)
        if not profile:
            return {"error": "Student profile not found"}
        
        # Get task statistics
        task_stats = self.get_task_statistics(student_id)
        
        # Get learning statistics
        learning_stats = self.get_learning_stats(student_id)
        
        # Generate advice based on available data
        learning_style = profile.get("learning_style", "")
        study_hours = profile.get("study_hours", 0)
        
        # Basic advice based on learning style
        style_advice = {
            "Visual": [
                "Use diagrams, charts, and visual aids in your study materials",
                "Color-code your notes to highlight different types of information",
                "Watch video tutorials and demonstrations when available"
            ],
            "Auditory": [
                "Record lectures and listen to them during review",
                "Read your notes aloud when studying",
                "Participate in study groups where you can discuss material verbally"
            ],
            "Reading/Writing": [
                "Take detailed notes during lectures and while reading",
                "Rewrite key concepts in your own words",
                "Use written summaries and lists to organize information"
            ],
            "Kinesthetic": [
                "Incorporate movement into your study sessions",
                "Use physical models or manipulatives when possible",
                "Take frequent short breaks for movement between study sessions"
            ]
        }.get(learning_style, ["Determine your preferred learning style to get more tailored advice"])
        
        # Time management advice based on task statistics
        time_advice = []
        completion_rate = task_stats.get("completion_rate_14d", 0)
        
        if completion_rate < 50:
            time_advice.append("Your recent task completion rate is below 50%. Consider setting more realistic deadlines or breaking tasks into smaller chunks.")
        elif completion_rate < 80:
            time_advice.append("Your task completion rate is good but could be improved. Try using time blocking to ensure you allocate sufficient time for each task.")
        else:
            time_advice.append("Your task completion rate is excellent! Continue with your current time management strategies.")
        
        if task_stats.get("upcoming_week_deadlines", 0) > 5:
            time_advice.append("You have many deadlines approaching this week. Consider prioritizing tasks by urgency and importance.")
        
        if study_hours < 2:
            time_advice.append("Your daily study hours are quite low. Try to increase your study time gradually to improve subject mastery.")
        elif study_hours > 8:
            time_advice.append("You're allocating significant time to studying. Ensure you're taking sufficient breaks and using active study techniques to maintain effectiveness.")
        
        # Study technique advice based on note statistics
        study_advice = []
        
        if learning_stats.get("total_notes", 0) < 5:
            study_advice.append("You have few saved notes. Consider taking more structured notes to enhance your learning and retention.")
        
        if learning_stats.get("avg_note_length", 0) < 500:
            study_advice.append("Your notes tend to be brief. Try to include more examples and explanations to improve understanding.")
        elif learning_stats.get("avg_note_length", 0) > 3000:
            study_advice.append("Your notes are quite detailed. Consider creating shorter summary notes for quick review before exams.")
        
        # Process specific query if provided
        query_response = ""
        if query:
            # This would use LLM processing in a real implementation
            if "time" in query.lower() or "schedule" in query.lower() or "planning" in query.lower():
                query_response = "Based on your time management query, I recommend: " + time_advice[0]
            elif "study" in query.lower() or "learn" in query.lower() or "remember" in query.lower():
                query_response = "For your study technique question, I suggest: " + study_advice[0] if study_advice else "Focus on active recall and spaced repetition to enhance your learning."
            elif "stress" in query.lower() or "overwhelm" in query.lower() or "anxiety" in query.lower():
                query_response = "For managing academic stress, try breaking large tasks into smaller steps, practicing mindfulness techniques, and ensuring you're taking regular breaks during study sessions."
            else:
                query_response = f"To best answer your question about '{query}', I would analyze your learning patterns and academic data in more detail."
        
        # Compile all advice
        return {
            "learning_style": learning_style,
            "style_recommendations": style_advice,
            "time_management_advice": time_advice,
            "study_technique_advice": study_advice,
            "query_response": query_response,
            "task_statistics": task_stats,
            "learning_statistics": learning_stats
        }
    
    def close_connection(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

# Helper function to get an advisor instance
def get_advisor() -> Advisor:
    """Get an advisor agent instance"""
    return Advisor() 