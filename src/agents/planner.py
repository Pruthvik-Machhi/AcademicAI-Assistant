"""
Planner Agent - Schedule optimization and time management

This agent is responsible for:
1. Calendar integration and management
2. Task prioritization and scheduling
3. Time allocation optimization
4. Deadline tracking and reminders
"""

from typing import Dict, List, Any, Optional
import datetime
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

# Load environment variables
load_dotenv()

# Database connection settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "academic_assistant")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

class Planner:
    """Planner Agent for schedule and time management"""
    
    def __init__(self):
        """Initialize the planner agent"""
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
    
    def get_tasks(self, student_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tasks for a specific student with optional status filter"""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        
        # Construct query based on whether status filter is provided
        if status:
            cursor.execute("""
                SELECT id, title, description, due_date, priority, status
                FROM tasks
                WHERE student_id = %s AND status = %s
                ORDER BY due_date ASC
            """, (student_id, status))
        else:
            cursor.execute("""
                SELECT id, title, description, due_date, priority, status
                FROM tasks
                WHERE student_id = %s
                ORDER BY due_date ASC
            """, (student_id,))
        
        # Fetch tasks and format as dictionary
        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "due_date": row[3],
                "priority": row[4],
                "status": row[5]
            })
        
        cursor.close()
        return tasks
    
    def add_task(self, student_id: int, task_data: Dict[str, Any]) -> Optional[int]:
        """Add a new task for a student"""
        if not self.conn:
            return None
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO tasks (
                    student_id, title, description, due_date, priority, status
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                student_id,
                task_data.get("title", "Untitled Task"),
                task_data.get("description", ""),
                task_data.get("due_date", datetime.now() + timedelta(days=7)),
                task_data.get("priority", "Medium"),
                task_data.get("status", "pending")
            ))
            
            task_id = cursor.fetchone()[0]
            self.conn.commit()
            return task_id
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding task: {e}")
            return None
        finally:
            cursor.close()
    
    def update_task(self, task_id: int, student_id: int, task_data: Dict[str, Any]) -> bool:
        """Update an existing task"""
        if not self.conn:
            return False
        
        cursor = self.conn.cursor()
        
        # Build dynamic update query based on provided data
        update_fields = []
        params = []
        
        if "title" in task_data:
            update_fields.append("title = %s")
            params.append(task_data["title"])
        
        if "description" in task_data:
            update_fields.append("description = %s")
            params.append(task_data["description"])
        
        if "due_date" in task_data:
            update_fields.append("due_date = %s")
            params.append(task_data["due_date"])
        
        if "priority" in task_data:
            update_fields.append("priority = %s")
            params.append(task_data["priority"])
        
        if "status" in task_data:
            update_fields.append("status = %s")
            params.append(task_data["status"])
        
        # If no fields to update, return early
        if not update_fields:
            return False
        
        # Add task_id and student_id to params
        params.extend([task_id, student_id])
        
        try:
            cursor.execute(f"""
                UPDATE tasks 
                SET {", ".join(update_fields)}
                WHERE id = %s AND student_id = %s
            """, params)
            
            affected_rows = cursor.rowcount
            self.conn.commit()
            return affected_rows > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating task: {e}")
            return False
        finally:
            cursor.close()
    
    def delete_task(self, task_id: int, student_id: int) -> bool:
        """Delete a task"""
        if not self.conn:
            return False
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM tasks
                WHERE id = %s AND student_id = %s
            """, (task_id, student_id))
            
            affected_rows = cursor.rowcount
            self.conn.commit()
            return affected_rows > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting task: {e}")
            return False
        finally:
            cursor.close()
    
    def get_overdue_tasks(self, student_id: int) -> List[Dict[str, Any]]:
        """Get overdue tasks for a student"""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, title, description, due_date, priority, status
            FROM tasks
            WHERE student_id = %s AND due_date < NOW() AND status != 'completed'
            ORDER BY due_date ASC
        """, (student_id,))
        
        # Fetch tasks and format as dictionary
        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "due_date": row[3],
                "priority": row[4],
                "status": row[5]
            })
        
        cursor.close()
        return tasks
    
    def get_upcoming_tasks(self, student_id: int, days: int = 7) -> List[Dict[str, Any]]:
        """Get upcoming tasks for a student within a specified number of days"""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, title, description, due_date, priority, status
            FROM tasks
            WHERE student_id = %s 
              AND due_date BETWEEN NOW() AND NOW() + INTERVAL %s DAY
              AND status != 'completed'
            ORDER BY due_date ASC
        """, (student_id, days))
        
        # Fetch tasks and format as dictionary
        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "due_date": row[3],
                "priority": row[4],
                "status": row[5]
            })
        
        cursor.close()
        return tasks
    
    def generate_optimized_schedule(self, student_id: int, learning_style: str, study_hours: int) -> Dict[str, Any]:
        """Generate an optimized study schedule based on tasks and preferences"""
        # This is a simplified implementation - a full version would use ML/LLM
        # to optimize schedule based on student patterns, learning style, etc.
        
        # Get upcoming tasks
        upcoming_tasks = self.get_upcoming_tasks(student_id)
        
        # Get overdue tasks
        overdue_tasks = self.get_overdue_tasks(student_id)
        
        # Create subject categories based on tasks
        subjects = {}
        all_tasks = overdue_tasks + upcoming_tasks
        
        for task in all_tasks:
            # Extract subject from task title (simplified approach)
            # In a real implementation, this might use a more sophisticated classification
            title_words = task["title"].lower().split()
            subject = next((word for word in title_words if word in ["math", "physics", "history", "english", "biology", "chemistry", "literature", "programming"]), "other")
            
            if subject not in subjects:
                subjects[subject] = {
                    "tasks": [],
                    "priority_score": 0,
                    "recommended_hours": 0
                }
            
            # Add task to subject and calculate priority score
            subjects[subject]["tasks"].append(task)
            
            # Calculate priority score based on due date and priority
            days_until_due = (task["due_date"] - datetime.now()).days if task["due_date"] else 7
            priority_multiplier = {"High": 3, "Medium": 2, "Low": 1}.get(task["priority"], 1)
            
            # Score formula: higher for more urgent and higher priority tasks
            task_score = priority_multiplier * (10 - min(days_until_due, 10)) if days_until_due > 0 else priority_multiplier * 10
            subjects[subject]["priority_score"] += task_score
        
        # Allocate study hours based on priority scores
        total_priority_score = sum(data["priority_score"] for data in subjects.values())
        
        if total_priority_score > 0:
            for subject, data in subjects.items():
                # Allocate hours proportionally to priority score
                data["recommended_hours"] = round((data["priority_score"] / total_priority_score) * study_hours, 1)
        else:
            # Equal distribution if no priority scores
            equal_hours = study_hours / max(len(subjects), 1)
            for data in subjects.values():
                data["recommended_hours"] = round(equal_hours, 1)
        
        # Adjust based on learning style
        if learning_style == "Visual":
            # Visual learners might benefit from more concentrated blocks
            pass
        elif learning_style == "Auditory":
            # Auditory learners might benefit from more frequent, shorter sessions
            pass
        # Other learning style adjustments would go here
        
        # Return the schedule
        return {
            "daily_study_hours": study_hours,
            "subjects": subjects,
            "total_tasks": len(all_tasks),
            "overdue_tasks": len(overdue_tasks),
            "schedule_generated_at": datetime.now()
        }
    
    def close_connection(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

# Helper function to get a planner instance
def get_planner() -> Planner:
    """Get a planner agent instance"""
    return Planner() 