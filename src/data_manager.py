from datetime import datetime, timezone, timedelta
import json
from typing import Dict, List

class DataManager:

    def __init__(self):
        """
        Initialize data storage containers.
        All data sources start as None until explicitly loaded through load_data().
        """
        self.profile_data = None
        self.calendar_data = None
        self.task_data = None

    def load_data(self, profile_json: str, calendar_json: str, task_json: str):
        """
        Load and parse multiple JSON data sources simultaneously.

        Args:
            profile_json (str): JSON string containing user profile information
            calendar_json (str): JSON string containing calendar events
            task_json (str): JSON string containing task/todo items

        Note: This method expects valid JSON strings. Any parsing errors will propagate up.
        """
        self.profile_data = json.loads(profile_json)
        self.calendar_data = json.loads(calendar_json)
        self.task_data = json.loads(task_json)

    def get_student_profile(self, student_id: str) -> Dict:
        """
        Retrieve a specific student's profile using their unique identifier.

        Args:
            student_id (str): Unique identifier for the student

        Returns:
            Dict: Student profile data if found, None otherwise

        Implementation Note:
            Uses generator expression with next() for efficient search through profiles,
            avoiding full list iteration when possible.
        """
        if self.profile_data:
            return next((p for p in self.profile_data["profiles"]
                        if p["id"] == student_id), None)
        return None

    def parse_datetime(self, dt_str: str) -> datetime:
        """
        Smart datetime parser that handles multiple formats and ensures UTC timezone.

        Args:
            dt_str (str): DateTime string in ISO format, with or without timezone

        Returns:
            datetime: Parsed datetime object in UTC timezone

        Implementation Note:
            Handles both timezone-aware and naive datetime strings by:
            1. First attempting to parse with timezone information
            2. Falling back to assuming UTC if no timezone is specified
        """
        try:
            # First attempt: Parse ISO format with timezone
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            return dt.astimezone(timezone.utc)
        except ValueError:
            # Fallback: Assume UTC if no timezone provided
            dt = datetime.fromisoformat(dt_str)
            return dt.replace(tzinfo=timezone.utc)

    def get_upcoming_events(self, days: int = 7) -> List[Dict]:
        """
        Intelligently filter and retrieve upcoming calendar events within a specified timeframe.

        Args:
            days (int): Number of days to look ahead (default: 7)

        Returns:
            List[Dict]: List of upcoming events, chronologically ordered

        Implementation Note:
            - Uses UTC timestamps for consistent timezone handling
            - Implements error handling for malformed event data
            - Only includes events that start in the future up to the specified timeframe
        """
        if not self.calendar_data:
            return []

        now = datetime.now(timezone.utc)
        future = now + timedelta(days=days)

        events = []
        for event in self.calendar_data.get("events", []):
            try:
                start_time = self.parse_datetime(event["start"]["dateTime"])

                if now <= start_time <= future:
                    events.append(event)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not process event due to {str(e)}")
                continue

        return events

    def get_active_tasks(self) -> List[Dict]:
        """
        Retrieve and filter active tasks, enriching them with parsed datetime information.

        Returns:
            List[Dict]: List of active tasks with parsed due dates

        Implementation Note:
            - Filters for tasks that are:
              1. Not completed ("needsAction" status)
              2. Due in the future
            - Enriches task objects with parsed datetime for easier processing
            - Implements robust error handling for malformed task data
        """
        if not self.task_data:
            return []

        now = datetime.now(timezone.utc)
        active_tasks = []

        for task in self.task_data.get("tasks", []):
            try:
                due_date = self.parse_datetime(task["due"])
                if task["status"] == "needsAction" and due_date > now:
                    # Enrich task object with parsed datetime
                    task["due_datetime"] = due_date
                    active_tasks.append(task)
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not process task due to {str(e)}")
                continue

        return active_tasks