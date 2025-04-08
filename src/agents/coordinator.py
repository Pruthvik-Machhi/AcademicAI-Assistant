"""
Coordinator Agent - The orchestration layer for the Academic AI Assistant

This agent is responsible for:
1. Managing the workflow between specialized agents
2. Maintaining the system state
3. Routing tasks to appropriate specialized agents
4. Ensuring consistent communication
"""

from typing import Dict, List, Any, Optional, Tuple, Literal
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
from enum import Enum

# Load environment variables
load_dotenv()

# Agent state types
class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

# Define the overall system state schema
class AssistantState:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.current_agent: Optional[str] = None
        self.agent_states: Dict[str, AgentState] = {
            "coordinator": AgentState.IDLE,
            "planner": AgentState.IDLE,
            "notewriter": AgentState.IDLE,
            "advisor": AgentState.IDLE
        }
        self.task_state: Dict[str, Any] = {}
        self.student_profile: Dict[str, Any] = {}
        self.last_error: Optional[str] = None

# Main coordinator class
class Coordinator:
    """Coordinator Agent for managing workflow between specialized agents"""
    
    def __init__(self, state: Optional[AssistantState] = None):
        """Initialize coordinator with optional existing state"""
        self.state = state or AssistantState()
        self.agent_graph = self._build_agent_graph()
    
    def _build_agent_graph(self) -> StateGraph:
        """Build the state graph for agent interaction workflow"""
        # Define the state graph
        workflow = StateGraph(AssistantState)
        
        # Define the nodes - these would be the agent processing functions in full implementation
        workflow.add_node("coordinator", self.process_coordinator)
        workflow.add_node("planner", self.process_planner)
        workflow.add_node("notewriter", self.process_notewriter)
        workflow.add_node("advisor", self.process_advisor)
        
        # Define the edges (transitions between agents)
        workflow.add_edge("coordinator", "planner")
        workflow.add_edge("coordinator", "notewriter")
        workflow.add_edge("coordinator", "advisor")
        workflow.add_edge("planner", "coordinator")
        workflow.add_edge("notewriter", "coordinator")
        workflow.add_edge("advisor", "coordinator")
        
        # Define conditional routing
        workflow.add_conditional_edges(
            "coordinator",
            self.route_task,
            {
                "planner": "planner",
                "notewriter": "notewriter", 
                "advisor": "advisor",
                "end": END
            }
        )
        
        # Set the entry point
        workflow.set_entry_point("coordinator")
        
        return workflow
    
    def process_coordinator(self, state: AssistantState) -> AssistantState:
        """Process the coordinator agent's tasks"""
        # Update state to show coordinator is working
        state.current_agent = "coordinator"
        state.agent_states["coordinator"] = AgentState.PROCESSING
        
        # In a full implementation, this would process the request and determine next steps
        
        # Mark coordinator as completed
        state.agent_states["coordinator"] = AgentState.COMPLETED
        return state
    
    def process_planner(self, state: AssistantState) -> AssistantState:
        """Process planner agent's tasks"""
        # Update state to show planner is working
        state.current_agent = "planner"
        state.agent_states["planner"] = AgentState.PROCESSING
        
        # In a full implementation, this would handle calendar and schedule tasks
        
        # Mark planner as completed
        state.agent_states["planner"] = AgentState.COMPLETED
        return state
    
    def process_notewriter(self, state: AssistantState) -> AssistantState:
        """Process notewriter agent's tasks"""
        # Update state to show notewriter is working
        state.current_agent = "notewriter"
        state.agent_states["notewriter"] = AgentState.PROCESSING
        
        # In a full implementation, this would process academic content
        
        # Mark notewriter as completed
        state.agent_states["notewriter"] = AgentState.COMPLETED
        return state
    
    def process_advisor(self, state: AssistantState) -> AssistantState:
        """Process advisor agent's tasks"""
        # Update state to show advisor is working
        state.current_agent = "advisor"
        state.agent_states["advisor"] = AgentState.PROCESSING
        
        # In a full implementation, this would generate personalized advice
        
        # Mark advisor as completed
        state.agent_states["advisor"] = AgentState.COMPLETED
        return state
    
    def route_task(self, state: AssistantState) -> Literal["planner", "notewriter", "advisor", "end"]:
        """Determine which agent should handle the current task next"""
        # This is a simplified routing logic
        # In a full implementation, this would analyze the request to determine the right agent
        
        # Check if a specific task type is set
        task_type = state.task_state.get("type", "")
        
        if task_type == "schedule" or task_type == "calendar":
            return "planner"
        elif task_type == "notes" or task_type == "content":
            return "notewriter"
        elif task_type == "advice" or task_type == "recommendation":
            return "advisor"
        else:
            # If all agents have completed their tasks, end the workflow
            if all(agent_state == AgentState.COMPLETED for agent_state in state.agent_states.values()):
                return "end"
            
            # No clear task type, default to advisor
            return "advisor"
    
    def process_request(self, user_request: str, student_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user request through the agent workflow"""
        # Initialize a new state for this request
        state = AssistantState()
        
        # Add user message to state
        state.messages.append({"role": "user", "content": user_request})
        
        # Update student profile if provided
        if student_profile:
            state.student_profile = student_profile
        
        # Determine task type from request
        # This is a simple keyword-based approach - a real implementation would use NLP
        task_type = "unknown"
        if any(word in user_request.lower() for word in ["schedule", "plan", "calendar", "time"]):
            task_type = "schedule"
        elif any(word in user_request.lower() for word in ["note", "summarize", "content", "lecture"]):
            task_type = "notes"
        elif any(word in user_request.lower() for word in ["advice", "help", "recommend", "suggestion"]):
            task_type = "advice"
        
        # Set the task type in state
        state.task_state["type"] = task_type
        
        # Process through the workflow
        try:
            final_state = self.agent_graph.invoke(state)
            return {
                "success": True,
                "messages": final_state.messages,
                "task_state": final_state.task_state
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "messages": [{"role": "assistant", "content": f"An error occurred: {str(e)}"}]
            }

# Helper function to create a coordinator instance
def get_coordinator() -> Coordinator:
    """Get a singleton coordinator instance"""
    return Coordinator() 