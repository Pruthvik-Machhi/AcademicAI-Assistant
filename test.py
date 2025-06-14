import os
from typing import Dict, List, Any, Optional, Literal
from enum import Enum
from dotenv import load_dotenv
# LangGraph imports assumed
from langgraph.graph import StateGraph, END

# RAG-related imports
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groqcloud import GroqCloud
import PyPDF2
import networkx as nx

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQCLOUD_API_KEY", "")
MODEL_NAME = os.getenv("GROQCLOUD_MODEL_NAME", "chatgroq-7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", 500))
PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", 100))
TOP_K = int(os.getenv("TOP_K", 3))

# Initialize RAG clients/models
gc = GroqCloud(api_key=API_KEY) if API_KEY else None
embedder = SentenceTransformer(EMBED_MODEL)

# Agent state types
enum AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class AssistantState:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.current_agent: Optional[str] = None
        self.agent_states: Dict[str, AgentState] = {
            "coordinator": AgentState.IDLE,
            "planner": AgentState.IDLE,
            "notewriter": AgentState.IDLE,
            "advisor": AgentState.IDLE,
            # RAG agents
            "rag": AgentState.IDLE
        }
        self.task_state: Dict[str, Any] = {}
        self.student_profile: Dict[str, Any] = {}
        self.last_error: Optional[str] = None
        # RAG state
        self.rag_index = None
        self.passages: List[str] = []

class Coordinator:
    """Coordinator Agent integrating Academic AI and RAG workflows"""
    def __init__(self, state: Optional[AssistantState] = None):
        self.state = state or AssistantState()
        self.agent_graph = self._build_agent_graph()

    def _build_agent_graph(self) -> StateGraph:
        workflow = StateGraph(AssistantState)
        workflow.add_node("coordinator", self.process_coordinator)
        workflow.add_node("planner", self.process_planner)
        workflow.add_node("notewriter", self.process_notewriter)
        workflow.add_node("advisor", self.process_advisor)
        workflow.add_node("rag", self.process_rag)
        # Transitions
        workflow.add_edge("coordinator", "planner")
        workflow.add_edge("coordinator", "notewriter")
        workflow.add_edge("coordinator", "advisor")
        workflow.add_edge("coordinator", "rag")
        workflow.add_edge("planner", "coordinator")
        workflow.add_edge("notewriter", "coordinator")
        workflow.add_edge("advisor", "coordinator")
        workflow.add_edge("rag", "coordinator")
        workflow.add_conditional_edges(
            "coordinator",
            self.route_task,
            {
                "planner": "planner",
                "notewriter": "notewriter",
                "advisor": "advisor",
                "rag": "rag",
                "end": END
            }
        )
        workflow.set_entry_point("coordinator")
        return workflow

    def process_coordinator(self, state: AssistantState) -> AssistantState:
        state.current_agent = "coordinator"
        state.agent_states["coordinator"] = AgentState.PROCESSING
        # In full impl, analyze messages and update state
        state.agent_states["coordinator"] = AgentState.COMPLETED
        return state

    def process_planner(self, state: AssistantState) -> AssistantState:
        state.current_agent = "planner"
        state.agent_states["planner"] = AgentState.PROCESSING
        # Calendar/scheduling logic
        state.agent_states["planner"] = AgentState.COMPLETED
        return state

    def process_notewriter(self, state: AssistantState) -> AssistantState:
        state.current_agent = "notewriter"
        state.agent_states["notewriter"] = AgentState.PROCESSING
        # Note-writing logic
        state.agent_states["notewriter"] = AgentState.COMPLETED
        return state

    def process_advisor(self, state: AssistantState) -> AssistantState:
        state.current_agent = "advisor"
        state.agent_states["advisor"] = AgentState.PROCESSING
        # Advice generation logic
        state.agent_states["advisor"] = AgentState.COMPLETED
        return state

    # RAG utility functions
    def load_pdf_and_index(self, pdf_path: str):
        text = self.load_pdf_text(pdf_path)
        passages = self.chunk_text(text, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP)
        index = self.build_faiss_index(passages)
        self.state.passages = passages
        self.state.rag_index = index

    def load_pdf_text(self, path: str) -> str:
        reader = PyPDF2.PdfReader(path)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)

    def chunk_text(self, text: str, size: int, overlap: int):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    def build_faiss_index(self, passages: List[str]):
        embeddings = embedder.encode(passages, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:
        if not self.state.rag_index:
            return []
        q_emb = embedder.encode([query], convert_to_numpy=True)
        _, ids = self.state.rag_index.search(q_emb, top_k)
        return [self.state.passages[i] for i in ids[0]]

    def chat_with_context(self, docs: List[str], question: str) -> str:
        if not gc:
            return "GroqCloud client not configured."
        prompt = (
            "Use the following context to answer the question:\n\n" +
            "\n".join(f"- {d}" for d in docs) +
            f"\n\nQuestion: {question}\nAnswer:"
        )
        messages = state.messages + [{"role": "user", "content": prompt}]
        resp = gc.chat.create(model=MODEL_NAME, messages=messages)
        return resp.choices[0].message.content

    # Example RAG processing agent
    def process_rag(self, state: AssistantState) -> AssistantState:
        state.current_agent = "rag"
        state.agent_states["rag"] = AgentState.PROCESSING
        # Expect state.task_state to include 'rag_query' and optional 'mode'
        q = state.task_state.get("rag_query", "")
        mode = state.task_state.get("rag_mode", "agentic")
        if q and state.rag_index:
            answer = self.execute_rag_mode(mode, q)
            state.messages.append({"role": "assistant", "content": answer})
        else:
            state.messages.append({"role": "assistant", "content": "No RAG query or index available."})
        state.agent_states["rag"] = AgentState.COMPLETED
        return state

    def execute_rag_mode(self, mode: str, q: str) -> str:
        funcs = {
            "corrective": self.corrective_rag,
            "speculative": self.speculative_rag,
            "agentic": self.agentic_rag,
            "single-router": self.single_router_rag,
            "multi-agent": self.multi_agent_rag,
            "self-reflect": self.self_reflective_rag,
            "self-route": self.self_route_rag,
            "graph": self.graph_rag,
        }
        func = funcs.get(mode, self.agentic_rag)
        return func(q)

    # RAG variants
    def corrective_rag(self, q: str) -> str:
        docs = self.retrieve(q)
        initial = self.chat_with_context(docs, q)
        verify = f"You answered:\n“{initial}”\n\nCheck against the provided context and correct any factual errors."
        resp = gc.chat.create(model=MODEL_NAME, messages=[{"role": "user", "content": verify}])
        return resp.choices[0].message.content

    def speculative_rag(self, q: str) -> str:
        _ = self.retrieve(q)
        return self.corrective_rag(q)

    def agentic_rag(self, q: str) -> str:
        use_vector = len(q) % 2 == 0
        docs = self.retrieve(q) if use_vector else ["<keyword-retrieved-doc>"]
        return self.chat_with_context(docs, q)

    def single_router_rag(self, q: str) -> str:
        mode = "vector" if any(x in q.lower() for x in ["define", "what is"]) else "keyword"
        docs = self.retrieve(q) if mode == "vector" else ["<keyword-doc>"]
        return self.chat_with_context(docs, q)

    def multi_agent_rag(self, q: str) -> str:
        v_docs = self.retrieve(q, top_k=2)
        k_docs = ["<keyword-doc>"]
        w_docs = ["<web-scraped-doc>"]
        all_docs = v_docs + k_docs + w_docs
        return self.chat_with_context(all_docs, q)

    def self_reflective_rag(self, q: str) -> str:
        docs = self.retrieve(q, top_k=5)
        filter_prompt = (
            "From these passages, select the 3 most relevant to the question:\n\n" +
            "\n".join(f"{i+1}. {d}" for i, d in enumerate(docs)) +
            f"\n\nQuestion: {q}\nRelevant passages (by number):"
        )
        pick = gc.chat.create(model=MODEL_NAME, messages=[{"role": "user", "content": filter_prompt}])
        picks = [int(n)-1 for n in pick.choices[0].message.content.split() if n.isdigit()]
        selected = [docs[i] for i in picks[:3] if 0 <= i < len(docs)]
        return self.chat_with_context(selected, q)

    def self_route_rag(self, q: str) -> str:
        small_docs = self.retrieve(q, top_k=2)
        small_ans = self.chat_with_context(small_docs, q)
        check_prompt = f"You answered:\n“{small_ans}”\n\nIs the context sufficient? Answer 'yes' or 'no'."
        verdict = gc.chat.create(model=MODEL_NAME, messages=[{"role": "user", "content": check_prompt}])
        if verdict.choices[0].message.content.strip().lower().startswith("no"):
            full_docs = self.retrieve(q, top_k=5)
            return self.chat_with_context(full_docs, q)
        return small_ans

    def graph_rag(self, q: str) -> str:
        # simple KG: co-occurrence graph
        G = nx.Graph()
        for i, doc in enumerate(self.state.passages):
            G.add_node(i, text=doc)
        for i in range(len(self.state.passages)):
            for j in range(i+1, len(self.state.passages)):
                G.add_edge(i, j)
        q_emb = embedder.encode([q], convert_to_numpy=True)
        sims = (embedder.encode(self.state.passages, convert_to_numpy=True) @ q_emb.T).flatten()
        top_idx = sims.argsort()[-2:]
        sub_nodes = set(top_idx)
        for idx in top_idx:
            sub_nodes.update(G.neighbors(idx))
        sub_docs = [self.state.passages[i] for i in sub_nodes]
        return self.chat_with_context(sub_docs, q)

    def route_task(self, state: AssistantState) -> Literal["planner", "notewriter", "advisor", "rag", "end"]:
        task_type = state.task_state.get("type", "")
        if task_type in ("schedule", "calendar"):
            return "planner"
        elif task_type in ("notes", "content"):
            return "notewriter"
        elif task_type in ("advice", "recommendation"):
            return "advisor"
        elif task_type == "rag":
            return "rag"
        else:
            if all(s == AgentState.COMPLETED for s in state.agent_states.values()):
                return "end"
            return "rag"

    def process_request(self, user_request: str, student_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = AssistantState()
        state.messages.append({"role": "user", "content": user_request})
        if student_profile:
            state.student_profile = student_profile
        # Detect if it's a RAG request
        if user_request.lower().startswith("rag:"):
            state.task_state["type"] = "rag"
            # Expect format: "rag:<mode>:<query>"
            parts = user_request.split(':', 2)
            if len(parts) == 3:
                _, mode, query = parts
                state.task_state["rag_mode"] = mode.strip()
                state.task_state["rag_query"] = query.strip()
        else:
            # existing detection
            task_type = "unknown"
            if any(word in user_request.lower() for word in ["schedule", "plan", "calendar", "time"]):
                task_type = "schedule"
            elif any(word in user_request.lower() for word in ["note", "summarize", "content", "lecture"]):
                task_type = "notes"
            elif any(word in user_request.lower() for word in ["advice", "help", "recommend", "suggestion"]):
                task_type = "advice"
            state.task_state["type"] = task_type
        try:
            final_state = self.agent_graph.invoke(state)
            return {"success": True, "messages": final_state.messages, "task_state": final_state.task_state}
        except Exception as e:
            return {"success": False, "error": str(e), "messages": [{"role": "assistant", "content": f"An error occurred: {str(e)}"}]}  

def get_coordinator() -> Coordinator:
    return Coordinator()

class MockCoordinator:
    def __init__(self):
        from datetime import datetime
        # Use the real Coordinator but monkey-patch chat_with_context
        from academic_ai_coordinator_rag import Coordinator, AssistantState
        self.coord = Coordinator(state=AssistantState())
        
        # Monkey-patch chat_with_context to return a predictable response
        def mock_chat_with_context(docs: List[str], question: str) -> str:
            # For testing, return a simple response indicating the question processed
            return f"Processed question: {question}"
        
        # Apply monkey patch
        self.coord.chat_with_context = mock_chat_with_context
        # Also patch corrective_rag speculative etc. to use the mock chat
        self.coord.corrective_rag = lambda q: f"Corrected: {q}"
        self.coord.speculative_rag = lambda q: f"Speculative: {q}"
        self.coord.agentic_rag = lambda q: f"Agentic: {q}"
        self.coord.single_router_rag = lambda q: f"Single-Router: {q}"
        self.coord.multi_agent_rag = lambda q: f"Multi-Agent: {q}"
        self.coord.self_reflective_rag = lambda q: f"Self-Reflective: {q}"
        self.coord.self_route_rag = lambda q: f"Self-Route: {q}"
        self.coord.graph_rag = lambda q: f"Graph: {q}"

    def process(self, user_request: str, profile: Dict[str, Any] = None) -> Dict[str, Any]:
        return self.coord.process_request(user_request, student_profile=profile)


# Define 10 test cases focused on types of Machine Learning
test_cases = [
    {"id": 1, "query": "rag:agentic:What is supervised learning?", "description": "Querying definition of supervised learning."},
    {"id": 2, "query": "rag:agentic:Explain unsupervised learning with example.", "description": "Asking for unsupervised learning explanation."},
    {"id": 3, "query": "rag:agentic:Describe reinforcement learning workflow.", "description": "Requesting reinforcement learning description."},
    {"id": 4, "query": "rag:agentic:What is semi-supervised learning?", "description": "Querying semi-supervised learning concept."},
    {"id": 5, "query": "rag:agentic:Define self-supervised learning.", "description": "Asking about self-supervised learning."},
    {"id": 6, "query": "rag:agentic:Explain transfer learning and its applications.", "description": "Requesting transfer learning explanation."},
    {"id": 7, "query": "rag:agentic:What is online learning in ML?", "description": "Querying online learning concept."},
    {"id": 8, "query": "rag:agentic:Describe batch learning vs online learning.", "description": "Comparison of batch vs online learning."},
    {"id": 9, "query": "rag:agentic:Explain active learning and its benefits.", "description": "Asking about active learning."},
    {"id": 10, "query": "rag:agentic:What is deep learning and how does it differ from traditional ML?", "description": "Querying deep learning vs traditional ML."},
]

# Build a DataFrame to show test definitions
df_tests = pd.DataFrame(test_cases)
import ace_tools as tools; tools.display_dataframe_to_user(name="ML Types Test Cases", dataframe=df_tests)

# Define unittest TestCase
class TestMLTypeQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_coord = MockCoordinator()

    def test_supervised_learning(self):
        resp = self.mock_coord.process("rag:agentic:What is supervised learning?")
        self.assertTrue(resp["success"])
        self.assertIn("Processed question: What is supervised learning?", resp["messages"][-1]["content"])

    def test_unsupervised_learning(self):
        resp = self.mock_coord.process("rag:agentic:Explain unsupervised learning with example.")
        self.assertTrue(resp["success"])
        self.assertIn("Processed question: Explain unsupervised learning with example.", resp["messages"][-1]["content"])

    def test_reinforcement_learning(self):
        resp = self.mock_coord.process("rag:agentic:Describe reinforcement learning workflow.")
        self.assertTrue(resp["success"])
        self.assertIn("Processed question: Describe reinforcement learning workflow.", resp["messages"][-1]["content"])

    def test_semi_supervised_learning(self):
        resp = self.mock_coord.process("rag:agentic:What is semi-supervised learning?")
        self.assertTrue(resp["success"])
        self.assertIn("Processed question: What is semi-supervised learning?", resp["messages"][-1]["content"])

    def test_self_supervised_learning(self):
        resp = self.mock_coord.process("rag:agentic:Define self-supervised learning.")
        self.assertTrue(resp["success"])
        self.assertIn("Processed question: Define self-supervised learning.", resp["messages"][-1]["content"])

    def test_transfer_learning(self):
        resp = self.mock_coord.process("rag:agentic:Explain transfer learning and its applications.")
        self.assertTrue(resp["success"])
        self.assertIn("Processed question: Explain transfer learning and its applications.", resp["messages"][-1]["content"])

    def test_online_learning(self):
        resp = self.mock_coord.process("rag:agentic:What is online learning in ML?")
        self.assertTrue(resp["success"])
        self.assertIn("Processed question: What is online learning in ML?", resp["messages"][-1]["content"])

    def test_batch_vs_online_learning(self):
        resp = self.mock_coord.process("rag:agentic:Describe batch learning vs online learning.")
        self.assertTrue(resp["success"])
        self.assertIn("Processed question: Describe batch learning vs online learning.", resp["messages"][-1]["content"])

    def test_active_learning(self):
        resp = self.mock_coord.process("rag:agentic:Explain active learning and its benefits.")
        self.assertTrue(resp["success"])
        self.assertIn("Processed question: Explain active learning and its benefits.", resp["messages"][-1]["content"])

    def test_deep_learning(self):
        resp = self.mock_coord.process("rag:agentic:What is deep learning and how does it differ from traditional ML?")
        self.assertTrue(resp["success"])
        self.assertIn("Processed question: What is deep learning and how does it differ from traditional ML?", resp["messages"][-1]["content"])

# Run tests and capture results
if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
