# Academic AI Assistant

Academic AI Assistant is a powerful multi-agent system designed to transform the way students manage their academic life. Using LangChain's advanced RAG capabilities, it creates a network of specialized AI agents that work together to provide personalized academic support.

## Features

- **Home**: Central dashboard, student profile management
- **Notewriter**: Processing academic content into personalized study materials using AI
- **Planner**: Schedule optimization and task management
- **Advisor**: Personalized learning advice based on your profile and academic data
- **PDF Chat**: Advanced RAG-based chat with PDFs, notes, and multi-source knowledge bases
- **Profile Management**: Create and manage student profiles with learning styles, academic goals, and performance metrics.
- **AI-Enhanced Learning Assistant**: Chat with an AI assistant trained on educational resources and personalized to your profile.
- **Notewriter Agent**: Transform content from various sources (text, web pages, PDFs, YouTube videos) into structured study notes tailored to your learning style.
- **Study Planner**: Create and manage study plans with time blocks and task prioritization.
- **Performance Tracking**: Record and analyze academic performance over time.
- **Resource Management**: Save and organize learning resources and references.
- **Advisor**: Receive personalized learning strategies and time management advice.
- **Chatbot Interface**: Ask questions about your academic journey or get assistance with specific topics.
- **RAG-based PDF Chat**: Interact with your study materials using advanced retrieval-augmented generation.
- **Multi-Source Knowledge Base**: Combine multiple documents for unified querying.

## Content Processing Features

The Academic AI Assistant can process content from multiple sources:

### YouTube Video Notes
Convert any educational YouTube video into comprehensive study notes:
1. Simply paste a YouTube URL in the Notewriter section
2. Select your preferred output format (Comprehensive Notes, Brief Summary, etc.)
3. The system will extract the video transcript and generate structured notes
4. Notes include timestamps for easy reference back to the video

### Web Page Notes
Transform articles, blog posts, and educational websites into study materials:
1. Enter any webpage URL in the Notewriter section
2. The system will extract the main content, removing ads and navigation elements
3. Generate well-organized notes focused on key concepts

### PDF Document Processing
Extract and process content from PDF lecture slides, research papers, and textbooks:
1. Upload any PDF file through the simple interface
2. The system will extract text content page by page
3. Generate structured notes that preserve the document's organization

### Advanced PDF & Notes Chat (New!)
The PDF Chat feature allows you to have intelligent conversations with your study materials:

1. **PDF Upload**: Upload any PDF document to chat with its content
2. **Notes Integration**: Select from your saved notes to ask questions
3. **Multi-Source Knowledge Base**: Combine multiple documents (notes, PDFs, syllabi) into a unified knowledge base
4. **Retrieval-Augmented Generation (RAG)**: Get precise answers with references to specific sections of your documents
5. **Source Attribution**: Every answer includes citations to the specific parts of the documents where the information was found
6. **Intelligent Chunking**: Documents are automatically divided into optimal segments for accurate retrieval
7. **Context-Aware Responses**: The system understands the context of your questions in relation to your documents

## Requirements

- Python 3.8+
- PostgreSQL 12+
- Groq API key (for all LLM capabilities - this project does not use OpenAI)
- HuggingFace's all-MiniLM-L6-v2 model (automatically downloaded for embeddings)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Academic-AI-Assistant.git
cd Academic-AI-Assistant
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Set up PostgreSQL:
   - Install PostgreSQL if you haven't already
   - Update the connection settings in the `.env` file if needed
   - Run the database initialization script:
   ```bash
   python init_db.py
   ```

4. Create `.env` file with your configurations:

```
# PostgreSQL Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=academic_assistant
DB_USER=postgres
DB_PASSWORD=postgres

# LLM API Keys
GROQ_API_KEY=your_groq_api_key

# Application Settings
DEBUG=True
SECRET_KEY=your_secret_key_here
```

## Usage

Run the Streamlit application:

```bash
streamlit run academic_ai_assistant.py
```

The application will be available at `http://localhost:8501`.

## Agent Architecture

The Academic AI Assistant uses a multi-agent architecture:

1. **Coordinator Agent**: Orchestrates the interaction between specialized agents and manages the overall system state
2. **Planner Agent**: Handles calendar integration and schedule optimization
3. **Notewriter Agent**: Processes academic content and generates study materials
4. **Advisor Agent**: Provides personalized learning and time management advice

These agents work together to provide a comprehensive academic support system.

## Database Structure

The application uses PostgreSQL to store:

- Student profiles
- Tasks and deadlines
- Notes and study materials
- Knowledge base content (including syllabus data)

## RAG Implementation Details

The PDF Chat feature uses a sophisticated Retrieval-Augmented Generation pipeline:

1. **Document Processing**: PDFs and notes are processed into document objects with metadata
2. **Text Chunking**: Documents are split into smaller segments with optimal overlap using RecursiveCharacterTextSplitter
3. **Embedding Generation**: Text chunks are converted to vector embeddings using HuggingFace's all-MiniLM-L6-v2 model
4. **Vector Storage**: FAISS is used for efficient similarity search
5. **Query Processing**: User questions are processed to retrieve the most relevant document chunks
6. **Answer Generation**: The LLM generates answers based only on the retrieved content, ensuring accuracy
7. **Source Attribution**: Answers include references to the specific chunks or documents used

## Development

### Project Structure

```
Academic-AI-Assistant/
├── academic_ai_assistant.py       # Main Streamlit application
├── requirements.txt               # Dependencies
├── .env                          # Environment variables
├── src/
│   ├── __init__.py
│   ├── LLM.py                    # LLM integration
│   ├── extractors.py             # Content extraction utilities
│   └── agents/
│       ├── __init__.py
│       ├── planner.py            # Planner agent
│       ├── notewriter.py         # Notewriter agent
│       └── advisor.py            # Advisor agent
```

### Adding New Features

To add new features:

1. Develop the functionality in the appropriate agent module
2. Update the corresponding Streamlit page in `academic_ai_assistant.py`
3. Update any necessary database tables

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- LangChain for RAG components and LLM integration
- FAISS for vector search capabilities 
- HuggingFace for embedding models
- Streamlit for the web interface
- Groq for LLM API access

## LLM Integration

The application exclusively uses Groq's powerful LLMs (via ChatGroq) to power all AI features:

1. **Chat Interface**: General academic assistance on the home page
2. **Content Processing**: Convert lecture notes and readings into structured study materials
3. **Personalized Advice**: Generate tailored studying advice based on learning style and profile
4. **Document Q&A**: Answer questions about your study materials with RAG-enhanced precision

You'll need a Groq API key to use these features. The application is designed to work with Groq's models for optimal performance.