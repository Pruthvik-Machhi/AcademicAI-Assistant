import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2
import random
import tempfile

load_dotenv()

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file object with error handling"""
    text = ""
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_path = temp_file.name
        
        # Process the temporary file
        with open(temp_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                # Extract text and split into lines
                page_text = page.extract_text()
                if page_text:  # Check if text extraction succeeded
                    page_lines = page_text.splitlines()
                    # Iterate through lines, keeping only those starting with a letter or number
                    text += "\n".join([line for line in page_lines if line and (line[0].isalpha() or line[0].isdigit())])
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        if not text.strip():
            raise ValueError("No readable text content found in the PDF")
            
        return text
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")

def select_text_from_pdf(pdf_file, batch_size=10000, num_batches=2):
    """Select representative text from PDF with error handling"""
    try:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_file)
        
        # Check if we have enough text
        total_chars = len(pdf_text)
        if total_chars < batch_size:
            return pdf_text  # Return all text if less than batch size
        
        # Calculate required length of sample
        required_length = batch_size * num_batches
        if total_chars <= required_length:
            return pdf_text  # Return all text if less than required length
        
        # Randomly select starting indices for the batches
        max_start_index = total_chars - batch_size
        selected_start_indices = random.sample(range(max_start_index), min(num_batches, max_start_index))
        
        # Initialize the selected text
        selected_text = ""
        
        # Extract text for each selected batch and concatenate
        for start_index in selected_start_indices:
            end_index = start_index + batch_size
            selected_text += pdf_text[start_index:end_index]
        
        return selected_text
    except Exception as e:
        raise ValueError(f"Error selecting text from PDF: {str(e)}")

# Initialize Groq model
def get_groq_model():
    """Get Groq model with error handling"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return ChatGroq(model="llama3-70b-8192", api_key=api_key)
    except Exception as e:
        raise ValueError(f"Error initializing Groq model: {str(e)}")

def get_content_from_input(uploaded_file, text_input):
    """Extract text from either PDF or text input with validation"""
    if uploaded_file is not None:
        try:
            return select_text_from_pdf(uploaded_file)
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
    elif text_input and text_input.strip():
        return text_input.strip()
    raise ValueError("Please provide either a PDF file or text input")

def generate_mcqs(content, difficulty, num_questions=10):
    """Generate MCQs using Groq model with number of questions parameter"""
    if not content or len(content) < 100:
        raise ValueError("Insufficient content provided. Please provide more text.")
    
    # Limit content size to avoid token limits
    max_content_length = 25000  # Approximate token limit
    if len(content) > max_content_length:
        content = content[:max_content_length]
    
    mcq_prompt = ChatPromptTemplate.from_template(
        """You're an expert quiz creator specializing in {difficulty} level questions. 
        Generate {num_questions} high-quality MCQs based EXCLUSIVELY on the following content:
        {content}
        
        Requirements:
        - Each question must cover different key concepts
        - Questions should progress from basic to advanced (for higher difficulty)
        - Format each question as:
            Q1. [Question text]
            a) [Option A]
            b) [Option B]
            c) [Option C]
            d) [Option D]
        - Provide answer key in format:
            Answer Key:
            1. [correct_letter]
            2. [correct_letter]
            ...
            {num_questions}. [correct_letter]
        - Avoid markdown formatting
        - Ensure answers are factually correct based on provided content"""
    )
    
    try:
        model = get_groq_model()
        chain = mcq_prompt | model
        response = chain.invoke({
            "difficulty": difficulty, 
            "content": content,
            "num_questions": num_questions
        })
        return response.content
    except Exception as e:
        raise ValueError(f"Error generating MCQs: {str(e)}")

def parse_mcq_response(response, expected_count):
    """Parse generated MCQs and answer key with robust validation"""
    questions = []
    answer_key = []
    
    # Validate response
    if not response or "Answer Key:" not in response:
        raise ValueError("Invalid response format. Please try generating MCQs again.")
    
    # Split questions and answers
    parts = response.split("Answer Key:")
    question_part = parts[0].strip() if len(parts) > 0 else ""
    answer_part = parts[1].strip() if len(parts) > 1 else ""
    
    # Extract questions using improved regex
    # Look for patterns like "Q1." or "1." at the beginning of a line
    question_blocks = re.split(r'(?:^|\n)(?:Q?(\d+)\.)', question_part)
    
    # Process question blocks
    current_q = None
    for block in question_blocks:
        if not block:
            continue
        if block.isdigit():
            current_q = int(block)
        elif current_q is not None:
            # Add the question with its number
            questions.append(f"Q{current_q}. {block.strip()}")
            current_q = None
    
    # Extract answers with improved pattern matching
    answer_entries = re.findall(r'(?:^|\n)(?:Q?(\d+)\.?\s*([a-d]))', answer_part, re.IGNORECASE)
    answer_dict = {int(num): letter.lower() for num, letter in answer_entries}
    
    # Create ordered answer key
    for i in range(1, expected_count + 1):
        if i in answer_dict:
            answer_key.append(answer_dict[i])
        else:
            # If missing answers, add placeholder
            answer_key.append("")
    
    # Validate results
    if len(questions) < expected_count or len(answer_key) < expected_count:
        # Return partial results rather than failing completely
        st.warning(f"Expected {expected_count} questions but only parsed {len(questions)} questions and {len(answer_key)} answers.")
    
    return questions[:expected_count], answer_key[:expected_count]

def analyze_performance(content, questions, correct_answers, user_answers):
    """Analyze user performance and provide feedback"""
    # Validate inputs
    if not questions or not correct_answers or not user_answers:
        raise ValueError("Missing data for performance analysis")
    
    # Calculate actual score for validation
    score = sum(1 for ua, ca in zip(user_answers, correct_answers) if ua == ca)
    total = len(questions)
    
    # Limit content size to avoid token limits
    max_content_length = 20000  # Approximate token limit
    if len(content) > max_content_length:
        content = content[:max_content_length]
    
    # Create detailed wrong answer information
    wrong_answers = []
    for i, (q, ua, ca) in enumerate(zip(questions, user_answers, correct_answers)):
        if ua != ca:
            wrong_answers.append(f"Question {i+1}: User answered '{ua}', correct answer is '{ca}'")
    
    analysis_prompt = f"""Analyze the quiz performance based on:
    - Original content: {content}
    - Questions: {questions}
    - Correct answers: {correct_answers}
    - User answers: {user_answers}
    - Actual score: {score}/{total} ({int(score/total*100)}%)
    - Wrong answers: {wrong_answers}

    Provide detailed analysis covering:
    1. Overall score and accuracy percentage (which is {score}/{total}, {int(score/total*100)}%)
    2. List of incorrect answers with brief explanations for questions: {', '.join([str(i+1) for i, (ua, ca) in enumerate(zip(user_answers, correct_answers)) if ua != ca])}
    3. Identification of 2-3 weak areas/topics needing improvement
    4. Specific study recommendations for each weak area
    5. Encouraging feedback highlighting strengths
    
    Format the analysis clearly with headings for each section.
    Avoid markdown and keep language professional yet supportive."""
    
    try:
        model = get_groq_model()
        chain = ChatPromptTemplate.from_messages([("human", analysis_prompt)]) | model
        response = chain.invoke({})
        return response.content
    except Exception as e:
        raise ValueError(f"Error analyzing performance: {str(e)}")

# Streamlit UI
st.title("📚 MCQ Generator & Analyzer Agent")

# Input Section
st.header("Input Content")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
text_input = st.text_area("Or Enter Text Content", height=200)
difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])

# Add number of questions input
num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)

# MCQ generation with error handling
if st.button("Generate MCQs"):
    try:
        with st.spinner("Processing content..."):
            content = get_content_from_input(uploaded_file, text_input)
        
        with st.spinner(f"Generating {num_questions} questions..."):
            mcq_response = generate_mcqs(content, difficulty, num_questions)
            questions, answer_key = parse_mcq_response(mcq_response, num_questions)
            
            if len(questions) == num_questions and len(answer_key) == num_questions:
                st.session_state.questions = questions
                st.session_state.answer_key = answer_key
                st.session_state.user_answers = [""] * num_questions
                st.session_state.num_questions = num_questions
                st.session_state.content = content
                st.success(f"{num_questions} MCQs generated successfully!")
            else:
                st.warning(f"Generated {len(questions)} questions instead of {num_questions}. Proceeding with available questions.")
                st.session_state.questions = questions
                st.session_state.answer_key = answer_key
                st.session_state.user_answers = [""] * len(questions)
                st.session_state.num_questions = len(questions)
                st.session_state.content = content
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Question display section with dynamic number of questions
if "questions" in st.session_state and st.session_state.questions:
    st.header("Generated Questions")
    for i, question in enumerate(st.session_state.questions):
        st.write(f"**Question {i+1}**")
        # Format question display
        formatted_question = question.replace("a)", "\n\na)").replace("b)", "\nb)").replace("c)", "\nc)").replace("d)", "\nd)")
        st.write(formatted_question)
        
        # Get user answer with validation
        user_input = st.text_input(
            f"Your answer for Q{i+1} (a-d):", 
            key=f"ans_{i}",
            value=st.session_state.user_answers[i] if i < len(st.session_state.user_answers) else ""
        ).lower()
        
        # Validate input (only accept a, b, c, d)
        if user_input and user_input not in ['a', 'b', 'c', 'd']:
            st.warning(f"Please enter only a, b, c, or d for question {i+1}")
            st.session_state.user_answers[i] = ""
        else:
            st.session_state.user_answers[i] = user_input

# Evaluation with error handling
if "answer_key" in st.session_state and st.button("Evaluate Answers"):
    # Check if all questions have been answered
    unanswered = [i+1 for i, ans in enumerate(st.session_state.user_answers) if not ans]
    if unanswered:
        st.warning(f"Please answer all questions. Missing answers for questions: {', '.join(map(str, unanswered))}")
    else:
        score = 0
        results = []
        
        for i in range(len(st.session_state.questions)):
            if st.session_state.user_answers[i] == st.session_state.answer_key[i]:
                score += 1
                results.append(f"Q{i+1}: Correct ✅")
            else:
                results.append(f"Q{i+1}: Incorrect ❌ (Correct: {st.session_state.answer_key[i].upper()})")
        
        st.header("Results")
        percentage = round((score / len(st.session_state.questions)) * 100)
        st.subheader(f"Score: {score}/{len(st.session_state.questions)} ({percentage}%)")
        
        with st.expander("See Detailed Results"):
            for result in results:
                st.write(result)
        
        # Performance Analysis
        try:
            with st.spinner("Analyzing performance..."):
                analysis = analyze_performance(
                    st.session_state.content,
                    st.session_state.questions,
                    st.session_state.answer_key,
                    st.session_state.user_answers
                )
                
                st.header("Performance Analysis")
                st.write(analysis)
        except Exception as e:
            st.error(f"Error during performance analysis: {str(e)}")
            st.info("Performance analysis is unavailable, but your score is shown above.")