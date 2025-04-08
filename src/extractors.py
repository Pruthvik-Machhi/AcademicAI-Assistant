"""
Content extraction utilities for different sources.

This module provides functions to extract content from:
- Web pages
- PDF files
- YouTube videos
"""

import os
import sys
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
from typing import Dict, Any, Optional, Union, List, Tuple
import tempfile
from pathlib import Path
import validators
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

# Import LangChain's document loaders
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Third-party libraries for PDF processing
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("Warning: PDF extraction dependencies not installed")
        # Fallback error message
        def extract_pdf_content(pdf_path):
            return "Error: PDF extraction requires pypdf or PyPDF2 to be installed."

async def extract_website_content(url: str) -> str:
    """
    Extract content from a web page
    
    Args:
        url (str): URL of the webpage to extract
        
    Returns:
        str: Extracted text content
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Get page title
                    title = soup.title.string if soup.title else "Untitled Page"
                    
                    # Remove script and style tags
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.extract()
                    
                    # Extract main content (article, main, or body)
                    main_content = soup.find("article") or soup.find("main") or soup.find("body")
                    
                    # Get text and clean it
                    if main_content:
                        text = main_content.get_text(separator='\n')
                    else:
                        text = soup.get_text(separator='\n')
                    
                    # Clean the text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    # Format the output
                    return f"# {title}\n\nSource: {url}\n\n{text}"
                else:
                    return f"Error: Unable to fetch URL. Status code: {response.status}"
    except Exception as e:
        return f"Error extracting content from website: {str(e)}"

def extract_pdf_content(source: Union[str, bytes]) -> str:
    """
    Extract content from a PDF file using LangChain's PyPDFLoader
    
    Args:
        source (str or bytes): Path to PDF file or PDF bytes
        
    Returns:
        str: Extracted text content
    """
    try:
        # Import LangChain's PDF loader
        from langchain_community.document_loaders import PyPDFLoader
        import tempfile
        
        pdf_file = None
        cleanup_needed = False
        
        # Handle input as file path or bytes
        if isinstance(source, str):
            # It's a file path
            pdf_path = source
        else:
            # It's bytes data, write to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(source)
            temp_file.close()
            pdf_path = temp_file.name
            cleanup_needed = True
        
        # Use LangChain's PyPDFLoader to extract text
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Get metadata from the first page if available
        title = "Untitled PDF"
        num_pages = len(documents)
        
        if documents:
            # Try to extract title from metadata if available
            metadata = documents[0].metadata
            if metadata and 'source' in metadata:
                title = os.path.basename(metadata['source'])
                title = os.path.splitext(title)[0]  # Remove extension
        
        # Combine all page content
        text_content = "\n\n".join([
            f"--- Page {doc.metadata.get('page', i+1)} ---\n\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        # Clean up temp file if needed
        if cleanup_needed:
            os.unlink(pdf_path)
        
        # Format the output
        return f"# {title}\n\nPages: {num_pages}\n\n{text_content}"
    
    except Exception as e:
        # Clean up temp file in case of error
        if cleanup_needed and 'pdf_path' in locals():
            try:
                os.unlink(pdf_path)
            except:
                pass
        return f"Error extracting content from PDF: {str(e)}"

def extract_youtube_id(url):
    """
    Extract YouTube video ID from various URL formats.
    
    Args:
        url (str): YouTube URL in any format
        
    Returns:
        str or None: The YouTube video ID or None if not found
    """
    # Standard YouTube URL format
    parsed_url = urlparse(url)
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            query = parse_qs(parsed_url.query)
            if 'v' in query:
                return query['v'][0]
    
    # Shortened YouTube URL format
    elif parsed_url.netloc == 'youtu.be':
        return parsed_url.path.lstrip('/')
    
    # Try to extract from URL using regex as a fallback
    video_id_match = re.search(r'(?:v=|youtu\.be/|embed/)([^&\?]+)', url)
    if video_id_match:
        return video_id_match.group(1)
    
    return None

async def extract_youtube_content(url: str) -> Union[str, Tuple[bool, str]]:
    """
    Extract transcript from a YouTube video using youtube_transcript_api
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        Union[str, Tuple[bool, str]]: Either the transcript text (if successful)
                                     or a tuple (False, error_message) if failed
    """
    try:
        # Extract video ID from URL
        video_id = extract_youtube_id(url)
        if not video_id:
            return (False, f"Could not extract video ID from URL: {url}")
        
        # Get transcript using youtube_transcript_api
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        if not transcript_list:
            return (False, f"No transcript could be extracted from: {url}")
        
        # Combine all transcript text
        transcript = " ".join(entry["text"] for entry in transcript_list)
        
        # Return the transcript text if successful
        return transcript
    
    except Exception as e:
        # Return a tuple indicating failure and the error message
        return (False, f"Error extracting YouTube content: {str(e)}") 