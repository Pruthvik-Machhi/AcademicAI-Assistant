import groq
import asyncio
from langchain_groq import ChatGroq
from typing import List, Dict, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

class LLMConfig:
    """Configuration settings for the LLM."""
    model: str = "llama3-70b-8192"  # Updated to use Groq's llama3 model
    max_tokens: int = 1024
    default_temp: float = 0.5

class GroqLLaMa:
    """
    A wrapper class for ChatGroq to maintain backward compatibility.
    This class provides a bridge between the existing codebase and LangChain's expectations.
    """

    def __init__(self, api_key: str):
        """Initialize GroqLLaMa with API key.

        Args:
            api_key (str): Groq Cloud API authentication key
        """
        self.config = LLMConfig()
        # Create a ChatGroq model that implements the Runnable interface
        self.chat_model = ChatGroq(
            model_name=self.config.model,
            api_key=api_key,
            max_tokens=self.config.max_tokens,
            temperature=self.config.default_temp
        )
        # Keep the direct groq client for backward compatibility
        self.groq_client = groq.Client(api_key=api_key)
        
    def __getattr__(self, name):
        """
        Pass through any attribute access to the underlying ChatGroq model.
        This allows this class to behave like the ChatGroq model for LangChain compatibility.
        """
        return getattr(self.chat_model, name)
    
    async def agenerate(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None
    ) -> str:
        """Generate text using Groq's model via ChatGroq.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0, default from config)

        Returns:
            str: Generated text response
        """
        # Use the groq client directly for async operation
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.groq_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature or self.config.default_temp,
                max_tokens=self.config.max_tokens
            )
        )
        
        return response.choices[0].message.content
    
    def generate(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None
    ) -> str:
        """Synchronous version of generate for non-async contexts.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0, default from config)

        Returns:
            str: Generated text response
        """
        # For direct usage outside of LangChain's Runnable interface
        response = self.groq_client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature or self.config.default_temp,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content
