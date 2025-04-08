# LLM Integration: Groq

This document outlines how Academic AI Assistant integrates with Groq's powerful language models to provide AI-powered features.

## Overview

Academic AI Assistant **exclusively** uses Groq for all LLM capabilities. The application does not depend on or use OpenAI or any other LLM provider.

## Why Groq?

- **High Performance**: Groq's platform delivers extremely fast inference times, which is critical for a responsive educational assistant.
- **Cost-Effective**: Groq provides competitive pricing for high-quality model inference.
- **Powerful Models**: Groq hosts high-quality models like Llama3, which provide excellent performance for educational use cases.

## Implementation Details

The LLM integration is implemented in `src/LLM.py` using two main components:

1. **ChatGroq**: For asynchronous chat completions
2. **Groq Client**: For synchronous completions

The application uses the llama3-70b-8192 model, which provides excellent capabilities for:
- Understanding and generating complex academic content
- Providing personalized learning advice
- Processing and summarizing lecture notes

## Setting Up Your Groq API Key

To use the application, you'll need a Groq API key:

1. Sign up for an account at [groq.com](https://groq.com)
2. Obtain your API key from the dashboard
3. Add it to your `.env` file: `GROQ_API_KEY=your_api_key_here`

## Troubleshooting

If you encounter errors related to LLM functionality:

1. Verify your Groq API key is correctly set in the `.env` file
2. Check that you have internet connectivity
3. Ensure you have the required packages installed: `groq` and `chatgroq`

## Technical Reference

The main class in `src/LLM.py` is `GroqLLaMa`, which provides two primary methods:

- `async agenerate(messages, temperature)`: Asynchronous generation for chat interfaces
- `generate(messages, temperature)`: Synchronous generation for forms and other blocking contexts

Both methods accept:
- `messages`: A list of message dictionaries with "role" and "content" keys
- `temperature`: Optional parameter to control randomness (0.0-1.0)

Example usage:

```python
from src.LLM import GroqLLaMa

# Initialize with your API key
llm = GroqLLaMa(api_key="your_groq_api_key")

# Generate a response
response = llm.generate([
    {"role": "user", "content": "Explain the concept of photosynthesis."}
])

print(response)
``` 