# DocumentAI Chat

An enterprise-level document chatbot with advanced RAG capabilities, token optimization, and PDF processing features.

## Features

- **Advanced PDF Processing**
  - Intelligent detection of document type (digital vs. scanned)
  - High-quality OCR for scanned documents with image preprocessing
  - Structured text extraction with metadata
  - Multi-threaded processing for better performance

- **Token Management & Optimization**
  - Precise token counting with tiktoken
  - Dynamic context selection based on token limits
  - Smart chunk prioritization with relevance scores
  - Graceful recovery from token limit errors

- **Enterprise-Grade RAG**
  - Semantic search using FAISS and HuggingFace embeddings
  - Token-aware retrieval for optimal context selection
  - Section-aware chunking to preserve document structure
  - Configurable parameters for fine-tuning

- **User-Friendly Interface**
  - Intuitive Streamlit UI with chat interface
  - Document information display
  - Source citations with page references
  - Advanced settings for customization

## Architecture

```
project/
├── app.py                 # Main application entry point
├── core/
│   ├── ocr.py            # OCR processing
│   ├── pdf_parser.py     # PDF extraction and processing
│   ├── chunking.py       # Text chunking with token awareness
│   └── utils.py          # Utility functions
├── embeddings/
│   └── rag_engine.py     # RAG implementation with token management
├── templates/
│   └── prompt_template.txt  # LLM prompt template
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/documentai-chat.git
cd documentai-chat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following:
```
HUGGINGFACE_API_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
MAX_CONTEXT_TOKENS=4000
MAX_RESPONSE_TOKENS=1000
DEFAULT_MODEL=llama3-70b-8192
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload a PDF document using the sidebar.
2. Wait for processing to complete (this might take a few moments for large documents).
3. Ask questions about the document in the chat interface.
4. View sources by expanding the "Sources" section in responses.
5. Adjust settings in the "Advanced Settings" panel if needed.

## Token Management

This application implements sophisticated token management to handle Groq API token limits:

- **Token Counting**: Uses tiktoken for precise token counting
- **Context Selection**: Optimizes retrieval to stay within token limits
- **Dynamic Adjustment**: Adapts to different document sizes and query complexity
- **Error Recovery**: Gracefully handles token limit errors with informative messages

## Requirements

- Python 3.8+
- OCR dependencies (optional):
  - Tesseract OCR
  - Poppler (for pdf2image)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Groq for their powerful LLM API
- HuggingFace for embedding models
- Langchain for RAG components 