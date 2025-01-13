# LLKMS - Language Learning Knowledge Management System

A document processing and question-answering system built with LangChain and OpenAI, supporting multiple document types and S3 storage integration.

## Features

- **Multi-format Document Processing**
  - PDF documents
  - Text files
  - Images (with OCR capabilities)
  
- **Cloud Storage Integration**
  - Amazon S3 bucket support
  - Configurable bucket and prefix paths
  
- **RAG (Retrieval Augmented Generation)**
  - Vector storage using FAISS
  - Context-aware responses
  - Configurable LLM models
  
- **Cost and Usage Tracking**
  - Token usage monitoring
  - Cost calculation for API calls
  - Detailed usage statistics
  
- **Comprehensive Logging**
  - File and console logging
  - Debug and error tracking
  - Operation auditing

## Prerequisites

- Python 3.8+
- OpenAI API key
- AWS credentials (for S3 access)
- Tesseract OCR (for image processing)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd llkms
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
cp .env.example .env

# Add your credentials
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

## Usage

1. Start the application:
```bash
python main.py
```

2. The system will:
   - Connect to your S3 bucket
   - Process all supported documents
   - Create a vector store
   - Start an interactive query session

3. Enter questions when prompted, or type 'quit' to exit

## Project Structure

```
llkms/
├── main.py                 # Application entry point
├── utils/
│   ├── aws/
│   │   └── s3_client.py   # S3 integration
│   ├── langchain/
│   │   ├── document_processor.py  # Document processing
│   │   └── rag_pipeline.py       # RAG implementation
│   └── logger.py          # Logging configuration
├── logs/                  # Log files
└── temp/                  # Temporary file storage
```

## Configuration

- Modify chunk sizes in `document_processor.py`
- Adjust LLM model in `rag_pipeline.py`
- Configure logging levels in `logger.py`

## Token Usage and Costs

The system tracks:
- Total tokens used
- Prompt and completion tokens
- Successful API requests
- Total cost in USD

A usage summary is displayed when exiting the application.

## Supported File Types

- `.txt` - Text files
- `.pdf` - PDF documents
- `.png`, `.jpg`, `.jpeg` - Images

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments

- LangChain
- OpenAI
- FAISS
- Tesseract OCR
