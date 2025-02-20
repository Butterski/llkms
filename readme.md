# LLKMS - Language Learning Knowledge Management System

LLKMS is a powerful document processing and question-answering system designed for managing and retrieving knowledge from diverse document formats. Built with **LangChain** and powered by configurable language models (e.g., OpenAI or DeepSeek), it integrates with **Amazon S3** for storage, leverages **Retrieval Augmented Generation (RAG)** for context-aware responses, and includes features like cost tracking and detailed logging. Whether you're processing PDFs, images, or Word documents, LLKMS provides a flexible and extensible platform for language learning and knowledge management.

## Features

- **Multi-format Document Processing**  
  - Supports `.pdf`, `.txt`, `.png`/`.jpg`/`.jpeg` (with OCR), `.docx`, and `.html`/`.htm` files.  
  - Splits documents into chunks for efficient indexing and retrieval using `RecursiveCharacterTextSplitter`.

- **Cloud Storage Integration**  
  - Connects to Amazon S3 buckets with configurable prefixes.  
  - Downloads and processes files asynchronously for scalability.

- **Retrieval Augmented Generation (RAG)**  
  - Utilizes **FAISS** for fast vector storage and similarity search.  
  - Delivers concise, context-aware answers (limited to three sentences) using a custom prompt.  
  - Supports configurable LLMs via a `ModelFactory` (e.g., DeepSeek’s `deepseek-chat` or OpenAI models).

- **Cost and Usage Tracking**  
  - Monitors token usage (total, prompt, completion) and API costs.  
  - Logs successful requests and displays a summary upon exit.  
  - Integrates with LangChain’s callback system for accurate metrics.

- **Comprehensive Logging**  
  - Logs to both console and file (`logs/llkms.log`) with configurable levels (INFO, DEBUG, etc.).  
  - Tracks operations, errors, and debugging details for transparency.

## Prerequisites

- **Python 3.9+**  
- **API Key**: OpenAI (`OPENAI_API_KEY`) or DeepSeek (`DEEPSEEK_API_KEY`), depending on your model provider.  
- **AWS Credentials**: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for S3 access.  
- **Tesseract OCR**: Required for image processing (install separately).

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Butterski/llkms.git
   cd llkms
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**  
   - Copy the example `.env` file:  
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` with your credentials:  
     ```
     AWS_ACCESS_KEY_ID=your_aws_access_key
     AWS_SECRET_ACCESS_KEY=your_aws_secret_key
     OPENAI_API_KEY=your_openai_api_key        # Optional, for OpenAI
     DEEPSEEK_API_KEY=your_deepseek_api_key    # Optional, for DeepSeek
     ```

4. **Install Tesseract OCR**  
   - Follow the [Tesseract installation guide](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract) for your operating system.

## Usage

1. **Run the Application**  
   ```bash
   python src/llkms/main.py
   ```
   - The system loads configurations from `config.yaml`, connects to your S3 bucket, processes documents, builds a vector store, and starts an interactive session.

2. **Interactive Query Mode**  
   - Choose "RAG Pipeline with S3" from the menu.  
   - Ask questions (e.g., "What is in my documents?") and get concise answers.  
   - Optionally view retrieved documents for context.  
   - Type `quit` to exit and see a usage summary.

3. **Force Reindexing**  
   - To rebuild the vector store (bypassing the cache), use:  
     ```bash
     python src/llkms/main.py --reindex
     ```

4. **Configuration Options**  
   - Edit `config.yaml` to adjust:  
     - **AWS**: Bucket name (`eng-llkms`) and prefix (`knowledge`).  
     - **Model**: Provider (`deepseek` or `openai`), model name, temperature, and max tokens.  
     - **App**: Temporary directory (`temp`) and vector store cache (`vector_store_cache`).


## Token Usage and Costs

LLKMS tracks:  
- **Total Tokens**: Sum of all tokens used.  
- **Prompt/Completion Tokens**: Split for detailed analysis.  
- **Successful Requests**: Number of API calls completed.  
- **Total Cost**: Estimated cost in USD.  
A summary is logged and displayed when you exit the application.

## Supported File Types

- `.txt`: Plain text  
- `.pdf`: PDF documents  
- `.png`, `.jpg`, `.jpeg`: Images (via OCR)  
- `.docx`: Microsoft Word  
- `.html`, `.htm`: HTML pages  

## Contributing

1. Fork the repository.  
2. Create a feature branch (`git checkout -b feature/awesome-feature`).  
3. Commit your changes (`git commit -m "Add awesome feature"`).  
4. Push to the branch (`git push origin feature/awesome-feature`).  
5. Submit a Pull Request.

## Acknowledgments

- [LangChain](https://pypi.org/project/langchain/) - Core framework for RAG and document processing.  
- [OpenAI](https://openai.com/) - Optional LLM provider.  
- [DeepSeek](https://api-docs.deepseek.com/) - Default LLM provider.  
- [FAISS](https://github.com/facebookresearch/faiss) - Vector storage and search.  
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Image text extraction.  
- [Questionary](https://pypi.org/project/questionary/) - Interactive CLI prompts.

