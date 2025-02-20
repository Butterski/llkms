# LLKMS - Language Learning Knowledge Management System

**LLKMS** is a powerful tool for processing and querying documents in various formats, designed to support language learning and knowledge management. It integrates with **Amazon S3** for cloud storage, uses **LangChain** and **FAISS** for advanced Retrieval Augmented Generation (RAG), and supports configurable language models like OpenAI and DeepSeek. Whether you’re a learner, researcher, or knowledge enthusiast, LLKMS makes it easy to manage and extract insights from your documents.

## Key Features

- **Multi-format Support**: Process `.pdf`, `.txt`, `.png`/`.jpg`/`.jpeg` (with OCR), `.docx`, and `.html`/`.htm` files.
- **Cloud Integration**: Seamlessly connect to Amazon S3 for document storage and retrieval.
- **Smart Retrieval**: Leverage RAG with FAISS for fast, context-aware answers (limited to three sentences).
- **Flexible Models**: Use language models from OpenAI, DeepSeek, or others via a configurable `ModelFactory`.
- **Usage Tracking**: Monitor token usage and API costs with a summary on exit.
- **Detailed Logging**: Comprehensive logs for debugging and transparency (`logs/llkms.log`).

## Prerequisites

- **Python**: 3.9 or higher
- **API Keys**: 
  - OpenAI (`OPENAI_API_KEY`) or DeepSeek (`DEEPSEEK_API_KEY`)
  - AWS (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) for S3
- **Tesseract OCR**: For image processing (install separately)

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
     OPENAI_API_KEY=your_openai_api_key        # Optional
     DEEPSEEK_API_KEY=your_deepseek_api_key    # Optional
     ```

4. **Install Tesseract OCR**
   - See the [Tesseract installation guide](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract) for your OS.

## Usage

1. **Start the Application**
   ```bash
   python src/llkms/main.py
   ```
   - Loads `config.yaml`, connects to S3, processes documents, and opens an interactive menu.

2. **Query Your Documents**
   - Select **"RAG Pipeline with S3"** from the menu.
   - Ask questions (e.g., "What’s in my documents?") and get concise answers.
   - Optionally view retrieved documents.
   - Type `quit` to exit and see usage stats.

3. **Force Reindexing**
   - Rebuild the vector store (skips cache):
     ```bash
     python src/llkms/main.py --reindex
     ```

## Configuration

Customize settings in `config.yaml`:
- **AWS**: Bucket (`eng-llkms`), prefix (`knowledge`)
- **Model**: Provider (`deepseek`/`openai`), model name, temperature, max tokens
- **App**: Temp directory (`temp`), vector store cache (`vector_store_cache`)

Example snippet:
```yaml
aws:
  bucket: eng-llkms
  prefix: knowledge
model:
  provider: deepseek
  model: deepseek-chat
  temperature: 0.7
  max_tokens: 1024
```

## Usage and Cost Tracking

LLKMS tracks:
- **Total Tokens**: All tokens used
- **Prompt/Completion Tokens**: Detailed breakdown
- **Requests**: Number of successful API calls
- **Cost**: Estimated USD cost
- View the summary when exiting the app.

## How It Works

1. Downloads documents from S3 to a temp directory.
2. Processes files into chunks using `RecursiveCharacterTextSplitter`.
3. Indexes chunks with FAISS for efficient retrieval.
4. Answers queries via a RAG pipeline with your chosen language model.

## Contributing

1. Fork the repo: `https://github.com/Butterski/llkms`
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Submit a Pull Request.

## Acknowledgments

- **[LangChain](https://pypi.org/project/langchain/)**: RAG and document processing framework
- **[OpenAI](https://openai.com/)**: Optional LLM provider
- **[DeepSeek](https://api-docs.deepseek.com/)**: Default LLM provider
- **[FAISS](https://github.com/facebookresearch/faiss)**: Vector storage
- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)**: Image text extraction
- **[Questionary](https://pypi.org/project/questionary/)**: Interactive CLI

## Resources

- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Amazon S3 Docs](https://docs.aws.amazon.com/s3/index.html)