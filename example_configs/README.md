# Example Configurations for LLKMS

This directory contains example configuration files for different deployment scenarios.

## Available Configurations

### `config_local_ollama.yaml`
**Full Local Setup with Ollama**
- LLM: `bielik:7b` (Polish language model via Ollama)
- Embeddings: `nomic-embed-text` (Ollama)
- Cost: Free (runs locally)
- Requirements: Ollama installed with models pulled

### `config_local_bielik.yaml`
**Polish Language Model Setup**
- LLM: `bielik:latest` (Polish language model)
- Embeddings: `mxbai-embed-large` (higher quality)
- Cost: Free (runs locally)
- Requirements: Ollama with bielik and mxbai-embed-large

### `config_cloud_deepseek.yaml`
**Full Cloud with DeepSeek**
- LLM: `deepseek-chat` (DeepSeek API)
- Embeddings: `text-embedding-3-small` (OpenAI)
- Cost: Pay per token
- Requirements: DeepSeek API key, OpenAI API key

### `config_cloud_openai.yaml`
**Full Cloud with OpenAI**
- LLM: `gpt-4o-mini` (OpenAI)
- Embeddings: `text-embedding-3-small` (OpenAI)
- Cost: Pay per token
- Requirements: OpenAI API key

### `config_hybrid_local_llm.yaml`
**Hybrid: Local LLM + Cloud Embeddings**
- LLM: `llama3.2` (Ollama - local)
- Embeddings: `text-embedding-3-small` (OpenAI - cloud)
- Cost: Only embedding costs
- Requirements: Ollama with llama3.2, OpenAI API key

## Usage

Copy the desired configuration to the project root and rename to `config.yaml`:

```powershell
Copy-Item example_configs\config_local_ollama.yaml config.yaml
```

Or specify directly when running:

```powershell
python -m llkms.main --config example_configs\config_local_ollama.yaml
```

## Required Environment Variables

All configurations require AWS credentials:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

Cloud configurations additionally require:
```
DEEPSEEK_API_KEY=your_deepseek_key  # For DeepSeek
OPENAI_API_KEY=your_openai_key      # For OpenAI
```

## Ollama Setup

For local configurations, install Ollama and pull required models:

```bash
# Install Ollama from https://ollama.ai

# Pull LLM models
ollama pull bielik:7b        # Polish language model
ollama pull llama3.2         # General purpose

# Pull embedding models
ollama pull nomic-embed-text     # Fast, good quality
ollama pull mxbai-embed-large    # Higher quality, larger

# Verify models are available
ollama list
```

## RAG Configuration

All configurations support a `rag` section to tune retrieval:

```yaml
rag:
  retriever_k: 8  # Number of documents to retrieve for context (default: 8)
```

Increase `retriever_k` if the model needs more context to answer questions accurately.

## Notes

- Vector store cache is NOT compatible between different embedding models
- When switching embedding providers, delete `vector_store_cache/` and run with `--reindex`
- Ollama must be running before starting the application
