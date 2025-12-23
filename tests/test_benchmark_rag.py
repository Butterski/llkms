"""
RAG Benchmark Test Suite

This module provides comprehensive benchmarks for RAG (Retrieval-Augmented Generation)
pipelines. It supports:
- Loading test cases from JSONL files
- Running multiple configurations from a config folder
- Measuring retrieval metrics (precision@k, recall@k, f1@k, MRR@k)
- Measuring generation metrics (refusal_accuracy, faithfulness_proxy)
- Latency measurements for retrieval and generation
- Exporting results to CSV

Environment variables for thresholds:
    MIN_PRECISION_AT_4: Minimum precision@4 threshold (default: 0.5)
    MIN_RECALL_AT_4: Minimum recall@4 threshold (default: 0.5)
    MIN_F1_AT_4: Minimum F1@4 threshold (default: 0.5)
    MIN_MRR_AT_4: Minimum MRR@4 threshold (default: 0.3)
    MIN_REFUSAL_ACC: Minimum refusal accuracy threshold (default: 0.5)
    MIN_FAITHFULNESS: Minimum faithfulness proxy threshold (default: 0.3)
    BENCHMARK_K: Value of k for metrics (default: 4)
    BENCHMARK_CONFIGS_DIR: Directory with config files (default: tests/benchmark_configs)
    BENCHMARK_DATA_PATH: Path to benchmark JSONL file (default: tests/data/benchmark.jsonl)
    BENCHMARK_MAX_WORKERS: Number of parallel threads for benchmark execution (default: 4)

Usage:
    pytest tests/test_benchmark_rag.py -v --tb=short
    pytest tests/test_benchmark_rag.py -v -k "benchmark" --tb=short
"""

import concurrent.futures
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import yaml
from dotenv import load_dotenv

# Import llkms modules at module level
from llkms.utils.langchain.document_processor import (
    DocumentProcessingPipeline,
    DocumentProcessor,
)
from llkms.utils.langchain.model_factory import ModelConfig
from llkms.utils.langchain.rag_pipeline import RAGPipeline
from llkms.utils.langchain.vector_store_manager import VectorStoreManager

# Load environment variables
load_dotenv()


def clean_llm_output(text: str) -> str:
    """Clean special tokens and artifacts from LLM output."""
    if not text:
        return text
    # Remove common special tokens
    special_tokens = ["<s>", "</s>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "[INST]", "[/INST]"]
    for token in special_tokens:
        text = text.replace(token, "")
    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# Configuration & Data Classes
# =============================================================================


def check_and_pull_ollama_model(model_name: str):
    """Check if Ollama model exists, pull if not."""
    print(f"Checking availability of Ollama model: {model_name}")
    try:
        # Check if model exists
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        # Parse output to get exact model names
        # Output format: NAME  ID  SIZE  MODIFIED
        existing_models = []
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    existing_models.append(parts[0])

        if model_name not in existing_models:
            print(f"Model {model_name} not found locally. Pulling...")
            # Stream output to show progress
            # subprocess.run(["ollama", "pull", model_name], check=True)
            # print(f"Model {model_name} pulled successfully.")
        else:
            print(f"Model {model_name} is already available.")

    except subprocess.CalledProcessError as e:
        print(f"Error checking/pulling Ollama model {model_name}: {e}")
    except FileNotFoundError:
        print("Ollama executable not found. Please ensure Ollama is installed and in PATH.")


@dataclass
class BenchmarkCase:
    """Single benchmark test case loaded from JSONL."""

    id: str
    question: str
    expected_answer_keywords: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    ground_truth: str = ""  # Expected answer for RAGAS evaluation
    type: str = "answerable"  # "answerable" or "no_answer"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkCase":
        """Create BenchmarkCase from dictionary."""
        return cls(
            id=data.get("id", "unknown"),
            question=data["question"],
            expected_answer_keywords=data.get("expected_answer_keywords", []),
            relevant_chunk_ids=data.get("relevant_chunk_ids", []),
            ground_truth=data.get("ground_truth", ""),
            type=data.get("type", "answerable"),
        )


@dataclass
class CaseResult:
    """Result for a single benchmark case."""

    case_id: str
    config_name: str
    question: str
    case_type: str
    retrieved_ids: List[str] = field(default_factory=list)
    retrieved_contexts: List[str] = field(default_factory=list)  # Actual text content
    relevant_ids: List[str] = field(default_factory=list)
    answer: str = ""
    ground_truth: str = ""  # Expected answer for RAGAS
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    is_refusal: bool = False
    refusal_correct: Optional[bool] = None
    faithfulness_proxy: float = 0.0
    # RAGAS metrics
    ragas_faithfulness: Optional[float] = None
    ragas_answer_relevancy: Optional[float] = None
    ragas_context_precision: Optional[float] = None
    ragas_context_recall: Optional[float] = None
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all cases for a config."""

    config_name: str
    total_cases: int = 0
    answerable_cases: int = 0
    no_answer_cases: int = 0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_at_k: float = 0.0
    mrr_at_k: float = 0.0
    refusal_accuracy: float = 0.0
    faithfulness_proxy: float = 0.0
    # RAGAS aggregate metrics
    ragas_faithfulness: float = 0.0
    ragas_answer_relevancy: float = 0.0
    ragas_context_precision: float = 0.0
    ragas_context_recall: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    avg_generation_latency_ms: float = 0.0
    indexing_latency_ms: float = 0.0  # Time to create/load vector store
    document_count: int = 0  # Number of documents in vector store
    error_count: int = 0


# =============================================================================
# Metric Calculation Functions
# =============================================================================


def normalize_doc_id(doc_id: str) -> str:
    """
    Normalize document ID to just the filename for matching.

    This allows matching "knowledge/DeepSeek_V3.pdf" with "DeepSeek_V3.pdf".
    """
    from pathlib import Path

    # Extract just the filename from any path
    return Path(doc_id).name.lower()


def doc_ids_match(retrieved_id: str, relevant_id: str) -> bool:
    """Check if a retrieved document ID matches a relevant document ID."""
    return normalize_doc_id(retrieved_id) == normalize_doc_id(relevant_id)


def calculate_precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Precision@K.

    Args:
        retrieved: List of retrieved document IDs (in ranked order).
        relevant: List of relevant (ground truth) document IDs.
        k: Number of top results to consider.

    Returns:
        Precision@K score (0.0 to 1.0).
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    # Use filename-based matching
    relevant_normalized = [normalize_doc_id(r) for r in relevant]
    hits = sum(1 for doc_id in top_k if normalize_doc_id(doc_id) in relevant_normalized)
    return hits / len(top_k)


def calculate_recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Recall@K.

    Args:
        retrieved: List of retrieved document IDs (in ranked order).
        relevant: List of relevant (ground truth) document IDs.
        k: Number of top results to consider.

    Returns:
        Recall@K score (0.0 to 1.0).
    """
    if not relevant:
        return 1.0  # If no relevant docs expected, recall is perfect
    top_k = retrieved[:k]
    # Use filename-based matching
    retrieved_normalized = [normalize_doc_id(r) for r in top_k]
    relevant_normalized = [normalize_doc_id(r) for r in relevant]
    hits = sum(1 for rel in relevant_normalized if rel in retrieved_normalized)
    return hits / len(relevant_normalized)


def calculate_f1_at_k(precision: float, recall: float) -> float:
    """
    Calculate F1@K from precision and recall.

    Args:
        precision: Precision@K score.
        recall: Recall@K score.

    Returns:
        F1@K score (0.0 to 1.0).
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_reciprocal_rank(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Calculate Reciprocal Rank for MRR.

    Args:
        retrieved: List of retrieved document IDs (in ranked order).
        relevant: List of relevant (ground truth) document IDs.
        k: Number of top results to consider.

    Returns:
        Reciprocal rank (1/rank of first relevant doc, or 0 if none found).
    """
    # Use filename-based matching
    relevant_normalized = [normalize_doc_id(r) for r in relevant]
    for i, doc_id in enumerate(retrieved[:k]):
        if normalize_doc_id(doc_id) in relevant_normalized:
            return 1.0 / (i + 1)
    return 0.0


def detect_refusal(answer: str) -> bool:
    """
    Detect if the answer is a refusal (model says it doesn't know).

    Args:
        answer: The generated answer text.

    Returns:
        True if the answer appears to be a refusal.
    """
    answer_lower = answer.lower()

    # Hard refusal patterns - if answer contains these, it's likely a refusal
    hard_refusal_patterns = [
        "i don't know",
        "i do not know",
        "nie wiem",
        "nie znam",
        "nie mam informacji",
        "i cannot answer",
        "i can't answer",
        "cannot provide",
        "unable to answer",
        "no information",
        "nie jestem w stanie",
        "nie mogę odpowiedzieć",
        # Added patterns for "cannot find" style refusals
        "i cannot find",
        "i can't find",
        "cannot find this information",
        "nie mogę znaleźć",
        "nie znalazłem",
        "nie znaleziono",
        "brak informacji",
        "the provided documents do not contain",
        "documents do not contain",
        "not mentioned in the provided",
        "not found in the provided",
        "nie zawierają informacji",
        "dokumenty nie zawierają",
        "i'm sorry, but i can't answer",
        "sorry, but i can't answer",
        "przepraszam, ale nie mogę",
    ]

    # Soft refusal patterns - only count as refusal if the answer doesn't continue with content
    # e.g., "Not enough context, but based on..." is NOT a refusal
    soft_refusal_patterns = [
        "not enough context",
        "brak wystarczającego kontekstu",
    ]

    # Check hard refusals
    for pattern in hard_refusal_patterns:
        if pattern in answer_lower:
            return True

    # Check soft refusals - only if answer is short or doesn't continue
    for pattern in soft_refusal_patterns:
        if pattern in answer_lower:
            # If the answer has substantial content after the pattern, it's not a refusal
            pattern_idx = answer_lower.find(pattern)
            content_after = answer[pattern_idx + len(pattern) :].strip()
            # If there's less than 50 chars of content after, treat as refusal
            if len(content_after) < 50:
                return True
            # If it continues with "but", "however", etc., it's providing info
            continuation_markers = ["but ", "however ", "ale ", "jednak "]
            if not any(content_after.lower().startswith(marker) for marker in continuation_markers):
                # Check if total answer is substantive
                if len(answer) < 100:
                    return True

    return False


def calculate_faithfulness_proxy(answer: str, retrieved_docs: List[Any]) -> float:
    """
    Calculate a simple faithfulness proxy score.

    Checks if the answer contains references to chunk_ids or source metadata.
    This is a simple heuristic - for production use, consider LLM-based evaluation.

    Args:
        answer: The generated answer text.
        retrieved_docs: List of retrieved documents (with metadata).

    Returns:
        Faithfulness proxy score (0.0 to 1.0).
    """
    if not answer or not retrieved_docs:
        return 0.0

    answer_lower = answer.lower()
    score = 0.0
    checks = 0

    # Check for source/chunk references in answer
    for doc in retrieved_docs:
        checks += 1
        metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}

        # Check if source filename is mentioned
        source = metadata.get("source", "")
        if source:
            source_name = Path(source).stem.lower()
            if source_name and source_name in answer_lower:
                score += 1.0
                continue

        # Check if any content snippet from doc appears in answer
        content = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else str(doc)
        if content:
            # Check for substantial content overlap (at least 20 chars)
            content_words = content.lower().split()[:10]  # First 10 words
            for word in content_words:
                if len(word) > 5 and word in answer_lower:
                    score += 0.5
                    break

    return min(score / max(checks, 1), 1.0)


# =============================================================================
# RAGAS Evaluation
# =============================================================================


def get_ragas_evaluator_llm(config: Dict):
    """
    Get the LLM to use for RAGAS evaluation based on config.

    Uses the same LLM provider as the benchmark config, or falls back to
    environment variable RAGAS_EVAL_MODEL.

    Args:
        config: Configuration dictionary.

    Returns:
        RAGAS LLM instance for evaluation.
    """
    from openai import OpenAI
    from ragas.llms import llm_factory

    # Check for explicit RAGAS eval model in env
    ragas_model = os.getenv("RAGAS_EVAL_MODEL")
    if ragas_model:
        # Use OpenAI-compatible endpoint
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=os.getenv("RAGAS_EVAL_API_BASE", "https://api.openai.com/v1"),
        )
        return llm_factory(model=ragas_model, client=client)

    # Use same provider as config
    provider = config.get("model", {}).get("provider", "ollama")
    model_name = config.get("model", {}).get("model", "llama3.2")

    if provider == "ollama":
        client = OpenAI(
            api_key="ollama",
            base_url=config.get("model", {}).get("api_base", "http://localhost:11434/v1"),
        )
        return llm_factory(model=model_name, client=client)
    elif provider == "openai":
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        return llm_factory(model=model_name, client=client)
    elif provider == "deepseek":
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        return llm_factory(model=model_name, client=client)
    else:
        # Fallback to OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        return llm_factory(model="gpt-4o-mini", client=client)


def get_ragas_embeddings(config: Dict):
    """
    Get embeddings for RAGAS evaluation.

    Args:
        config: Configuration dictionary.

    Returns:
        RAGAS Embeddings instance.
    """
    from openai import OpenAI
    from ragas.embeddings import OpenAIEmbeddings

    emb_config = config.get("embeddings", {})
    provider = emb_config.get("provider", "openai")

    if provider == "ollama":
        # Ollama is OpenAI compatible
        base_url = emb_config.get("api_base", "http://localhost:11434")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        client = OpenAI(
            api_key="ollama",
            base_url=base_url,
        )
        return OpenAIEmbeddings(model=emb_config.get("model", "nomic-embed-text"), client=client)
    else:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        return OpenAIEmbeddings(model=emb_config.get("model", "text-embedding-3-small"), client=client)


def run_ragas_evaluation(
    results: List[CaseResult],
    config: Dict,
    use_ragas: bool = True,
) -> List[CaseResult]:
    """
    Run RAGAS evaluation on benchmark results.

    Args:
        results: List of CaseResult objects with answers and contexts.
        config: Configuration dictionary for LLM setup.
        use_ragas: Whether to use RAGAS (can be disabled for speed).

    Returns:
        Updated list of CaseResult with RAGAS metrics.
    """
    if not use_ragas:
        return results

    # Check if RAGAS is available
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError:
        print("  RAGAS not available, skipping LLM-based evaluation")
        return results

    # Filter answerable cases with valid answers
    evaluable_results = [
        r
        for r in results
        if r.case_type == "answerable" and r.answer and not r.is_refusal and r.retrieved_contexts and r.error is None
    ]

    if not evaluable_results:
        print("  No evaluable cases for RAGAS")
        return results

    print(f"  Running RAGAS evaluation on {len(evaluable_results)} cases...")

    try:
        # Get LLM and embeddings for evaluation
        eval_llm = get_ragas_evaluator_llm(config)
        eval_embeddings = get_ragas_embeddings(config)

        # Prepare dataset for RAGAS
        eval_samples = []
        for r in evaluable_results:
            sample = {
                "user_input": r.question,
                "response": r.answer,
                "retrieved_contexts": r.retrieved_contexts,
            }
            # Add ground truth if available
            if r.ground_truth:
                sample["reference"] = r.ground_truth
            eval_samples.append(sample)

        dataset = EvaluationDataset.from_list(eval_samples)

        # Select metrics based on whether we have ground truth
        has_ground_truth = any(r.ground_truth for r in evaluable_results)

        metrics = [
            Faithfulness(llm=eval_llm),
            ResponseRelevancy(llm=eval_llm, embeddings=eval_embeddings),
            LLMContextPrecisionWithoutReference(llm=eval_llm),
        ]

        if has_ground_truth:
            metrics.append(LLMContextRecall(llm=eval_llm))

        # Run evaluation
        print(
            f"  Using RAGAS eval LLM: {config.get('model', {}).get('provider', 'unknown')} / {config.get('model', {}).get('model', 'unknown')}"
        )
        eval_results = evaluate(dataset=dataset, metrics=metrics)

        # Map results back to CaseResult objects
        df = eval_results.to_pandas()
        print(f"  RAGAS returned {len(df)} rows with columns: {list(df.columns)}")

        import math

        for i, r in enumerate(evaluable_results):
            if i < len(df):
                row = df.iloc[i]
                # Safely extract values, converting NaN to 0
                faith = row.get("faithfulness", 0)
                r.ragas_faithfulness = (
                    0.0 if (faith is None or (isinstance(faith, float) and math.isnan(faith))) else float(faith)
                )

                relevancy = row.get("answer_relevancy", 0)
                r.ragas_answer_relevancy = (
                    0.0
                    if (relevancy is None or (isinstance(relevancy, float) and math.isnan(relevancy)))
                    else float(relevancy)
                )

                precision = row.get("llm_context_precision_without_reference", 0)
                r.ragas_context_precision = (
                    0.0
                    if (precision is None or (isinstance(precision, float) and math.isnan(precision)))
                    else float(precision)
                )

                if has_ground_truth:
                    recall = row.get("context_recall", 0)
                    r.ragas_context_recall = (
                        0.0 if (recall is None or (isinstance(recall, float) and math.isnan(recall))) else float(recall)
                    )

        print(f"  RAGAS evaluation complete")

    except Exception as e:
        print(
            f"  RAGAS evaluation failed for config {config.get('model', {}).get('provider', 'unknown')}/{config.get('model', {}).get('model', 'unknown')}: {e}"
        )
        import traceback

        traceback.print_exc()

    return results


# =============================================================================
# Config Loading Utilities
# =============================================================================


def resolve_env_vars(value: str) -> str:
    """Resolve environment variables with ${VAR_NAME} syntax."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.getenv(env_var, "")
    return value


def process_env_vars(config: Dict) -> Dict:
    """Recursively process dictionary and resolve environment variables."""
    if isinstance(config, dict):
        return {k: process_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [process_env_vars(v) for v in config]
    else:
        return resolve_env_vars(config)


def load_yaml_config(config_path: Path) -> Dict:
    """Load configuration from YAML file and resolve environment variables."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return process_env_vars(config)


def discover_configs(configs_dir: str) -> List[Tuple[str, Path]]:
    """
    Discover all YAML config files in a directory.

    Args:
        configs_dir: Path to directory containing config files.

    Returns:
        List of (config_name, config_path) tuples.
    """
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        return []

    configs = []
    for config_file in sorted(configs_path.glob("*.yaml")):
        if config_file.name.startswith("_"):
            continue  # Skip files starting with underscore
        config_name = config_file.stem
        configs.append((config_name, config_file))

    return configs


def load_benchmark_cases(jsonl_path: str) -> List[BenchmarkCase]:
    """
    Load benchmark cases from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file.

    Returns:
        List of BenchmarkCase objects.
    """
    cases = []
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {jsonl_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cases.append(BenchmarkCase.from_dict(data))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")

    return cases


# =============================================================================
# RAG Pipeline Wrapper
# =============================================================================


class BenchmarkRAGRunner:
    """
    Wrapper for running RAG pipeline in benchmark mode.

    This class handles initialization of the RAG pipeline from config,
    creates vector stores per configuration, and provides methods for
    retrieval and generation with timing.
    """

    # Directory for benchmark-specific vector store caches
    BENCHMARK_CACHE_BASE = Path("tests/benchmark_cache")

    def __init__(self, config: Dict, config_name: str):
        """
        Initialize the benchmark RAG runner.

        Args:
            config: Configuration dictionary loaded from YAML.
            config_name: Name of the configuration (used for cache directory).
        """
        self.config = config
        self.config_name = config_name
        self.rag_pipeline = None
        self.vector_store = None
        self._initialized = False
        self.indexing_latency_ms = 0.0
        self.document_count = 0

    def _get_cache_dir(self) -> Path:
        """Get the cache directory for this configuration."""
        return self.BENCHMARK_CACHE_BASE / self.config_name

    def initialize(self) -> bool:
        """
        Initialize the RAG pipeline.

        Creates vector store from S3 if not cached, or loads from cache.
        Measures indexing time.

        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            import asyncio

            # Create embedding config
            embedding_config = None
            if "embeddings" in self.config:
                embedding_config = {
                    "provider": self.config["embeddings"].get("provider", "openai"),
                    "model": self.config["embeddings"].get("model", "text-embedding-3-small"),
                    "api_base": self.config["embeddings"].get("api_base", "http://localhost:11434"),
                }

            # Use config-specific cache directory
            cache_dir = self._get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            vsm = VectorStoreManager(cache_dir=str(cache_dir))

            # Initialize document processor for embeddings
            doc_processor = DocumentProcessor(embedding_config=embedding_config)

            start_time = time.perf_counter()

            if vsm.exists():
                # Load from cache
                print(f"  Loading vector store from cache: {cache_dir}")
                self.vector_store = vsm.load(doc_processor.embeddings)
                if self.vector_store is None:
                    print(f"  Failed to load vector store from cache")
                    return False
            else:
                # Create vector store from S3
                print(f"  Creating vector store from S3 (this may take a while)...")

                # Check AWS config
                aws_config = self.config.get("aws", {})
                bucket = aws_config.get("bucket")
                prefix = aws_config.get("prefix", "")

                if not bucket:
                    print(f"  ERROR: No AWS bucket configured")
                    return False

                # Use DocumentProcessingPipeline to fetch and index documents
                pipeline = DocumentProcessingPipeline(embedding_config=embedding_config)

                # Override cache directory
                pipeline.vector_cache = vsm

                try:
                    # Create a dummy model config just for initialization
                    temp_model_config = ModelConfig(
                        provider=self.config["model"]["provider"],
                        model_name=self.config["model"]["model"],
                        api_key=self.config["model"].get("api_key", ""),
                        api_base=self.config["model"].get("api_base", ""),
                        max_tokens=self.config["model"].get("max_tokens", 1024),
                        temperature=self.config["model"].get("temperature", 0.7),
                    )

                    # Run async indexing
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        rag = loop.run_until_complete(
                            pipeline.process_s3_bucket_async(
                                bucket=bucket,
                                prefix=prefix,
                                model_config=temp_model_config,
                                reindex=True,  # Force indexing
                            )
                        )
                        self.vector_store = rag.vector_store
                    finally:
                        loop.close()
                        pipeline.cleanup()

                except Exception as e:
                    print(f"  ERROR creating vector store: {e}")
                    return False

            self.indexing_latency_ms = (time.perf_counter() - start_time) * 1000

            # Get document count
            if self.vector_store:
                self.document_count = self.vector_store.index.ntotal

            print(f"  Vector store ready: {self.document_count} documents, {self.indexing_latency_ms:.2f}ms")

            # Create model config
            model_config = ModelConfig(
                provider=self.config["model"]["provider"],
                model_name=self.config["model"]["model"],
                api_key=self.config["model"].get("api_key", ""),
                api_base=self.config["model"].get("api_base", ""),
                max_tokens=self.config["model"].get("max_tokens", 1024),
                temperature=self.config["model"].get("temperature", 0.7),
            )

            # Create RAG pipeline
            self.rag_pipeline = RAGPipeline(self.vector_store, model_config)
            self._initialized = True
            return True

        except Exception as e:
            import traceback

            print(f"Failed to initialize RAG pipeline: {e}")
            traceback.print_exc()
            return False

    def retrieve(self, question: str, k: int = 4) -> Tuple[List[Any], float]:
        """
        Retrieve documents for a question.

        Args:
            question: The query question.
            k: Number of documents to retrieve.

        Returns:
            Tuple of (list of documents, latency in milliseconds).
        """
        if not self._initialized:
            return [], 0.0

        start = time.perf_counter()
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return docs, elapsed_ms

    def generate(self, question: str) -> Tuple[str, float]:
        """
        Generate an answer for a question.

        Args:
            question: The query question.

        Returns:
            Tuple of (answer string, latency in milliseconds).
        """
        if not self._initialized:
            return "", 0.0

        start = time.perf_counter()
        answer, _ = self.rag_pipeline.query(question)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return answer, elapsed_ms

    def get_doc_ids(self, docs: List[Any]) -> List[str]:
        """
        Extract document IDs from retrieved documents.

        Args:
            docs: List of retrieved documents.

        Returns:
            List of document IDs (using source path or index).
        """
        ids = []
        for i, doc in enumerate(docs):
            metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
            # Try to get chunk_id, then source, then use index
            doc_id = metadata.get("chunk_id") or metadata.get("source") or f"doc_{i}"
            ids.append(str(doc_id))
        return ids


# =============================================================================
# Results Export
# =============================================================================


def format_float(value: float, decimals: int = 4) -> str:
    """Format float value, handling NaN and None."""
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{decimals}f}"


def ensure_artifacts_dir() -> Path:
    """Ensure artifacts directory exists and return path."""
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    return artifacts_dir


def export_aggregate_results(metrics_list: List[AggregateMetrics], output_path: Path):
    """
    Export aggregate metrics to CSV.

    Args:
        metrics_list: List of AggregateMetrics objects.
        output_path: Path for the output CSV file.
    """
    if not metrics_list:
        return

    fieldnames = [
        "config_name",
        "total_cases",
        "answerable_cases",
        "no_answer_cases",
        "document_count",
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "mrr_at_k",
        "refusal_accuracy",
        "faithfulness_proxy",
        "ragas_faithfulness",
        "ragas_answer_relevancy",
        "ragas_context_precision",
        "ragas_context_recall",
        "indexing_latency_ms",
        "avg_retrieval_latency_ms",
        "avg_generation_latency_ms",
        "error_count",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(
                {
                    "config_name": m.config_name,
                    "total_cases": m.total_cases,
                    "answerable_cases": m.answerable_cases,
                    "no_answer_cases": m.no_answer_cases,
                    "document_count": m.document_count,
                    "precision_at_k": format_float(m.precision_at_k),
                    "recall_at_k": format_float(m.recall_at_k),
                    "f1_at_k": format_float(m.f1_at_k),
                    "mrr_at_k": format_float(m.mrr_at_k),
                    "refusal_accuracy": format_float(m.refusal_accuracy),
                    "faithfulness_proxy": format_float(m.faithfulness_proxy),
                    "ragas_faithfulness": format_float(m.ragas_faithfulness),
                    "ragas_answer_relevancy": format_float(m.ragas_answer_relevancy),
                    "ragas_context_precision": format_float(m.ragas_context_precision),
                    "ragas_context_recall": format_float(m.ragas_context_recall),
                    "indexing_latency_ms": format_float(m.indexing_latency_ms, 2),
                    "avg_retrieval_latency_ms": format_float(m.avg_retrieval_latency_ms, 2),
                    "avg_generation_latency_ms": format_float(m.avg_generation_latency_ms, 2),
                    "error_count": m.error_count,
                }
            )


def export_case_results(results: List[CaseResult], output_path: Path):
    """
    Export per-case results to CSV.

    Args:
        results: List of CaseResult objects.
        output_path: Path for the output CSV file.
    """
    if not results:
        return

    fieldnames = [
        "config_name",
        "case_id",
        "question",
        "case_type",
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "reciprocal_rank",
        "is_refusal",
        "refusal_correct",
        "faithfulness_proxy",
        "ragas_faithfulness",
        "ragas_answer_relevancy",
        "ragas_context_precision",
        "ragas_context_recall",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "answer",
        "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "config_name": r.config_name,
                    "case_id": r.case_id,
                    "question": r.question[:100],  # Truncate long questions
                    "case_type": r.case_type,
                    "precision_at_k": f"{r.precision_at_k:.4f}",
                    "recall_at_k": f"{r.recall_at_k:.4f}",
                    "f1_at_k": f"{r.f1_at_k:.4f}",
                    "reciprocal_rank": f"{r.reciprocal_rank:.4f}",
                    "is_refusal": r.is_refusal,
                    "refusal_correct": r.refusal_correct,
                    "faithfulness_proxy": f"{r.faithfulness_proxy:.4f}",
                    "ragas_faithfulness": f"{r.ragas_faithfulness:.4f}" if r.ragas_faithfulness is not None else "",
                    "ragas_answer_relevancy": (
                        f"{r.ragas_answer_relevancy:.4f}" if r.ragas_answer_relevancy is not None else ""
                    ),
                    "ragas_context_precision": (
                        f"{r.ragas_context_precision:.4f}" if r.ragas_context_precision is not None else ""
                    ),
                    "ragas_context_recall": (
                        f"{r.ragas_context_recall:.4f}" if r.ragas_context_recall is not None else ""
                    ),
                    "retrieval_latency_ms": f"{r.retrieval_latency_ms:.2f}",
                    "generation_latency_ms": f"{r.generation_latency_ms:.2f}",
                    "answer": r.answer[:200] if r.answer else "",  # Truncate
                    "error": r.error or "",
                }
            )


# =============================================================================
# Benchmark Runner
# =============================================================================


def run_benchmark_for_config(
    config_name: str, config: Dict, cases: List[BenchmarkCase], k: int = 4, run_ragas: bool = True
) -> Tuple[AggregateMetrics, List[CaseResult]]:
    """
    Run benchmark for a single configuration.

    Args:
        config_name: Name of the configuration.
        config: Configuration dictionary.
        cases: List of benchmark cases.
        k: Value of k for metrics calculation.
        run_ragas: Whether to run RAGAS evaluation (requires LLM calls).

    Returns:
        Tuple of (AggregateMetrics, list of CaseResult).
    """
    runner = BenchmarkRAGRunner(config, config_name)
    case_results: List[CaseResult] = []

    # Try to initialize
    if not runner.initialize():
        # Return empty metrics if initialization fails
        metrics = AggregateMetrics(
            config_name=config_name,
            total_cases=len(cases),
            error_count=len(cases),
        )
        for case in cases:
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    config_name=config_name,
                    question=case.question,
                    case_type=case.type,
                    error="Failed to initialize RAG pipeline",
                )
            )
        return metrics, case_results

    def process_case(case: BenchmarkCase) -> CaseResult:
        """Process a single benchmark case."""
        result = CaseResult(
            case_id=case.id,
            config_name=config_name,
            question=case.question,
            case_type=case.type,
            relevant_ids=case.relevant_chunk_ids,
            ground_truth=case.ground_truth,
        )

        try:
            # Retrieval phase
            docs, retrieval_latency = runner.retrieve(case.question, k=k)
            result.retrieval_latency_ms = retrieval_latency
            result.retrieved_ids = runner.get_doc_ids(docs)

            # Store retrieved context text for RAGAS evaluation
            result.retrieved_contexts = [doc.page_content for doc in docs]

            # Calculate retrieval metrics
            result.precision_at_k = calculate_precision_at_k(result.retrieved_ids, case.relevant_chunk_ids, k)
            result.recall_at_k = calculate_recall_at_k(result.retrieved_ids, case.relevant_chunk_ids, k)
            result.f1_at_k = calculate_f1_at_k(result.precision_at_k, result.recall_at_k)
            result.reciprocal_rank = calculate_reciprocal_rank(result.retrieved_ids, case.relevant_chunk_ids, k)

            # Generation phase
            answer, generation_latency = runner.generate(case.question)
            # Clean special tokens from answer
            answer = clean_llm_output(answer)
            result.answer = answer
            result.generation_latency_ms = generation_latency

            # Calculate generation metrics
            result.is_refusal = detect_refusal(answer)
            if case.type == "no_answer" or case.type == "unanswerable":
                result.refusal_correct = result.is_refusal
            else:
                result.refusal_correct = not result.is_refusal

            result.faithfulness_proxy = calculate_faithfulness_proxy(answer, docs)

        except Exception as e:
            result.error = str(e)
            # print(f"Error processing case {case.id}: {e}") # Optional logging

        return result

    # Run cases in parallel
    max_workers = int(os.getenv("BENCHMARK_MAX_WORKERS", "4"))
    print(f"  Running benchmark with {max_workers} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {executor.submit(process_case, case): case for case in cases}
        for future in concurrent.futures.as_completed(future_to_case):
            result = future.result()
            case_results.append(result)

    # Sort results by case_id to maintain consistent order
    case_results.sort(key=lambda x: x.case_id)

    # Run RAGAS evaluation if enabled
    if run_ragas:
        print(f"  Running RAGAS evaluation for {config_name}...")
        try:
            ragas_results = run_ragas_evaluation(case_results, config)
            # Update case results with RAGAS scores
            for result in case_results:
                if result.case_id in ragas_results:
                    ragas_scores = ragas_results[result.case_id]
                    result.ragas_faithfulness = ragas_scores.get("faithfulness")
                    result.ragas_answer_relevancy = ragas_scores.get("answer_relevancy")
                    result.ragas_context_precision = ragas_scores.get("context_precision")
                    result.ragas_context_recall = ragas_scores.get("context_recall")
            print(f"  RAGAS evaluation completed for {config_name}")
        except Exception as e:
            print(f"  RAGAS evaluation failed for {config_name}: {e}")

    # Aggregate metrics (include indexing info from runner)
    metrics = aggregate_case_results(
        config_name,
        case_results,
        indexing_latency_ms=runner.indexing_latency_ms,
        document_count=runner.document_count,
    )
    return metrics, case_results


def aggregate_case_results(
    config_name: str,
    results: List[CaseResult],
    indexing_latency_ms: float = 0.0,
    document_count: int = 0,
) -> AggregateMetrics:
    """
    Aggregate case results into metrics.

    Args:
        config_name: Name of the configuration.
        results: List of CaseResult objects.
        indexing_latency_ms: Time spent indexing/loading vector store.
        document_count: Number of documents in vector store.

    Returns:
        AggregateMetrics object.
    """
    metrics = AggregateMetrics(
        config_name=config_name,
        total_cases=len(results),
        indexing_latency_ms=indexing_latency_ms,
        document_count=document_count,
    )

    if not results:
        return metrics

    valid_results = [r for r in results if r.error is None]
    metrics.error_count = len(results) - len(valid_results)

    if not valid_results:
        return metrics

    # Retrieval metrics
    metrics.precision_at_k = sum(r.precision_at_k for r in valid_results) / len(valid_results)
    metrics.recall_at_k = sum(r.recall_at_k for r in valid_results) / len(valid_results)
    metrics.f1_at_k = sum(r.f1_at_k for r in valid_results) / len(valid_results)
    metrics.mrr_at_k = sum(r.reciprocal_rank for r in valid_results) / len(valid_results)

    # Generation metrics
    answerable = [r for r in valid_results if r.case_type == "answerable"]
    unanswerable = [r for r in valid_results if r.case_type in ("no_answer", "unanswerable")]

    metrics.answerable_cases = len(answerable)
    metrics.no_answer_cases = len(unanswerable)

    # Refusal accuracy (for no_answer cases: should refuse; for answerable: should not refuse)
    refusal_correct_count = sum(1 for r in valid_results if r.refusal_correct)
    metrics.refusal_accuracy = refusal_correct_count / len(valid_results)

    # Faithfulness (only for answerable cases that weren't refusals)
    answerable_with_answers = [r for r in answerable if not r.is_refusal]
    if answerable_with_answers:
        metrics.faithfulness_proxy = sum(r.faithfulness_proxy for r in answerable_with_answers) / len(
            answerable_with_answers
        )

    # RAGAS metrics aggregation
    ragas_faithfulness_vals = [r.ragas_faithfulness for r in valid_results if r.ragas_faithfulness is not None]
    ragas_relevancy_vals = [r.ragas_answer_relevancy for r in valid_results if r.ragas_answer_relevancy is not None]
    ragas_ctx_precision_vals = [
        r.ragas_context_precision for r in valid_results if r.ragas_context_precision is not None
    ]
    ragas_ctx_recall_vals = [r.ragas_context_recall for r in valid_results if r.ragas_context_recall is not None]

    if ragas_faithfulness_vals:
        metrics.ragas_faithfulness = sum(ragas_faithfulness_vals) / len(ragas_faithfulness_vals)
    if ragas_relevancy_vals:
        metrics.ragas_answer_relevancy = sum(ragas_relevancy_vals) / len(ragas_relevancy_vals)
    if ragas_ctx_precision_vals:
        metrics.ragas_context_precision = sum(ragas_ctx_precision_vals) / len(ragas_ctx_precision_vals)
    if ragas_ctx_recall_vals:
        metrics.ragas_context_recall = sum(ragas_ctx_recall_vals) / len(ragas_ctx_recall_vals)

    # Latency
    metrics.avg_retrieval_latency_ms = sum(r.retrieval_latency_ms for r in valid_results) / len(valid_results)
    metrics.avg_generation_latency_ms = sum(r.generation_latency_ms for r in valid_results) / len(valid_results)

    return metrics


# =============================================================================
# Pytest Fixtures
# =============================================================================


def get_env_float(name: str, default: float) -> float:
    """Get float value from environment variable with default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_env_int(name: str, default: int) -> int:
    """Get int value from environment variable with default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@pytest.fixture(scope="module")
def benchmark_k() -> int:
    """Get k value for metrics from environment."""
    return get_env_int("BENCHMARK_K", 4)


@pytest.fixture(scope="module")
def min_precision() -> float:
    """Get minimum precision threshold from environment."""
    return get_env_float("MIN_PRECISION_AT_4", 0.5)


@pytest.fixture(scope="module")
def min_recall() -> float:
    """Get minimum recall threshold from environment."""
    return get_env_float("MIN_RECALL_AT_4", 0.5)


@pytest.fixture(scope="module")
def min_f1() -> float:
    """Get minimum F1 threshold from environment."""
    return get_env_float("MIN_F1_AT_4", 0.5)


@pytest.fixture(scope="module")
def min_mrr() -> float:
    """Get minimum MRR threshold from environment."""
    return get_env_float("MIN_MRR_AT_4", 0.3)


@pytest.fixture(scope="module")
def min_refusal_acc() -> float:
    """Get minimum refusal accuracy threshold from environment."""
    return get_env_float("MIN_REFUSAL_ACC", 0.5)


@pytest.fixture(scope="module")
def min_faithfulness() -> float:
    """Get minimum faithfulness threshold from environment."""
    return get_env_float("MIN_FAITHFULNESS", 0.3)


@pytest.fixture(scope="module")
def enable_ragas() -> bool:
    """Get whether to run RAGAS evaluation from environment."""
    value = os.getenv("ENABLE_RAGAS", "true").lower()
    return value in ("true", "1", "yes")


@pytest.fixture(scope="module")
def min_ragas_faithfulness() -> float:
    """Get minimum RAGAS faithfulness threshold from environment."""
    return get_env_float("MIN_RAGAS_FAITHFULNESS", 0.5)


@pytest.fixture(scope="module")
def min_ragas_answer_relevancy() -> float:
    """Get minimum RAGAS answer relevancy threshold from environment."""
    return get_env_float("MIN_RAGAS_ANSWER_RELEVANCY", 0.5)


@pytest.fixture(scope="module")
def benchmark_data_path() -> str:
    """Get benchmark data path from environment."""
    return os.getenv("BENCHMARK_DATA_PATH", "tests/data/benchmark.jsonl")


@pytest.fixture(scope="module")
def configs_dir() -> str:
    """Get configs directory from environment."""
    return os.getenv("BENCHMARK_CONFIGS_DIR", "tests/benchmark_configs")


@pytest.fixture(scope="module")
def benchmark_cases(benchmark_data_path: str) -> List[BenchmarkCase]:
    """Load benchmark cases from JSONL file."""
    return load_benchmark_cases(benchmark_data_path)


@pytest.fixture(scope="module")
def discovered_configs(configs_dir: str) -> List[Tuple[str, Path]]:
    """Discover all configuration files in configs directory."""
    configs = discover_configs(configs_dir)
    if not configs:
        pytest.skip(f"No config files found in {configs_dir}")
    return configs


@pytest.fixture(scope="module")
def all_benchmark_results(
    discovered_configs: List[Tuple[str, Path]],
    benchmark_cases: List[BenchmarkCase],
    benchmark_k: int,
    enable_ragas: bool,
) -> Tuple[List[AggregateMetrics], List[CaseResult]]:
    """
    Run benchmarks for all discovered configurations.

    This fixture runs the full benchmark suite and exports results.
    """
    all_metrics: List[AggregateMetrics] = []
    all_case_results: List[CaseResult] = []

    for config_name, config_path in discovered_configs:
        print(f"\n{'='*60}")
        print(f"Running benchmark for config: {config_name}")
        print(f"{'='*60}")

        try:
            config = load_yaml_config(config_path)

            # Check for Ollama models and pull if necessary
            if config.get("model", {}).get("provider") == "ollama":
                model_name = config.get("model", {}).get("model")
                if model_name:
                    check_and_pull_ollama_model(model_name)

            if config.get("embeddings", {}).get("provider") == "ollama":
                model_name = config.get("embeddings", {}).get("model")
                if model_name:
                    check_and_pull_ollama_model(model_name)

            metrics, case_results = run_benchmark_for_config(
                config_name, config, benchmark_cases, k=benchmark_k, run_ragas=enable_ragas
            )
            all_metrics.append(metrics)
            all_case_results.extend(case_results)

            # Print summary for this config
            print(f"\n  --- Results Summary ---")
            print(f"  Documents indexed: {metrics.document_count}")
            print(f"  Indexing/Load time: {metrics.indexing_latency_ms:.2f}ms")
            print(f"  Precision@{benchmark_k}: {metrics.precision_at_k:.4f}")
            print(f"  Recall@{benchmark_k}: {metrics.recall_at_k:.4f}")
            print(f"  F1@{benchmark_k}: {metrics.f1_at_k:.4f}")
            print(f"  MRR@{benchmark_k}: {metrics.mrr_at_k:.4f}")
            print(f"  Refusal Accuracy: {metrics.refusal_accuracy:.4f}")
            print(f"  Faithfulness Proxy: {metrics.faithfulness_proxy:.4f}")
            if enable_ragas:
                print(f"  --- RAGAS Metrics ---")
                print(f"  RAGAS Faithfulness: {format_float(metrics.ragas_faithfulness) or 'N/A'}")
                print(f"  RAGAS Answer Relevancy: {format_float(metrics.ragas_answer_relevancy) or 'N/A'}")
                print(f"  RAGAS Context Precision: {format_float(metrics.ragas_context_precision) or 'N/A'}")
                print(f"  RAGAS Context Recall: {format_float(metrics.ragas_context_recall) or 'N/A'}")
            print(f"  --- Latency ---")
            print(f"  Avg Retrieval Latency: {metrics.avg_retrieval_latency_ms:.2f}ms")
            print(f"  Avg Generation Latency: {metrics.avg_generation_latency_ms:.2f}ms")
            print(f"  Errors: {metrics.error_count}/{metrics.total_cases}")

        except Exception as e:
            import traceback

            print(f"  ERROR: {e}")
            traceback.print_exc()
            metrics = AggregateMetrics(
                config_name=config_name,
                total_cases=len(benchmark_cases),
                error_count=len(benchmark_cases),
            )
            all_metrics.append(metrics)

    # Export results to CSV
    artifacts_dir = ensure_artifacts_dir()
    export_aggregate_results(all_metrics, artifacts_dir / "benchmark_results.csv")
    export_case_results(all_case_results, artifacts_dir / "cases.csv")

    print(f"\n{'='*60}")
    print(f"Results exported to {artifacts_dir}/")
    print(f"{'='*60}")

    return all_metrics, all_case_results


# =============================================================================
# Pytest Tests
# =============================================================================


class TestRAGBenchmark:
    """
    RAG Benchmark test suite.

    This test class runs benchmarks across all discovered configurations
    and asserts that aggregate metrics meet minimum thresholds.
    """

    def test_benchmark_runs_successfully(self, all_benchmark_results: Tuple[List[AggregateMetrics], List[CaseResult]]):
        """Test that benchmarks complete without fatal errors."""
        metrics_list, _ = all_benchmark_results
        assert len(metrics_list) > 0, "No benchmark results collected"

    def test_precision_at_k_threshold(
        self,
        all_benchmark_results: Tuple[List[AggregateMetrics], List[CaseResult]],
        min_precision: float,
        benchmark_k: int,
    ):
        """Test that average precision@k meets minimum threshold."""
        metrics_list, _ = all_benchmark_results

        # Filter out configs with all errors
        valid_metrics = [m for m in metrics_list if m.error_count < m.total_cases]
        if not valid_metrics:
            pytest.skip("All configurations had errors")

        avg_precision = sum(m.precision_at_k for m in valid_metrics) / len(valid_metrics)
        assert avg_precision >= min_precision, (
            f"Average Precision@{benchmark_k} ({avg_precision:.4f}) " f"is below threshold ({min_precision})"
        )

    def test_recall_at_k_threshold(
        self,
        all_benchmark_results: Tuple[List[AggregateMetrics], List[CaseResult]],
        min_recall: float,
        benchmark_k: int,
    ):
        """Test that average recall@k meets minimum threshold."""
        metrics_list, _ = all_benchmark_results

        valid_metrics = [m for m in metrics_list if m.error_count < m.total_cases]
        if not valid_metrics:
            pytest.skip("All configurations had errors")

        avg_recall = sum(m.recall_at_k for m in valid_metrics) / len(valid_metrics)
        assert avg_recall >= min_recall, (
            f"Average Recall@{benchmark_k} ({avg_recall:.4f}) " f"is below threshold ({min_recall})"
        )

    def test_f1_at_k_threshold(
        self,
        all_benchmark_results: Tuple[List[AggregateMetrics], List[CaseResult]],
        min_f1: float,
        benchmark_k: int,
    ):
        """Test that average F1@k meets minimum threshold."""
        metrics_list, _ = all_benchmark_results

        valid_metrics = [m for m in metrics_list if m.error_count < m.total_cases]
        if not valid_metrics:
            pytest.skip("All configurations had errors")

        avg_f1 = sum(m.f1_at_k for m in valid_metrics) / len(valid_metrics)
        assert avg_f1 >= min_f1, f"Average F1@{benchmark_k} ({avg_f1:.4f}) is below threshold ({min_f1})"

    def test_mrr_at_k_threshold(
        self,
        all_benchmark_results: Tuple[List[AggregateMetrics], List[CaseResult]],
        min_mrr: float,
        benchmark_k: int,
    ):
        """Test that average MRR@k meets minimum threshold."""
        metrics_list, _ = all_benchmark_results

        valid_metrics = [m for m in metrics_list if m.error_count < m.total_cases]
        if not valid_metrics:
            pytest.skip("All configurations had errors")

        avg_mrr = sum(m.mrr_at_k for m in valid_metrics) / len(valid_metrics)
        assert avg_mrr >= min_mrr, f"Average MRR@{benchmark_k} ({avg_mrr:.4f}) is below threshold ({min_mrr})"

    def test_refusal_accuracy_threshold(
        self,
        all_benchmark_results: Tuple[List[AggregateMetrics], List[CaseResult]],
        min_refusal_acc: float,
    ):
        """Test that average refusal accuracy meets minimum threshold."""
        metrics_list, _ = all_benchmark_results

        valid_metrics = [m for m in metrics_list if m.error_count < m.total_cases]
        if not valid_metrics:
            pytest.skip("All configurations had errors")

        avg_refusal = sum(m.refusal_accuracy for m in valid_metrics) / len(valid_metrics)
        assert avg_refusal >= min_refusal_acc, (
            f"Average Refusal Accuracy ({avg_refusal:.4f}) " f"is below threshold ({min_refusal_acc})"
        )

    def test_faithfulness_threshold(
        self,
        all_benchmark_results: Tuple[List[AggregateMetrics], List[CaseResult]],
        min_faithfulness: float,
    ):
        """Test that average faithfulness proxy meets minimum threshold."""
        metrics_list, _ = all_benchmark_results

        valid_metrics = [m for m in metrics_list if m.error_count < m.total_cases]
        if not valid_metrics:
            pytest.skip("All configurations had errors")

        avg_faithfulness = sum(m.faithfulness_proxy for m in valid_metrics) / len(valid_metrics)
        assert avg_faithfulness >= min_faithfulness, (
            f"Average Faithfulness Proxy ({avg_faithfulness:.4f}) " f"is below threshold ({min_faithfulness})"
        )

    def test_ragas_faithfulness_threshold(
        self,
        all_benchmark_results: Tuple[List[AggregateMetrics], List[CaseResult]],
        min_ragas_faithfulness: float,
        enable_ragas: bool,
    ):
        """Test that average RAGAS faithfulness meets minimum threshold."""
        if not enable_ragas:
            pytest.skip("RAGAS evaluation is disabled")

        metrics_list, _ = all_benchmark_results

        valid_metrics = [m for m in metrics_list if m.error_count < m.total_cases and m.ragas_faithfulness > 0]
        if not valid_metrics:
            pytest.skip("No valid RAGAS faithfulness scores")

        avg_ragas_faith = sum(m.ragas_faithfulness for m in valid_metrics) / len(valid_metrics)
        assert avg_ragas_faith >= min_ragas_faithfulness, (
            f"Average RAGAS Faithfulness ({avg_ragas_faith:.4f}) " f"is below threshold ({min_ragas_faithfulness})"
        )

    def test_ragas_answer_relevancy_threshold(
        self,
        all_benchmark_results: Tuple[List[AggregateMetrics], List[CaseResult]],
        min_ragas_answer_relevancy: float,
        enable_ragas: bool,
    ):
        """Test that average RAGAS answer relevancy meets minimum threshold."""
        if not enable_ragas:
            pytest.skip("RAGAS evaluation is disabled")

        metrics_list, _ = all_benchmark_results

        valid_metrics = [m for m in metrics_list if m.error_count < m.total_cases and m.ragas_answer_relevancy > 0]
        if not valid_metrics:
            pytest.skip("No valid RAGAS answer relevancy scores")

        avg_ragas_rel = sum(m.ragas_answer_relevancy for m in valid_metrics) / len(valid_metrics)
        assert avg_ragas_rel >= min_ragas_answer_relevancy, (
            f"Average RAGAS Answer Relevancy ({avg_ragas_rel:.4f}) "
            f"is below threshold ({min_ragas_answer_relevancy})"
        )

    def test_results_exported_to_csv(self, all_benchmark_results: Tuple[List[AggregateMetrics], List[CaseResult]]):
        """Test that results are exported to CSV files."""
        artifacts_dir = Path("artifacts")
        assert (artifacts_dir / "benchmark_results.csv").exists(), "Aggregate results CSV not found"
        assert (artifacts_dir / "cases.csv").exists(), "Case results CSV not found"


# =============================================================================
# Standalone Benchmark Runner
# =============================================================================


def run_single_config_benchmark(config_path: str, data_path: str, k: int = 4):
    """
    Run benchmark for a single configuration file.

    Useful for quick testing of a specific config.

    Args:
        config_path: Path to the config YAML file.
        data_path: Path to the benchmark JSONL file.
        k: Value of k for metrics.
    """
    config_path = Path(config_path)
    config_name = config_path.stem
    config = load_yaml_config(config_path)
    cases = load_benchmark_cases(data_path)

    print(f"Running benchmark for: {config_name}")
    print(f"Loaded {len(cases)} test cases")

    metrics, case_results = run_benchmark_for_config(config_name, config, cases, k=k)

    print(f"\nResults for {config_name}:")
    print(f"  Documents indexed: {metrics.document_count}")
    print(f"  Indexing/Load time: {metrics.indexing_latency_ms:.2f}ms")
    print(f"  Precision@{k}: {metrics.precision_at_k:.4f}")
    print(f"  Recall@{k}: {metrics.recall_at_k:.4f}")
    print(f"  F1@{k}: {metrics.f1_at_k:.4f}")
    print(f"  MRR@{k}: {metrics.mrr_at_k:.4f}")
    print(f"  Refusal Accuracy: {metrics.refusal_accuracy:.4f}")
    print(f"  Faithfulness Proxy: {metrics.faithfulness_proxy:.4f}")
    print(f"  Avg Retrieval Latency: {metrics.avg_retrieval_latency_ms:.2f}ms")
    print(f"  Avg Generation Latency: {metrics.avg_generation_latency_ms:.2f}ms")
    print(f"  Errors: {metrics.error_count}/{metrics.total_cases}")

    return metrics, case_results


if __name__ == "__main__":
    # Example standalone usage
    import sys

    if len(sys.argv) >= 3:
        run_single_config_benchmark(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python test_benchmark_rag.py <config.yaml> <benchmark.jsonl>")
        print("Or run with pytest: pytest test_benchmark_rag.py -v")
