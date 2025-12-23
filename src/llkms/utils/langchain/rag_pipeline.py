import re
from typing import Any, Dict, List, Tuple

from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from llkms.utils.langchain.model_factory import ModelConfig, ModelFactory
from llkms.utils.logger import logger


def clean_llm_output(text: str) -> str:
    """Clean special tokens and artifacts from LLM output."""
    # Remove common special tokens
    special_tokens = [
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "[INST]",
        "[/INST]",
        "<<SYS>>",
        "<</SYS>>",
    ]
    for token in special_tokens:
        text = text.replace(token, "")
    # Remove any remaining angle bracket tokens like <|...|>
    text = re.sub(r"<\|[^|]+\|>", "", text)

    # Extract content from XML-like tags (for Bielik responses)
    # Match <odpowiedz>...</odpowiedz> or just content after <odpowiedz>
    odpowiedz_match = re.search(r"<odpowiedz>\s*(.*?)\s*(?:</odpowiedz>|$)", text, re.DOTALL)
    if odpowiedz_match:
        text = odpowiedz_match.group(1)

    # Remove other XML-like tags that might be in response
    text = re.sub(r"</?(?:tresc|cytaty|odpowiedz)>", "", text)

    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_docs_with_sources(docs: List[Document]) -> str:
    """Format retrieved documents with source attribution for better grounding."""
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        # Extract just filename from path
        source_name = source.split("/")[-1].split("\\")[-1] if source else "Unknown"
        content = doc.page_content.strip()
        formatted_parts.append(f"[Source {i}: {source_name}]\n{content}")
    return "\n\n".join(formatted_parts)


class CleanOutputParser(StrOutputParser):
    """Custom output parser that cleans special tokens from LLM output."""

    def parse(self, text: str) -> str:
        return clean_llm_output(super().parse(text))


class RAGPipeline:
    # Prompt templates for different languages - designed for strict context adherence
    PROMPT_EN = """You are a document search assistant. Your ONLY task is to find information in the provided documents.

=== ABSOLUTE RULES (NEVER BREAK THESE) ===

❌ FORBIDDEN:
- Using your general knowledge
- Answering questions not covered in context
- Guessing or assuming anything
- Providing definitions of things not described in documents
- Saying "I know that..." if it's not in the documents

✅ REQUIRED:
- Answer ONLY based on the documents below
- If information is NOT in documents, respond EXACTLY:
  "I cannot find this information in the provided documents."
- Be concise (1-3 sentences)

=== DOCUMENTS TO SEARCH ===
{context}

=== QUESTION ===
{question}

=== YOUR ANSWER (only from documents above, if no info = refuse) ==="""

    PROMPT_PL = """Jesteś asystentem wyszukiwania dokumentów. Twoim JEDYNYM zadaniem jest znajdowanie informacji w podanych dokumentach.

=== ABSOLUTNE ZASADY (NIGDY ICH NIE ŁAM) ===

❌ ZAKAZANE:
- Używanie własnej wiedzy ogólnej
- Odpowiadanie na pytania spoza kontekstu
- Zgadywanie lub domyślanie się
- Podawanie definicji rzeczy nieopisanych w dokumentach
- Mówienie "wiem że..." jeśli tego nie ma w dokumentach

✅ WYMAGANE:
- Odpowiadaj WYŁĄCZNIE na podstawie poniższych dokumentów
- Jeśli informacji NIE MA w dokumentach, odpowiedz DOKŁADNIE:
  "Nie mogę znaleźć tej informacji w dostarczonych dokumentach."
- Bądź zwięzły (1-3 zdania)

=== DOKUMENTY DO PRZESZUKANIA ===
{context}

=== PYTANIE ===
{question}

=== TWOJA ODPOWIEDŹ (tylko z dokumentów powyżej, jeśli brak info = odmów) ==="""

    # Specjalny prompt dla Bielika z XML-like strukturą - lepiej przestrzega zasad
    PROMPT_BIELIK = """[INSTRUKCJA SYSTEMOWA]
JESTEŚ WERYFIKATOREM DOKUMENTÓW. CAŁKOWICIE IGNORUJ SWOJĄ WIEDZĘ OGÓLNĄ.
ANALIZUJESZ TYLKO dokumenty podane w sekcji [KONTEKST].
ODPOWIADAJ TYLKO PO POLSKU.

PROCEDURA:
KROK 1: Przeczytaj pytanie w [PYTANIE]
KROK 2: Sprawdź czy odpowiedź znajduje się w [KONTEKST]
KROK 3A: Jeśli TAK - odpowiedz cytując fragmenty z dokumentów
KROK 3B: Jeśli NIE - odpowiedz TYLKO: "Nie mogę znaleźć tej informacji w dostarczonych dokumentach."

ZAKAZY BEZWZGLĘDNE:
- NIE używaj wiedzy spoza [KONTEKST]
- NIE definiuj pojęć których nie ma w dokumentach
- NIE zgaduj ani nie domyślaj się
- NIE mów "wiem że..." jeśli tego nie ma w [KONTEKST]

[KONTEKST]
{context}
[/KONTEKST]

[PYTANIE]
{question}
[/PYTANIE]

[ODPOWIEDŹ]
<odpowiedz>"""

    def __init__(self, vector_store: FAISS, model_config: ModelConfig, retriever_k: int = 8):
        """
        Initialize the RAGPipeline.

        Args:
            vector_store (FAISS): The vector store instance.
            model_config (ModelConfig): Configuration for the language model.
            retriever_k (int): Number of documents to retrieve for context.
        """
        self.vector_store = vector_store
        self.llm = ModelFactory.create_model(model_config)
        self.retriever_k = retriever_k

        # Select prompt template based on model
        is_bielik = "bielik" in model_config.model_name.lower()

        if is_bielik:
            # Specjalny XML-like prompt dla Bielika - lepiej przestrzega zasad
            prompt_template = self.PROMPT_BIELIK
            logger.info("Using specialized XML-like prompt template for Bielik model")
        else:
            prompt_template = self.PROMPT_EN

        logger.info(f"RAG retriever configured with k={retriever_k}")

        # Define RAG prompt
        self.prompt = PromptTemplate.from_template(prompt_template)

        # Build the RAG chain with configurable k and formatted context
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": retriever_k})

        # Chain with source-attributed context formatting
        self.chain = (
            {"context": self.retriever | RunnableLambda(format_docs_with_sources), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | CleanOutputParser()
        )

    def get_retrieved_docs(self, question: str):
        """
        Retrieve the documents used as context for the given question.

        Args:
            question (str): The user's question.

        Returns:
            List[Document]: List of retrieved documents with metadata.
        """
        return self.retriever.invoke(question)

    def query(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Execute the RAG query.

        Args:
            question (str): The user's question.

        Returns:
            Tuple[str, Dict[str, Any]]: The answer and token usage details.
        """
        with get_openai_callback() as cb:
            response = self.chain.invoke(question)
            logger.debug(f"Query tokens - Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens}")
            return response, {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": cb.total_cost,
                "successful_requests": cb.successful_requests,
            }
