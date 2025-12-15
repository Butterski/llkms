from typing import Any, Dict, Tuple

from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from llkms.utils.langchain.model_factory import ModelConfig, ModelFactory
from llkms.utils.logger import logger


class RAGPipeline:
    # Prompt templates for different languages
    PROMPT_EN = """
        Answer the question based on the following context. If you don't know 
        the answer, just say you don't know. Use three sentences maximum.
                                                   
        If the context is not enough to answer the question, say so and if you know the answer start message with 'Not enough context, but...'
        
        Context: {context}
        Question: {question}
        
        Answer:"""

    PROMPT_PL = """
        Odpowiedz na pytanie na podstawie poniższego kontekstu. Jeśli nie znasz 
        odpowiedzi, po prostu powiedz, że nie wiesz. Użyj maksymalnie trzech zdań.
                                                   
        Jeśli kontekst nie jest wystarczający do odpowiedzi na pytanie, powiedz o tym, a jeśli znasz odpowiedź, rozpocznij wiadomość od 'Brak wystarczającego kontekstu, ale...'
        
        Kontekst: {context}
        Pytanie: {question}
        
        Odpowiedź:"""

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

        # Use Polish prompt for Bielik model
        is_polish_model = "bielik" in model_config.model_name.lower()
        prompt_template = self.PROMPT_PL if is_polish_model else self.PROMPT_EN

        if is_polish_model:
            logger.info("Using Polish prompt template for Bielik model")

        logger.info(f"RAG retriever configured with k={retriever_k}")

        # Define RAG prompt
        self.prompt = PromptTemplate.from_template(prompt_template)

        # Build the RAG chain with configurable k
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": retriever_k})
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} | self.prompt | self.llm | StrOutputParser()
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
