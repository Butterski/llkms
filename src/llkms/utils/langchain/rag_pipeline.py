from typing import Any, Dict, Tuple

from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from llkms.utils.langchain.model_factory import ModelConfig, ModelFactory
from llkms.utils.logger import logger


class RAGPipeline:
    def __init__(self, vector_store: FAISS, model_config: ModelConfig):
        """
        Initialize the RAGPipeline.

        Args:
            vector_store (FAISS): The vector store instance.
            model_config (ModelConfig): Configuration for the language model.
        """
        self.vector_store = vector_store
        self.llm = ModelFactory.create_model(model_config)

        # Define RAG prompt
        self.prompt = PromptTemplate.from_template(
            """
        Answer the question based on the following context. If you don't know 
        the answer, just say you don't know. Use three sentences maximum.
                                                   
        If the context is not enough to answer the question, say so and if you know the answer start message with 'Not enough context, but...'
        
        Context: {context}
        Question: {question}
        
        Answer:"""
        )

        # Build the RAG chain
        self.chain = (
            {"context": self.vector_store.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def get_retrieved_docs(self, question: str):
        """
        Retrieve the documents used as context for the given question.

        Args:
            question (str): The user's question.

        Returns:
            List[Document]: List of retrieved documents with metadata.
        """
        return self.vector_store.as_retriever().get_relevant_documents(question)

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
