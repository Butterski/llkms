import os
from typing import Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from ..logger import logger

class RAGPipeline:
    def __init__(self, vector_store: FAISS, model_provider: str = "deepseek", model: str = "deepseek-chat"):
        """
        Initialize the RAGPipeline.

        Args:
            vector_store (FAISS): The vector store instance.
            model_provider (str, optional): Provider of the model. Defaults to "deepseek".
            model (str, optional): Model name. Defaults to "deepseek-chat".
        """
        self.vector_store = vector_store
        self.model = model
        
        if model_provider == "deepseek":
            self.llm = ChatOpenAI(
                model=model,
                openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                openai_api_base='https://api.deepseek.com',
                max_tokens=1024
            )
        else:  # OpenAI
            self.llm = ChatOpenAI(model=model)
        
        # Define RAG prompt
        self.prompt = PromptTemplate.from_template("""
        Answer the question based on the following context. If you don't know 
        the answer, just say you don't know. Use three sentences maximum.
        
        Context: {context}
        Question: {question}
        
        Answer:""")
        
        # Build the RAG chain
        self.chain = (
            {"context": self.vector_store.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
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
                "successful_requests": cb.successful_requests
            }