from llkms.utils.logger import logger

def run_interactive_query(rag, update_usage_callback):
    """
    Run an interactive query loop.

    Args:
        rag: The RAG pipeline instance.
        update_usage_callback (function): Function to update cumulative usage.
    """
    logger.info("Starting interactive query loop")
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            logger.info("User requested to quit")
            break
        try:
            logger.debug(f"Processing question: {question}")
            answer, usage = rag.query(question)
            update_usage_callback(usage)
            print(f"\nAnswer: {answer}")
            
            # Optionally show retrieved documents
            show_docs = input("Show retrieved documents? (y/N): ").strip().lower()
            if show_docs == "y":
                docs = rag.get_retrieved_docs(question)
                for idx, doc in enumerate(docs):
                    print(f"\nDocument {idx + 1}:")
                    print(doc.page_content)
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"Error processing question: {e}")
