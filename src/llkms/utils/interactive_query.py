from llkms.utils.logger import logger
import questionary

def run_interactive_query(rag, update_usage_callback):
    """
    Run an interactive query loop.

    Args:
        rag: The RAG pipeline instance.
        update_usage_callback (function): Function to update cumulative usage.
    """
    logger.info("Starting interactive query loop")
    while True:
        question = questionary.text(
            "Enter your question (or 'quit' to exit):",
            qmark="🤔"
        ).ask()
        
        if not question or question.lower() == "quit":
            logger.info("User requested to quit")
            break
            
        try:
            logger.debug(f"Processing question: {question}")
            answer, usage = rag.query(question)
            update_usage_callback(usage)
            
            questionary.print("\n" + "─" * 80)
            questionary.questionary.print("Answer:", style="bold")
            questionary.print(answer)
            questionary.print("─" * 80 + "\n")
            
            show_docs = questionary.confirm(
                "Would you like to see the retrieved documents?",
                default=False
            ).ask()
            
            if show_docs:
                docs = rag.get_retrieved_docs(question)
                for idx, doc in enumerate(docs):
                    questionary.print(f"\n📄 Document {idx + 1}:")
                    questionary.print("─" * 40)
                    questionary.print(doc.page_content)
                    questionary.print("─" * 40)
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            questionary.questionary.print(
                f"Error processing question: {e}",
                style="bold red"
            )
