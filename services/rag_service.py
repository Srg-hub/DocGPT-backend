from services.document_processor import DocumentProcessor
from src.search import RAGSearch


class RAGService:

    def process_documents(
        self,
        upload_folder: str,
        vector_folder: str,
    ):

        processor = DocumentProcessor()

        processor.process(
            upload_folder=upload_folder,
            vector_folder=vector_folder,
        )

    def ask_question(
        self,
        question: str,
        vector_folder: str,
        session_id: str,
        top_k: int = 4,
    ):

        rag = RAGSearch(
            vector_path=vector_folder,
            session_id=session_id,
        )

        return rag.ask(
            question=question,
            top_k=top_k,
        )