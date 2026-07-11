from src.data_loader import load_all_documents
from src.chunker import DocumentChunker
from src.embedding import Embedder
from src.vectorstore import FaissVectorStore


class DocumentProcessor:

    def __init__(self):

        self.chunker = DocumentChunker()

        self.embedder = Embedder()

    def process(self, upload_folder: str, vector_folder: str):

        documents = load_all_documents(upload_folder)

        chunks = self.chunker.chunk_documents(documents)

        texts = [chunk.page_content for chunk in chunks]

        metadata = []

        for chunk in chunks:

            metadata.append(
                {
                    "text": chunk.page_content,
                    "source": chunk.metadata,
                }
            )

        embeddings = self.embedder.embed_documents(texts)

        store = FaissVectorStore(vector_folder)

        store.build(embeddings, metadata)