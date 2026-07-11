from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Responsible only for splitting documents into chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ],
        )

    def chunk_documents(
        self,
        documents: List[Any],
    ) -> List[Any]:

        chunks = self.splitter.split_documents(documents)

        return chunks