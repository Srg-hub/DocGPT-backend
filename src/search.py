import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.embedding import Embedder
from src.vectorstore import FaissVectorStore
from src.history import ChatHistory

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        vector_path: str,
        session_id: str,
        model_name: str = os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant",
        ),
    ):

        self.embedder = Embedder()

        self.vectorstore = FaissVectorStore(vector_path)
        self.vectorstore.load()

        self.history = ChatHistory(session_id)

        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
        )

    # -----------------------------------------------------
    # Rewrite follow-up questions
    # -----------------------------------------------------

    def rewrite_question(self, question: str):

        history = self.history.format_history()

        if history.strip() == "":
            return question

        prompt = f"""
You are an AI assistant.

Your ONLY job is to rewrite follow-up questions into standalone questions.

Examples

Conversation:

User: What is encapsulation?

Assistant: ...

User: Give example for it

Rewrite:

Give an example of encapsulation.


------------------------

Conversation:

User: Explain normalization.

Assistant: ...

User: What are its types?

Rewrite:

What are the types of normalization?


------------------------

Conversation:

{history}

Current Question:

{question}

Return ONLY the rewritten question.

Do not answer it.

If rewriting is unnecessary,
return the original question.
"""

        response = self.llm.invoke(prompt)

        rewritten = response.content.strip()

        print("\n========================")
        print("Original :", question)
        print("Rewritten:", rewritten)
        print("========================\n")

        return rewritten

    # -----------------------------------------------------
    # Main RAG
    # -----------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int = 4,
    ):

        rewritten_question = self.rewrite_question(question)

        query_embedding = self.embedder.embed_query(
            rewritten_question
        )

        results = self.vectorstore.search(
            query_embedding,
            top_k=top_k,
        )

        context = "\n\n".join(
            item["metadata"]["text"]
            for item in results
        )

        previous_chat = self.history.format_history()

        prompt = f"""
You are an intelligent AI assistant.

Rules:

- Answer ONLY using the retrieved context.

- NEVER mention:
    "According to the document"
    "According to the context"
    "Based on the uploaded document"

- Answer naturally.

- If the answer is missing, reply exactly:

I couldn't find that information in the uploaded document.

---------------------------------

Conversation

{previous_chat}

---------------------------------

Retrieved Context

{context}

---------------------------------

Question

{rewritten_question}
"""

        response = self.llm.invoke(prompt)

        self.history.save("User", question)
        self.history.save("Assistant", response.content)

        unique_sources = []
        seen = set()

        for item in results:

            source = item["metadata"]["source"]

            key = (
                Path(source["source"]).name,
                source["page"],
            )

            if key in seen:
                continue

            seen.add(key)

            unique_sources.append(
                {
                    "filename": Path(source["source"]).name,
                    "page": source["page"] + 1,
                }
            )

        return {
            "answer": response.content,
            "sources": unique_sources,
        }