# from pathlib import Path
# import shutil
# from typing import Annotated

# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse

# from src.session_manager import SessionManager
# from services.rag_service import RAGService
# from models.schemas import ChatRequest
# from fastapi.middleware.cors import CORSMiddleware
# from src.history import ChatHistory

# app = FastAPI(
#     title="RAG Document Chat API",
#     version="1.0.0"
# )
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://localhost:5174",
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# session_manager = SessionManager()
# rag_service = RAGService()


# @app.get("/")
# def root():
#     return {"message": "RAG Backend Running"}


# @app.post("/upload")
# async def upload(
#     files: Annotated[list[UploadFile], File(...)],
# ):
#     session = session_manager.create_session()

#     upload_folder = Path(session["upload_path"])
#     vector_folder = Path(session["vector_path"])

#     for file in files:
#         destination = upload_folder / file.filename

#         with open(destination, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#     rag_service.process_documents(
#         str(upload_folder),
#         str(vector_folder)
#     )

#     return {
#         "session_id": session["session_id"],
#         "message": "Documents processed successfully."
#     }


# @app.post("/chat")
# def chat(request: ChatRequest):

#     vector_folder = Path("sessions") / request.session_id / "vector_store"

#     if not vector_folder.exists():
#         raise HTTPException(status_code=404, detail="Invalid Session ID")

#     result = rag_service.ask_question(
#         question=request.question,
#         vector_folder=str(vector_folder),
#         session_id=request.session_id,
#     )

#     return JSONResponse(result)

# @app.get("/history/{session_id}")
# def get_history(session_id: str):

#     history_file = (
#         Path("sessions")
#         / session_id
#         / "history.json"
#     )

#     if not history_file.exists():
#         return []

#     history = ChatHistory(session_id)

#     return history.load()


from pathlib import Path
import shutil
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.session_manager import SessionManager
from src.history import ChatHistory
from services.rag_service import RAGService
from models.schemas import ChatRequest

app = FastAPI(
    title="RAG Document Chat API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_manager = SessionManager()
rag_service = RAGService()


# ==========================================================
# Root
# ==========================================================

@app.get("/")
def root():
    return {
        "message": "RAG Backend Running"
    }


# ==========================================================
# Upload Documents
# ==========================================================

@app.post("/upload")
async def upload(
    files: Annotated[list[UploadFile], File(...)],
):

    session = session_manager.create_session()

    upload_folder = Path(session["upload_path"])
    vector_folder = Path(session["vector_path"])

    for file in files:

        destination = upload_folder / file.filename

        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    rag_service.process_documents(
        str(upload_folder),
        str(vector_folder),
    )

    return {
        "session_id": session["session_id"],
        "message": "Documents processed successfully.",
    }


# ==========================================================
# Chat
# ==========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    vector_folder = (
        Path("sessions")
        / request.session_id
        / "vector_store"
    )

    if not vector_folder.exists():
        raise HTTPException(
            status_code=404,
            detail="Invalid Session ID",
        )

    result = rag_service.ask_question(
        question=request.question,
        vector_folder=str(vector_folder),
        session_id=request.session_id,
    )

    return JSONResponse(result)


# ==========================================================
# Chat History
# ==========================================================

@app.get("/history/{session_id}")
def get_history(session_id: str):

    history_file = (
        Path("sessions")
        / session_id
        / "history.json"
    )

    if not history_file.exists():
        return []

    history = ChatHistory(session_id)

    return history.load()


# ==========================================================
# Sidebar Sessions
# ==========================================================

@app.get("/sessions")
def get_sessions():

    return session_manager.list_sessions()


# ==========================================================
# Delete Session
# ==========================================================

@app.delete("/session/{session_id}")
def delete_session(session_id: str):

    session_manager.delete_session(session_id)

    return {
        "message": "Session deleted successfully."
    }