from pathlib import Path
import uuid
import logging
import shutil

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Handles creation and management of user sessions.
    """

    def __init__(self, base_path: str = "sessions"):

        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    def create_session(self):

        session_id = str(uuid.uuid4())

        session_path = self.base_path / session_id

        upload_path = session_path / "uploads"
        vector_path = session_path / "vector_store"

        upload_path.mkdir(parents=True, exist_ok=True)
        vector_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created session {session_id}")

        return {
            "session_id": session_id,
            "session_path": session_path,
            "upload_path": upload_path,
            "vector_path": vector_path,
        }

    # --------------------------------------------------
    # Return every session for sidebar
    # --------------------------------------------------

    def list_sessions(self):

        sessions = []

        for session in sorted(
            self.base_path.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):

            if not session.is_dir():
                continue

            uploads = session / "uploads"

            filename = "Untitled Chat"

            if uploads.exists():

                files = list(uploads.glob("*"))

                if files:
                    filename = files[0].name

            sessions.append(
                {
                    "session_id": session.name,
                    "filename": filename,
                }
            )

        return sessions

    # --------------------------------------------------
    # Delete one session
    # --------------------------------------------------

    def delete_session(self, session_id: str):

        folder = self.base_path / session_id

        if folder.exists():
            shutil.rmtree(folder)