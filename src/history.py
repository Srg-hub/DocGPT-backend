import json
from pathlib import Path


class ChatHistory:

    def __init__(self, session_id: str):

        self.file = (
            Path("sessions")
            / session_id
            / "history.json"
        )

        if not self.file.exists():
            self.file.write_text("[]")

    def load(self):

        with open(self.file, "r", encoding="utf8") as f:
            return json.load(f)

    def save(self, role: str, content: str):

        history = self.load()

        history.append(
            {
                "role": role,
                "content": content,
            }
        )

        with open(self.file, "w", encoding="utf8") as f:
            json.dump(
                history,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def format_history(self):

        history = self.load()

        conversation = ""

        for item in history:

            conversation += (
                f"{item['role']}: "
                f"{item['content']}\n\n"
            )

        return conversation