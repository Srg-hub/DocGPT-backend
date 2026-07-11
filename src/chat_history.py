import json
from pathlib import Path


class ChatHistory:

    def __init__(self, session_id: str):
        self.history_file = (
            Path("sessions")
            / session_id
            / "history.json"
        )

        if not self.history_file.exists():
            self.save([])

    def load(self):
        with open(self.history_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, history):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def add_user_message(self, message: str):
        history = self.load()

        history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        self.save(history)

    def add_ai_message(self, message: str):
        history = self.load()

        history.append(
            {
                "role": "assistant",
                "content": message,
            }
        )

        self.save(history)

    def get_history(self):
        return self.load()