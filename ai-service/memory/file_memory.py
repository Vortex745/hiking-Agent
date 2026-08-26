import json
from pathlib import Path
from memory.base import ChatMemory
from runtime_paths import runtime_dir

MEMORY_DIR = runtime_dir("MEMORY_DATA_DIR", "memory_data")
WINDOW_SIZE = 60


class FileChatMemory(ChatMemory):
    """File-based chat memory with sliding window."""

    def __init__(self, chat_id: str = "default", save_dir: str | None = None):
        self.chat_id = chat_id
        self._list_format = save_dir is not None
        if save_dir:
            base_dir = Path(save_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            self.file_path = base_dir / "messages.json"
        else:
            self.file_path = MEMORY_DIR / f"{chat_id}.json"
        self._messages: list[dict] = []
        self._load()

    def _load(self):
        """Load messages from file."""
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text("utf-8"))
                if isinstance(data, list):
                    self._messages = data
                elif isinstance(data, dict):
                    self._messages = data.get("messages", [])
                else:
                    self._messages = []
            except (json.JSONDecodeError, KeyError):
                self._messages = []

    def _save(self):
        """Save messages to file."""
        data = self._messages if self._list_format else {"chat_id": self.chat_id, "messages": self._messages}
        self.file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

    def add_message(self, role: str, content: str):
        """Add a message and maintain window size."""
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > WINDOW_SIZE:
            self._messages = self._messages[-WINDOW_SIZE:]
        self._save()

    def get_messages(self) -> list[dict]:
        """Get all stored messages."""
        return self._messages.copy()

    def clear(self):
        """Clear all messages."""
        self._messages = []
        self._save()
