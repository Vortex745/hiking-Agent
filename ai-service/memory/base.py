from abc import ABC, abstractmethod


class ChatMemory(ABC):
    """Abstract base class for chat memory implementations."""

    @abstractmethod
    def add_message(self, role: str, content: str):
        """Add a message to the memory store."""
        ...

    @abstractmethod
    def get_messages(self) -> list[dict]:
        """Retrieve all messages from the memory store."""
        ...

    @abstractmethod
    def clear(self):
        """Clear all messages from the memory store."""
        ...
