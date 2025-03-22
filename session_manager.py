import os
import json
from datetime import datetime
from typing import List, Tuple
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from config import SESSIONS_DIR


class SessionManager:
    """Manages chat sessions including loading, saving, and selection."""

    @staticmethod
    def get_session_files() -> List[str]:
        """Get list of existing session files."""
        if not os.path.exists(SESSIONS_DIR):
            return []
        return [f for f in os.listdir(SESSIONS_DIR) if f.startswith("chat_") and f.endswith(".json")]

    @staticmethod
    def load_history(session_file: str) -> Tuple[str, ChatMessageHistory]:
        """Load chat history from file."""
        with open(os.path.join(SESSIONS_DIR, session_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        history = ChatMessageHistory()
        for msg in data["messages"]:
            if msg["type"] == "human":
                history.add_user_message(msg["content"])
            elif msg["type"] == "ai":
                history.add_ai_message(msg["content"])

        return data["session_id"], history

    @staticmethod
    def save_history(session_id: str, history: ChatMessageHistory) -> None:
        """Save chat history to file."""
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        filename = os.path.join(SESSIONS_DIR, f"chat_{session_id}.json")

        data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "messages": [
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                } for msg in history.messages
            ]
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def select_session(cls) -> Tuple[str, ChatMessageHistory]:
        """Interactive session selection menu."""
        sessions = cls.get_session_files()

        if not sessions:
            print("No previous conversations found.")
            return cls.create_new_session()

        print("\nSaved conversations:")
        for i, session in enumerate(sessions, 1):
            with open(os.path.join(SESSIONS_DIR, session), "r") as f:
                data = json.load(f)
            print(f"{i}. {data['session_id']} - {data['created_at']}")

        print("\n0. Create new conversation")
        choice = input("\nSelect an option: ")

        if choice == "0":
            return cls.create_new_session()

        try:
            selected_file = sessions[int(choice)-1]
            session_id, history = cls.load_history(selected_file)
            print(f"\nLoaded session: {session_id}")
            return session_id, history
        except (ValueError, IndexError):
            print("Invalid choice, creating new session.")
            return cls.create_new_session()

    @staticmethod
    def create_new_session() -> Tuple[str, ChatMessageHistory]:
        """Create new chat session."""
        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        print(f"\nNew session created: {session_id}")
        return session_id, ChatMessageHistory()