"""Context management for the chatbot application."""

import os
import json
from typing import List, Dict
from config import CONTEXTS_FILE, DEFAULT_CONTEXT


class ContextManager:
    """Manages the loading, saving, and selection of conversation contexts."""

    @staticmethod
    def load_contexts() -> List[Dict]:
        """Load available contexts from JSON file."""
        if not os.path.exists(CONTEXTS_FILE):
            with open(CONTEXTS_FILE, 'w', encoding='utf-8') as f:
                json.dump([DEFAULT_CONTEXT], f, indent=2)
            return [DEFAULT_CONTEXT]

        with open(CONTEXTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_contexts(contexts: List[Dict]) -> None:
        """Save contexts to JSON file."""
        with open(CONTEXTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(contexts, f, ensure_ascii=False, indent=2)

    @classmethod
    def select_context(cls) -> str:
        """Interactive context selection menu."""
        contexts = cls.load_contexts()

        print("\nAvailable contexts:")
        for i, ctx in enumerate(contexts, 1):
            print(f"{i}. {ctx['name']}")

        print("\n0. Create new context")
        choice = input("\nSelect an option: ")

        if choice == "0":
            name = input("Context name: ")
            instructions = input("Instructions: ")
            new_context = {"name": name, "instructions": instructions}
            contexts.append(new_context)
            cls.save_contexts(contexts)
            return instructions

        try:
            return contexts[int(choice)-1]["instructions"]
        except (ValueError, IndexError):
            print("Invalid choice, using default context.")
            return DEFAULT_CONTEXT["instructions"]