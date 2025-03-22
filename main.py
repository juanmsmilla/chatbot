from context_manager import ContextManager
from session_manager import SessionManager
from chatbot import Chatbot


def main():
    """Main execution flow for the chatbot application."""
    # Initialize components
    chatbot = Chatbot()
    session_id, history = SessionManager.create_new_session()
    chatbot.set_session_history(session_id, history)

    print("\n--- Chat Mode Active (type 'exit' to quit) ---")
    print("Special commands: 'chat', 'context'")

    first_input = input("\nYou: ").strip().lower()

    # Handle context selection
    if first_input in ['context', 'contexto']:
        new_context = ContextManager.select_context()
        chatbot.update_context(new_context)

    # Handle session selection
    elif first_input in ['chat', 'chats']:
        new_session_id, new_history = SessionManager.select_session()
        session_id = new_session_id
        chatbot.set_session_history(session_id, new_history)

    # Process first message if not a command
    if first_input not in ['context', 'contexto', 'chat', 'chats'] and first_input != 'exit':
        response = chatbot.process_message(first_input, session_id)
        print(f"\nAssistant: {response}")

    # Main chat loop
    try:
        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'exit':
                break

            response = chatbot.process_message(user_input, session_id)
            print(f"\nAssistant: {response}")

    except KeyboardInterrupt:
        print("\nOperation cancelled...")

    finally:
        save_choice = input("\nSave conversation? (y/n): ").lower()
        if save_choice == "y":
            SessionManager.save_history(
                session_id, chatbot.message_history_store[session_id])
            print(f"Conversation saved as: chat_{session_id}.json")
        else:
            print("Conversation not saved.")


if __name__ == "__main__":
    main()
