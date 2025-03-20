import os
import json
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage

# Configuration constants
SESSIONS_DIR = "sessions"
CONTEXTS_FILE = "contexts.json"
DEFAULT_CONTEXT = {
    "name": "General Assistant",
    "instructions": "You are a helpful AI assistant. Respond in the language you are asked."
}

# Load environment variables
load_dotenv()

# Initialize AI model
chat = ChatDeepSeek(
    temperature=0.7,
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Context management functions
def load_contexts() -> List[Dict]:
    """Load available contexts from JSON file"""
    if not os.path.exists(CONTEXTS_FILE):
        with open(CONTEXTS_FILE, 'w', encoding='utf-8') as f:
            json.dump([DEFAULT_CONTEXT], f, indent=2)
        return [DEFAULT_CONTEXT]
    
    with open(CONTEXTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_contexts(contexts: List[Dict]) -> None:
    """Save contexts to JSON file"""
    with open(CONTEXTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)

def select_context() -> str:
    """Interactive context selection menu"""
    contexts = load_contexts()
    
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
        save_contexts(contexts)
        return instructions
    
    try:
        return contexts[int(choice)-1]["instructions"]
    except (ValueError, IndexError):
        print("Invalid choice, using default context.")
        return DEFAULT_CONTEXT["instructions"]

# Session management functions
def get_session_files() -> List[str]:
    """Get list of existing session files"""
    if not os.path.exists(SESSIONS_DIR):
        return []
    return [f for f in os.listdir(SESSIONS_DIR) if f.startswith("chat_") and f.endswith(".json")]

def load_history(session_file: str) -> tuple[str, ChatMessageHistory]:
    """Load chat history from file"""
    with open(os.path.join(SESSIONS_DIR, session_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    history = ChatMessageHistory()
    for msg in data["messages"]:
        if msg["type"] == "human":
            history.add_user_message(msg["content"])
        elif msg["type"] == "ai":
            history.add_ai_message(msg["content"])
    
    return data["session_id"], history

def save_history(session_id: str, history: ChatMessageHistory) -> None:
    """Save chat history to file"""
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

def select_session() -> tuple[str, ChatMessageHistory]:
    """Interactive session selection menu"""
    sessions = get_session_files()
    
    if not sessions:
        print("No previous conversations found.")
        return create_new_session()
    
    print("\nSaved conversations:")
    for i, session in enumerate(sessions, 1):
        with open(os.path.join(SESSIONS_DIR, session), "r") as f:
            data = json.load(f)
        print(f"{i}. {data['session_id']} - {data['created_at']}")
    
    print("\n0. Create new conversation")
    choice = input("\nSelect an option: ")
    
    if choice == "0":
        return create_new_session()
    
    try:
        selected_file = sessions[int(choice)-1]
        session_id, history = load_history(selected_file)
        print(f"\nLoaded session: {session_id}")
        return session_id, history
    except (ValueError, IndexError):
        print("Invalid choice, creating new session.")
        return create_new_session()

def create_new_session() -> tuple[str, ChatMessageHistory]:
    """Create new chat session"""
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    print(f"\nNew session created: {session_id}")
    return session_id, ChatMessageHistory()

# Chat chain configuration
message_history_store = {}
current_context = DEFAULT_CONTEXT["instructions"]

def get_message_history(session_id: str) -> ChatMessageHistory:
    """Retrieve message history for session"""
    return message_history_store.get(session_id, ChatMessageHistory())

# Initialize chat chain
prompt = ChatPromptTemplate.from_messages([
    ("system", current_context),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}"),
])
chain = prompt | chat
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)

# Main execution flow
if __name__ == "__main__":
    session_id, history = create_new_session()
    message_history_store[session_id] = history
    
    print("\n--- Chat Mode Active (type 'exit' to quit) ---")
    print("Special commands: 'chat', 'context'")
    
    first_input = input("\nYou: ").strip().lower()
    
    # Handle context selection
    if first_input in ['context', 'contexto']:
        new_context = select_context()
        current_context = new_context
        
        # Update prompt with new context
        prompt = ChatPromptTemplate.from_messages([
            ("system", current_context),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}"),
        ])
        chain = prompt | chat
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_message_history,
            input_messages_key="user_input",
            history_messages_key="chat_history"
        )
    
    # Handle session selection
    elif first_input in ['chat', 'chats']:
        new_session_id, new_history = select_session()
        session_id = new_session_id
        message_history_store[session_id] = new_history
    
    # Process first message if not a command
    if first_input not in ['context', 'contexto', 'chat', 'chats'] and first_input != 'exit':
        response = chain_with_history.invoke(
            {"user_input": first_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"\nAssistant: {response.content}")
    
    # Main chat loop
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                break
            
            response = chain_with_history.invoke(
                {"user_input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"\nAssistant: {response.content}")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled...")
    
    finally:
        save_choice = input("\nSave conversation? (y/n): ").lower()
        if save_choice == "y":
            save_history(session_id, message_history_store[session_id])
            print(f"Conversation saved as: chat_{session_id}.json")
        else:
            print("Conversation not saved.")