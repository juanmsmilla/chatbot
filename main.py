import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage 


CHATS_DIR = "chats"
CONTEXT_INSTRUCTIONS = "Eres un asistente de IA útil. Responde en español."

# Cargar variables de entorno
load_dotenv()

# 1. Configuración inicial del modelo
chat = ChatDeepSeek(
    temperature=0.7,
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 2. Plantilla del prompt con historial
prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXT_INSTRUCTIONS),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}"),
])

# 3. Funciones de persistencia mejoradas
def get_session_files():
    """Obtiene todos los archivos de sesión existentes"""
    if not os.path.exists(CHATS_DIR):
        return []
    return [f for f in os.listdir(CHATS_DIR) if f.startswith("chat_") and f.endswith(".json")]

def load_history(session_file: str) -> tuple[str, ChatMessageHistory]:
    """Carga el historial desde un archivo incluyendo el session_id"""
    with open(os.path.join(CHATS_DIR, session_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    history = ChatMessageHistory()
    for msg in data["messages"]:
        if msg["type"] == "human":
            history.add_user_message(msg["content"])
        elif msg["type"] == "ai":
            history.add_ai_message(msg["content"])
    
    return data["session_id"], history

def save_history(session_id: str, history: ChatMessageHistory) -> None:
    """Guarda el historial con metadata de sesión"""
    os.makedirs(CHATS_DIR, exist_ok=True)
    filename = os.path.join(CHATS_DIR, f"chat_{session_id}.json")
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

# 4. Menú de inicio interactivo
def select_session() -> tuple[str, ChatMessageHistory]:
    """Muestra menú para seleccionar o crear sesión"""
    sessions = get_session_files()
    
    if not sessions:
        print("No hay conversaciones anteriores.")
        return create_new_session()
    
    print("\nConversaciones guardadas:")
    for i, session in enumerate(sessions, 1):
        with open(os.path.join("chats", session), "r") as f:
            data = json.load(f)
        print(f"{i}. {data['session_id']} - {data['created_at']}")
    
    print("\n0. Crear nueva conversación")
    choice = input("\nSelecciona una opción: ")
    
    if choice == "0":
        return create_new_session()
    
    try:
        selected_file = sessions[int(choice)-1]
        session_id, history = load_history(selected_file)
        print(f"\nSesión cargada: {session_id}")
        return session_id, history
    except (ValueError, IndexError):
        print("Opción inválida, creando nueva sesión.")
        return create_new_session()

def create_new_session() -> tuple[str, ChatMessageHistory]:
    """Crea una nueva sesión con ID único"""
    session_id = datetime.now().strftime("sesion_%Y%m%d_%H%M%S")
    print(f"\nNueva sesión creada: {session_id}")
    return session_id, ChatMessageHistory()

# 5. Configuración de la cadena con historial
message_history_store = {}

def get_message_history(session_id: str) -> ChatMessageHistory:
    return message_history_store.get(session_id, ChatMessageHistory())

chain = prompt | chat
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)

if __name__ == "__main__":
    # Crear nueva sesión por defecto
    session_id, history = create_new_session()
    message_history_store[session_id] = history
    
    print("\n--- Modo Chat Activo (escribe 'exit' para salir) ---")
    print("Escribe 'chat' o 'chats' para ver conversaciones guardadas")
    
    # Capturar primera entrada
    first_input = input("\nTú: ").strip().lower()
    
    if first_input in ['chat', 'chats']:
        # Seleccionar sesión existente
        new_session_id, new_history = select_session()
        session_id = new_session_id
        message_history_store[session_id] = new_history
    elif first_input != 'exit':
        # Procesar primera entrada como mensaje
        response = chain_with_history.invoke(
            {"user_input": first_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"\nAsistente: {response.content}")
    
    # Bucle de conversación para entradas posteriores
    try:
        while True:
            user_input = input("\nTú: ").strip()
            
            if user_input.lower() == 'exit':
                break
            
            response = chain_with_history.invoke(
                {"user_input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"\nAsistente: {response.content}")
            
    except KeyboardInterrupt:
        print("\nInterrupción detectada...")
    
    finally:
        save_choice = input("\n¿Deseas guardar la conversación? (s/n): ").lower()
        if save_choice == "s":
            save_history(session_id, message_history_store[session_id])
            print(f"Conversación guardada como: chat_{session_id}.json")
        else:
            print("Conversación no guardada.")