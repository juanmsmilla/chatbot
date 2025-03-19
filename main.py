import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# Initialize DeepSeek chat model
chat = ChatDeepSeek(
    temperature=0.7,
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# Create prompt template with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de IA útil. Responde en español a menos que se te pida otro idioma."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}"),
])

# Create a single chat history for personal use
chat_history = ChatMessageHistory()

# Create chain with message history
chain = prompt | chat
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,  # Usamos el mismo historial siempre
    input_messages_key="user_input",
    history_messages_key="chat_history"
)

# Terminal interaction loop
if __name__ == "__main__":
    print("DeepSeek Chatbot - Escribe 'exit' para salir\n")
    while True:
        try:
            user_input = input("Tú: ")
            if user_input.lower() == 'exit':
                break
            
            response = chain_with_history.invoke(
                {"user_input": user_input},
                config={"configurable": {"session_id": "unica_sesion"}}  # Usamos un ID fijo
            )
            
            print(f"\nAsistente: {response.content}\n")
            
        except KeyboardInterrupt:
            print("\nChat finalizado.")
            break