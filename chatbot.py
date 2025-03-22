from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from config import MODEL_CONFIG, DEFAULT_CONTEXT


class Chatbot:
    """Manages the chatbot's conversation chain and responses."""

    def __init__(self):
        """Initialize the chatbot with default settings."""
        self.message_history_store = {}
        self.current_context = DEFAULT_CONTEXT["instructions"]
        self.chat = ChatDeepSeek(
            temperature=MODEL_CONFIG["temperature"],
            model=MODEL_CONFIG["model"],
            api_key=MODEL_CONFIG["api_key"]
        )
        self._setup_chain()

    def _setup_chain(self):
        """Set up the conversation chain with the current context."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.current_context),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}"),
        ])
        self.chain = self.prompt | self.chat
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self._get_message_history,
            input_messages_key="user_input",
            history_messages_key="chat_history"
        )

    def _get_message_history(self, session_id: str) -> ChatMessageHistory:
        """Retrieve message history for session."""
        return self.message_history_store.get(session_id, ChatMessageHistory())

    def update_context(self, new_context: str):
        """Update the chatbot's context and reinitialize the chain."""
        self.current_context = new_context
        self._setup_chain()

    def process_message(self, user_input: str, session_id: str) -> str:
        """Process a user message and return the chatbot's response."""
        response = self.chain_with_history.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content

    def set_session_history(self, session_id: str, history: ChatMessageHistory):
        """Set the message history for a session."""
        self.message_history_store[session_id] = history
