from langchain_ollama import ChatOllama

def init_model():
    return ChatOllama(
        model="gemma3:4b",
        temperature=0.1,
        num_predict=1024,
        validate_model_on_init=True,
        streaming=True,
    )