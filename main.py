from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOllama(
    model="gemma3:4b",
    validate_model_on_init=True,
    temperature=0.1,
    num_predict=1024,
)

for chunk in model.stream("This is a test prompt."):
    print(chunk.text, end="", flush=True)