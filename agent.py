from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain.agents import create_agent
from typing import Annotated, TypedDict
import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

###########################################
# LLM INIT
###########################################

def init_model():
    return ChatOllama(
        model="gemma3:4b",
        temperature=0.1,
        num_predict=1024,
        validate_model_on_init=True,
    )

###########################################
# TOOLS INIT
###########################################

def init_tools():
    # You can register real tools here
    return []

###########################################
# STATE
###########################################

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

###########################################
# REACT AGENT NODE
###########################################

def build_agent_node():
    llm = init_model()
    tools = init_tools()
    react_agent = create_agent(llm, tools)

    def agent_node(state: AgentState):
        print("\n[Agent response]: ", end="", flush=True)

        collected_text = ""

        # Stream output for display only
        for chunk in llm.stream(state["messages"]):
            content = getattr(chunk, "content", None)
            if content:
                print(content, end="", flush=True)
                collected_text += content

        print()  # newline after stream

        # Return proper dict matching AgentState
        return {"messages": [AIMessage(content=collected_text)]}

    return agent_node

agent_node = build_agent_node()

###########################################
# GRAPH BUILD — SINGLE NODE
###########################################

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

memory = InMemorySaver()
app = graph.compile(checkpointer=memory)

###########################################
# MAIN LOOP
###########################################

while True:
    user_input = input("\nEnter your message: ").strip()
    if user_input.lower() == "exit":
        break

    inputs = {"messages": [HumanMessage(content=user_input)]}

    for step in app.stream(inputs, {"configurable": {"thread_id": "session_42"}}):
        for node_name, value in step.items():
            if node_name != "agent":  # avoid printing streamed content twice
                print(f"\n🔹 Step: {node_name} finished.")

    print(f"\n🔹 Step: agent finished.")