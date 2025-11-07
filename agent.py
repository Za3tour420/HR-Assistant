from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from typing import Annotated, TypedDict
import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

######################
# LLM Init
######################

def init_model():
    return ChatOllama(
        model="gemma3:4b",
        temperature=0.1,
        num_predict=1024,
        validate_model_on_init=True,
    )

######################
# Tools Init
######################

def init_tools():
    return []  # or GmailToolkit().get_tools()

######################
# Agent + Tool Nodes
######################

def tool_node(state):
    tools = init_tools()
    return {"tools": tools}

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    tools: list

def agent_node(state: AgentState):
    llm = init_model()
    
    print("\n[Agent response]: ", end="", flush=True)
    
    collected_content = ""
    
    # Stream tokens from the LLM
    for chunk in llm.stream(state["messages"]):
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
            collected_content += chunk.content
    
    print()  # newline after stream
    
    # Return the collected response as an AIMessage
    return {"messages": [AIMessage(content=collected_content)]}


######################
# Graph Build
######################
graph = StateGraph(AgentState)
graph.add_node("tool_node", tool_node)
graph.add_node("agent_node", agent_node)
graph.add_edge(START, "tool_node")
graph.add_edge("tool_node", "agent_node")
graph.add_edge("agent_node", END)

memory = InMemorySaver()
app = graph.compile(checkpointer=memory)

######################
# Main Loop
######################
while True:
    user_input = input("\nEnter your message: ").strip()
    if user_input.lower() == "exit":
        break

    inputs = {"messages": [HumanMessage(content=user_input)]}

    for step in app.stream(inputs, {"configurable": {"thread_id": "session_42"}}):
        for node_name, value in step.items():
            if node_name != "agent_node":  # Don't print for agent_node since we're streaming
                print(f"\n🔹 Step: {node_name} finished.")
    
    print(f"\n🔹 Step: agent_node finished.")