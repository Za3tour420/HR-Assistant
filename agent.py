from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain.agents import create_agent
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
    agent = create_agent(
        model=init_model(),
        tools=state.get("tools", []),
        name="agent_test",
    )

    print("\n[Agent response]: ", end="", flush=True)
    
    # Invoke the agent and get the result
    result = agent.invoke(state)
    
    # Extract and print the last message
    if "messages" in result and result["messages"]:
        last_msg = result["messages"][-1]
        if hasattr(last_msg, "content"):
            print(last_msg.content, end="", flush=True)
    
    # Return only the new messages (not the entire state)
    return {"messages": result["messages"][len(state["messages"]):]}


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
            print(f"\n🔹 Step: {node_name} finished.")