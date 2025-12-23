from agentic.llm import init_model
from tools.tools_register import *

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, AIMessageChunk
from langchain.agents import create_agent
from typing import Annotated, TypedDict
import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

import psutil
import subprocess
from time import sleep


###########################################
# TOOLS INIT
###########################################

def is_ollama_running():
    """Check if any Ollama-related process is running."""
    for proc in psutil.process_iter(['name']):
        if 'ollama' in proc.info['name'].lower():
            return True
    return False

def run_ollama():
    """Run the Ollama process."""
    proc = subprocess.Popen(
        ["ollama"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )
    sleep(8)
    return proc

def terminate_ollama(proc):
    """Terminate the Ollama process."""
    proc.send_signal(subprocess.signal.CTRL_BREAK_EVENT)

def init_tools():
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
    # Ensure your init_model() returns ChatOllama(..., streaming=True)
    tools = init_tools()
    
    # We use the LLM directly to handle streaming logic more cleanly within the node
    def agent_node(state: AgentState):
        # We bind tools to the LLM so it can decide to use them
        llm_with_tools = llm.bind_tools(tools)
        
        # The node simply returns the prediction; 
        # the streaming happens at the graph level (app.stream)
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    return agent_node

agent_node = build_agent_node()

###########################################
# GRAPH BUILD
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

if __name__ == "__main__":
    while True:
        if not is_ollama_running():
            proc = run_ollama()
        user_input = input("\nEnter your message: ").strip()
        if user_input.lower() == "exit":
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}
        config = {"configurable": {"thread_id": "session_42"}}

        print("\n[Agent response]: ", end="", flush=True)

        # Use stream_mode="messages" to get token chunks
        # 
        for msg, metadata in app.stream(inputs, config, stream_mode="messages"):
            # Check if the message is coming from the agent node and is a chunk
            if metadata.get("langgraph_node") == "agent":
                if isinstance(msg, AIMessageChunk):
                    print(msg.content, end="", flush=True)
                elif isinstance(msg, AIMessage) and msg.content:
                    # Fallback for full messages
                    print(msg.content, end="", flush=True)

        print(f"\n\n🔹 Step: agent finished.")