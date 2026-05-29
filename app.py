import os
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# 1. Load environment variables & Initialize our free LLM
load_dotenv()
# We use llama3-70b via Groq because it excels at logical routing and agent orchestration
llm = ChatGroq(model="llama-3.3-70b-versatile" , groq_api_key = os.getenv("GROQ_API_KEY"))

# 2. Define the Shared State (The memory notebook our agents pass around)
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    next_step: str

# 3. Define the Helper function to create standard Worker Agents
def create_worker(llm, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | llm

# --- Define Worker System Prompts ---
researcher_agent = create_worker(
    llm, 
    "You are a World-Class Researcher. Extract key facts, data, and technical details. Be concise and accurate."
)
writer_agent = create_worker(
    llm, 
    "You are a Professional Tech Writer. Take raw facts and turn them into a polished, engaging markdown summary."
)

# 4. Define the Supervisor (The Router/Project Manager)
members = ["Researcher", "Writer"]
options = ["FINISH"] + members

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a supervisor managing a team of workers: {members}. "
        "Given the conversation history, select the next worker to act. "
        "If the task is fully complete and polished, respond with FINISH."
    )),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Select one of: {options}"),
])

# Use LLM Structured Outputs to guarantee the supervisor only returns valid choices
class RouteResponse(TypedDict):
    next_step: Literal["Researcher", "Writer", "FINISH"]

supervisor_chain = supervisor_prompt.partial(members=str(members), options=str(options)) | llm.with_structured_output(RouteResponse)

# 5. Define Graph Nodes (The actual actions functions)
def supervisor_node(state: AgentState):
    response = supervisor_chain.invoke(state)
    return {"next_step": response["next_step"]}

def researcher_node(state: AgentState):
    # Only send the original user message to keep it focused
    response = researcher_agent.invoke({"messages": [state["messages"][0]]})
    return {"messages": [HumanMessage(content=response.content, name="Researcher")]}

def writer_node(state: AgentState):
    # Send the whole conversation so the writer sees the researcher's output
    response = writer_agent.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content, name="Writer")]}

# 6. Build and Compile the Workflow Graph
workflow = StateGraph(AgentState)

# Add our nodes to the graph
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)

# Create the routing logic transitions
workflow.add_edge(START, "Supervisor")

# The Supervisor reads the 'next_step' variable and routes dynamically
workflow.add_conditional_edges(
    "Supervisor",
    lambda state: state["next_step"],
    {
        "Researcher": "Researcher",
        "Writer": "Writer",
        "FINISH": END
    }
)

# After a worker finishes its task, it ALWAYS reports back to the supervisor
workflow.add_edge("Researcher", "Supervisor")
workflow.add_edge("Writer", "Supervisor")

# Compile the graph into an executable application
app = workflow.compile()

# 7. Run and Test Your Multi-Agent System!
# 7. Wrap our Multi-Agent System in a Visual Web UI
import gradio as gr

def run_agents_web(user_prompt, history):
    """This function takes input from the web screen, runs the agents, and returns the result."""
    # Run our LangGraph workflow engine with the user's input
    final_state = app.invoke(
        {"messages": [HumanMessage(content=user_prompt)]},
        config={"configurable": {"thread_id": "web_user"}}
    )
    
    # Grab the very last message in the notebook (which will be the Writer's final output)
    final_output = final_state["messages"][-1].content
    return final_output

# Create a beautiful, dark-themed Chat Dashboard
demo = gr.ChatInterface(
    fn=run_agents_web, 
    title="🤖 Multi-Agent Orchestrator",
    description="Ask the Supervisor to manage the Researcher and Tech Writer to build an executive report."
    
)

if __name__ == "__main__":
    print("🚀 Launching your Multi-Agent Website Interface...")
    # This fires up a local web server right on your computer
    demo.launch()