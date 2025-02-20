from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
import os
from dotenv import load_dotenv


load_dotenv()
model = ChatOpenAI(temperature=0.7)

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def make_blog_generation_graph():
    """Create a blog generation Agent"""
    graph_workflow = StateGraph(State)

    def generate_title(state):
        prompt_1= SystemMessage(content="As an experinced writer generate the title for the given topic.")
        return {"messages":[model.invoke([prompt_1]+state["messages"])]}
    
    def generate_blog(state):
        prompt_2=SystemMessage(content="As an experinced content creator write a blog with 500 words limit in 4 paragraphs with precise output.")
        return {"messages":[model.invoke([prompt_2]+state["messages"])]}
    
    #Add nodes to the graph
    graph_workflow.add_node("title_generation", generate_title)
    graph_workflow.add_node("content_generation", generate_blog)

    #Add edges to the graph
    graph_workflow.add_edge(START, "title_generation")
    graph_workflow.add_edge("title_generation", "content_generation")
    graph_workflow.add_edge("content_generation", END)

    #compile the graph into excutable agent
    agent = graph_workflow.compile()
    return agent



blog_agent = make_blog_generation_graph()


