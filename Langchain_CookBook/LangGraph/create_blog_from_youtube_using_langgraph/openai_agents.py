from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
#from langchain import LangChain
from langchain.document_loaders import YoutubeLoader
from pydantic import BaseModel, Field
from typing import Optional
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv


load_dotenv()
llm = ChatOpenAI(temperature=0.7)

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


# Define the data model using pydantic
class VideoInput(BaseModel):
    video_url:str=Field(description="URL of the YouTube video")

class BlogOutput(BaseModel):
    blog_content:str=Field(description="Generated blog content")

class ReviewOutput(BaseModel):
    review_feedback:Optional[str]=Field(None, description="Feedback from the reviewer")

class HumanFeedbackOutput(BaseModel):
    refine_blog:bool=Field(description="whether to refine the blog or not")
    additional_comments: Optional[str]=Field(None, description="Additional comments to refine the blog")

class State(BaseModel):
    video_url:str
    transcript:Optional[str] = None
    blog_content:Optional[str] = None
    review_feedback:Optional[str] = None
    refine_blog:Optional[str] = None
    additional_comments:Optional[str] = None

def Document_Loader(state:State)-> State:
    loader = YoutubeLoader.from_youtube_url(state.video_url, add_video_info=False)
    documents = loader.load()
    transcript = " ".join([doc.page_content for doc in documents])
    state.transcript = transcript
    return state

def Blog_Creator(state:State)-> State:
    prompt = f"Craete a blog post from the following transcript\n\n{state.transcript}"
    blog_content = llm.invoke(prompt)
    state.blog_content = blog_content.content
    return state

def Reviewer(state:State)-> State:
    #simulate a review process
    feedback = "Looks Good!" #simulated feedback
    state.review_feedback = feedback
    return state

def HumanFeedback(state:State)-> State:
    #simulate human feedback
    return state

def should_continue(state:State, refine_blog: bool, additional_comments: Optional[str]=None)-> State:
    state.refine_blog = refine_blog
    state.additional_comments = additional_comments
    return state

def Blog_Refiner(state:State)-> State:
    if state.refine_blog:
        refined_prompt = f"Refine the following blog post with these comments: {state.additional_comments}\n\n{state.blog_content}"
        refined_blog_content = llm.invoke(refined_prompt)
        state.blog_content = refined_blog_content.content
        return state
    
#create the nodes
builder = StateGraph(State)

builder.add_node("Document_Loader", Document_Loader)
builder.add_node("Blog_Creator", Blog_Creator)
builder.add_node("Reviewer", Reviewer)
builder.add_node("HumanFeedback", HumanFeedback)
builder.add_node("Blog_Refiner", Blog_Refiner)


builder.add_edge(START, "Document_Loader")
builder.add_edge("Document_Loader", "Blog_Creator")
builder.add_edge("Blog_Creator", "Reviewer")
builder.add_edge("Reviewer", "HumanFeedback")
builder.add_conditional_edges(
    "HumanFeedback",
    should_continue,
    {
        "No": END,
        "Yes": "Blog_Refiner",
    })
builder.add_edge("Blog_Refiner", "Reviewer")

#memory = MemorySaver()
graph = builder.compile(interrupt_before=['HumanFeedback'])

video_url = input("Please provide the Youtube link for blog generation: ")
state = State(video_url=video_url)
output_state = graph.invoke(state)

print("Generated blog content:")
print(output_state['blog_content'])

## simulate the human feedback interuruption and resumption 
refine_blog_input = input("Do you want to refine the blog? (Yes/No): ").strip().lower()
refine_blog = refine_blog_input == "yes"
additional_comments = None
if refine_blog:
    additional_comments = input("Please provide the additonal comment to refine the blog: ").strip()

output_state = should_continue(output_state, refine_blog, additional_comments)


## continue the graph execution after human feedbcak
output_state = graph.invoke(output_state)
print(output_state['blog_content'])
                 