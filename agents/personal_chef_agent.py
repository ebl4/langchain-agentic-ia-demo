from pprint import pprint
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    
    """Search the web for the given query and return the results."""
    
    return tavily_client.search(query)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

system_prompt = """You are a personal chef. 
You will be given a question about what to cook for dinner. 
You should respond with a recipe that includes the name of the dish, 
a list of ingredients, and step-by-step instructions on how to prepare it."""

config = {"configurable": {"thread_id": "1"}}

# question = HumanMessage("Who is the current major of San Francisco?")
# question = HumanMessage(content="How up to date is your trainning knowledge?")

agent = create_agent(
    model=model, 
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

question = HumanMessage("I have some leftover chicken and rice. What can I make?")

response = agent.invoke(
    {"messages": [question] },
    config
)

pprint(response['messages'][-1].content)