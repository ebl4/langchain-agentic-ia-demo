from typing import Dict, Any
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from marshmallow import pprint
from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient()

@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters     
      (e.g., use 'capacite' instead of 'capacité').
    """
    
    pprint(f"[VenueAgent] Performing web search #{search_number} with query: {query}")
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    try:
        pprint(f"[VenueAgent] Calling TavilyClient.search with query: {query}")
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}

class VenueAgent:
    """Venue agent for finding wedding venues."""

    def __init__(self, model: ChatGoogleGenerativeAI):
        self.agent = None
        self.model = model

    def initialize(self):
        """Initialize the agent."""
        
        pprint("Initializing VenueAgent...")
        self.agent = create_agent(
            model=self.model,
            tools=[web_search],
            system_prompt="""
            You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
            You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
            - Price (lowest)
            - Capacity (exact match)
            - Reviews (highest)
            You may need to make multiple searches to iteratively find the best options. 
            You have a suggested limit of 12 web searches. Count every web_search call you make.
            After 12 searches, you should stop searching and summarize the best options you have
            found so far.
            """
        )
        
    def get_agent(self):
        """Get the initialized agent."""
        
        pprint("Getting VenueAgent...")
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.agent
    
def get_venue_agent(model: ChatGoogleGenerativeAI):
    """Factory function to create and initialize a VenueAgent."""
    
    pprint("Creating and initializing VenueAgent...")
    venue_agent = VenueAgent(model)
    venue_agent.initialize()
    return venue_agent.get_agent()
        
    