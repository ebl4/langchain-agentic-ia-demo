from langchain.tools import ToolRuntime, tool
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from agents import playlist_agent, travel_agent, venue_agent
from states.wedding_state import WeddingState
from pprint import pprint

class Coordinator:
    def __init__(self):
        pprint("Initializing coordinator...")
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.agent = create_agent(
            model=self.model,
            tools=[self.search_flights, self.search_venues, self.suggest_playlist, self.update_state],
            state_schema=WeddingState,
            system_prompt="""
            You are a wedding coordinator. 
            First find all the information you need to update the state. When you have the information, update the state.
            Once that has completed and returned, you can delegate the tasks 
            to your specialists for flights, venues, and playlists.
            Once you have received their answers, coordinate the perfect wedding for me.
            """
        )
        
    @tool
    async def search_flights(self, runtime: ToolRuntime) -> str:
        """Travel agent searches for flights to the desired destination wedding location."""
        
        pprint(f"[MCP coordinator] Searching flights with runtime state: {runtime.state}")
        origin = runtime.state["origin"]
        destination = runtime.state["destination"]
        agent = await travel_agent.get_travel_agent(self.model);
        response = await agent.ainvoke({"messages": [HumanMessage(content=f"Find flights from {origin} to {destination}")]})
        return response['messages'][-1].content

    @tool
    def search_venues(self, runtime: ToolRuntime) -> str:
        """Venue agent chooses the best venue for the given location and capacity."""
        
        pprint(f"[MCP coordinator] Searching venues with runtime state: {runtime.state}")
        destination = runtime.state["destination"]
        capacity = runtime.state["guest_count"]
        query = f"Find wedding venues in {destination} for {capacity} guests"
        agent = venue_agent.get_venue_agent(self.model);
        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        return response['messages'][-1].content

    @tool
    def suggest_playlist(self, runtime: ToolRuntime) -> str:
        """Playlist agent curates the perfect playlist for the given genre."""
        
        pprint(f"[MCP coordinator] Suggesting playlist with runtime state: {runtime.state}")
        genre = runtime.state["genre"]
        query = f"Find {genre} tracks for wedding playlist"
        agent = playlist_agent.get_playlist_agent(self.model);
        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        return response['messages'][-1].content

    @tool
    def update_state(origin: str, destination: str, guest_count: str, genre: str, runtime: ToolRuntime) -> str:
        """Update the state when you know all of the values: origin, destination, guest_count, genre. 
        This tool must be called alone, without any other tool calls. It must complete and return to make,
        the information available to other tools."""
        
        pprint(f"[MCP coordinator] Updating state with origin: {origin}, destination: {destination}, guest_count: {guest_count}, genre: {genre}")
        return Command(update={
            "origin": origin, 
            "destination": destination, 
            "guest_count": guest_count, 
            "genre": genre, 
            "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)]}
            )

    async def invoke(self, message: str, config: dict | None = None):
        pprint(f"[MCP coordinator] Invoking with message: {message} and config: {config}")
        return await self.agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config or {},
        )
            
    async def run_tests(self):
        pprint("Invoking coordinator with wedding planning request...")
        response = await self.invoke(
            "I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre",
            config={"tags": ["WP"], "recursion_limit": 40},
        )

        pprint(response)
        return response


if __name__ == "__main__":
    import asyncio
    
    coordinator = Coordinator()
    asyncio.run(coordinator.run_tests())