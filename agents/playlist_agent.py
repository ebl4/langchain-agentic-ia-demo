from pathlib import Path
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from marshmallow import pprint

DB_PATH = Path(__file__).resolve().parent / "resources" / "Chinook.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH.as_posix()}")

@tool
def query_playlist_db(query: str) -> str:

    """Query the database for playlist information"""

    try:
        pprint(f"[PlaylistAgent] Executing database query: {query}")
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"

class PlaylistAgent:
    """Playlist agent for curating wedding playlists."""

    def __init__(self, model: ChatGoogleGenerativeAI):
        self.agent = None
        self.model = model

    def initialize(self):
        """Initialize the agent."""
        pprint("Initializing PlaylistAgent...")
        self.agent = create_agent(
            model=self.model,
            tools=[query_playlist_db],
            system_prompt="""
            You are a playlist specialist. Query the sql database and curate the perfect playlist for a wedding given a genre.
            Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
            If you run into errors when querying the database, try to fix them by making changes to the query.
            Do not come back empty handed, keep trying to query the db until you find a list of songs.

            This is a SQLite database. Before writing any data queries, first discover the schema.
            """
        )
        
    def get_agent(self):
        """Get the initialized agent."""
        pprint("Getting PlaylistAgent...")
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.agent
    
def get_playlist_agent(model: ChatGoogleGenerativeAI):
    """Factory function to create and initialize a PlaylistAgent."""
    pprint("Creating and initializing PlaylistAgent...")
    playlist_agent = PlaylistAgent(model)
    playlist_agent.initialize()
    return playlist_agent.get_agent()
        
    