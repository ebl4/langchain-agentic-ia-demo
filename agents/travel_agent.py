from langchain.agents import create_agent
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

RETRYABLE_MCP_CODES = {-32603}

class RetryMCPInterceptor:
    """Intercept MCP tool calls: retry transient failures, surface all errors gracefully.

    - Retryable McpError codes (e.g. -32603): retry with exponential backoff.
    - Non-retryable McpError codes (e.g. -32602): return error message immediately.
    - Any other exception (fetch failed, network errors, etc.): retry then return error message.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(code {exc.error.code}, attempt {attempt+1}/{self.max_retries}): {exc}")
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(attempt {attempt+1}/{self.max_retries}): {exc}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        print(f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )


class TravelAgent:
    """Travel agent for finding flights to wedding destinations."""

    def __init__(self):
        self.client = None
        self.agent = None

    async def initialize(self):
        """Initialize the MCP client and create the agent."""
        self.client = MultiServerMCPClient(
            {
                "travel_server": {
                        "transport": "streamable_http",
                        "url": "https://mcp.kiwi.com"
                    }
            },
            tool_interceptors=[RetryMCPInterceptor()],
        )

        tools = await self.client.get_tools()

        self.agent = create_agent(
            model="gpt-5-nano",
            tools=tools,
            system_prompt="""
            You are a travel agent. Search for flights to the desired destination wedding location.
            You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
            - Price (lowest, economy class)
            - Duration (shortest)
            - Date (time of year which you believe is best for a wedding at this location)
            To make things easy, only look for one ticket, one way.
            You may need to make multiple searches to iteratively find the best options.
            You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
            If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
            Once you have found the best options, let the user know your shortlist of options.
            """
        )

    async def close(self):
        """Close the MCP client connection."""
        if self.client:
            await self.client.close()

    def get_agent(self):
        """Get the initialized agent."""
        if self.agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.agent


async def get_travel_agent():
    """Factory function to create and initialize a TravelAgent."""
    travel_agent = TravelAgent()
    await travel_agent.initialize()
    return travel_agent