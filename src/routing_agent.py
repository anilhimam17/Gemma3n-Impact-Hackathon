from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import ReActAgent, AgentStream, ToolCallResult

from src.config import settings
from src.query_engine import QueryEngine
from src.response_structures import ResponseTypes, ToolInput, SimpleResponse


# Global Configurations
Settings.llm = Ollama(
    model=settings.llm_model_name, request_timeout=300.0,
    context_window=30000, additional_kwargs={"num_predict": 4096}
)


class RoutingAgent:
    """Class that implements the Response Routing Agent."""
    
    def __init__(self, file_path: str):
        self.query_engine = QueryEngine(file_path)
        self.routing_agent = self.construct_routing_agent()

    def construct_routing_agent(self):
        """Constructs the routing agent by wrapping the tools into FunctionTools."""
        
        # Wrapping FunctionTools
        research_tool = FunctionTool.from_defaults(fn=self.execute_research_query, fn_schema=ToolInput)
        summary_tool = FunctionTool.from_defaults(fn=self.execute_summary_query, fn_schema=ToolInput)
        simple_tool = FunctionTool.from_defaults(fn=self.execute_simple_query, fn_schema=ToolInput)

        return ReActAgent(
            tools=[research_tool, summary_tool, simple_tool],
            llm=Settings.llm, verbose=True
        )

    async def resolve_route(self, user_prompt: str):
        """Resolve the route to be used by for generating a response."""
        response_json = None
        response_type = None

        handler = self.routing_agent.run(user_prompt)
        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
                response_json = ev.tool_output
                if ev.tool_name == "execute_research_query":
                    response_type = ResponseTypes.RESEARCH
                elif ev.tool_name == "execute_summary_query":
                    response_type = ResponseTypes.SUMMARY
                elif ev.tool_name == "execute_simple_query":
                    response_type = ResponseTypes.SIMPLE

                if response_json and response_type:
                    return response_json, response_type
                
            if isinstance(ev, AgentStream):
                print(f"{ev.delta}", end="", flush=True)
        
        final_answer = await handler
        final_response = SimpleResponse(answer=str(final_answer))
        return final_response.model_dump_json(indent=4), ResponseTypes.SIMPLE
    
    def extract_query(self, query: str | None = None, properties: dict | None = None, **kwargs):
        if query:
            return query
        if properties and "query" in properties.keys():
            return properties["query"]
        if kwargs.get("query"):
            return kwargs["query"]
        return ""

    def execute_research_query(self, query: str | None = None, properties: dict | None = None, **kwargs):
        """Use this tool for questions that require looking up information in the document.
        It provides a detailed answer with citations."""
        
        extracted_query = self.extract_query(query, properties, **kwargs)
        research_response = self.query_engine.run_query(extracted_query, ResponseTypes.RESEARCH)
        return research_response

    def execute_summary_query(self, query: str | None = None, properties: dict | None = None, **kwargs):
        """Use this tool when the user asks for a summary of the conversation history.
        This query will specifically state 'Generate a summary'."""

        extracted_query = self.extract_query(query, properties, **kwargs)
        summary_response = self.query_engine.run_query(extracted_query, ResponseTypes.SUMMARY)
        return summary_response
        
    def execute_simple_query(self, query: str | None = None, properties: dict | None = None, **kwargs):
        """Use this tool for simple greetings, conversation fillers or questions that 
        do not require explicitly looking up the document."""

        extracted_query = self.extract_query(query, properties, **kwargs)
        simple_response = self.query_engine.run_query(extracted_query, ResponseTypes.SIMPLE)
        return simple_response