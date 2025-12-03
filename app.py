import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

load_dotenv()


class WebAgent:
    def __init__(self, checkpointer: MemorySaver = None):
        self.checkpointer = checkpointer

    def build_graph(self, api_key: str, llm: ChatOpenAI, prompt: str):
        """
        Build and compile the LangGraph workflow.
        
        Args:
            api_key: Tavily API key
            llm: Main LLM for the agent
            prompt: System prompt
        """
        if not api_key:
            raise ValueError("Error: Tavily API key not provided.")
        
        # Create the tools with the API key
        search = TavilySearch(
            api_key=api_key,
            search_depth="advanced",
        )
        
        extract = TavilyExtract(
            api_key=api_key,
        )
        
        crawl = TavilyCrawl(
            api_key=api_key, 
            limit=20
        )
        
        return create_react_agent(
            prompt=prompt,
            model=llm,
            tools=[search, extract, crawl],
            checkpointer=self.checkpointer,
        )


app = FastAPI(title="Web Agent API")

# Global memory
memory = MemorySaver()

class QueryRequest(BaseModel):
    query: str
    thread_id: str = None

@app.post("/run")
async def run_agent(request: QueryRequest):
    """
    Execute the agent on a query string.
    """
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not tavily_api_key:
            raise HTTPException(status_code=500, detail="TAVILY_API_KEY not set")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        
        # Use the llm of your choice
        llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", api_key=openai_api_key)
        
        # Handle thread_id
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Build the graph
        agent = WebAgent(checkpointer=memory)
        graph = agent.build_graph(
            api_key=tavily_api_key,
            llm=llm,
            prompt="You are a helpful web research assistant."
        )
        
        # Execute the graph
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke({"messages": [("user", request.query)]}, config=config)
        
        # Extract the final message
        output = result["messages"][-1].content
        
        return {"query": request.query, "output": output, "thread_id": thread_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
