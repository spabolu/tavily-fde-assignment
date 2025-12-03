import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.genai as genai
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.genai import types
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
            tavily_api_key=api_key,
            search_depth="advanced",
        )
        
        extract = TavilyExtract(
            tavily_api_key=api_key,
        )
        
        crawl = TavilyCrawl(
            tavily_api_key=api_key, 
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


class MarketingRequest(BaseModel):
    brand: str
    city: str
    num_events: int = 3
    visual_style: str = "bright illustration style with clear readable text"
    thread_id: Optional[str] = None


class TavilyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.search_url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": True,
        }
        response = requests.post(self.search_url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return {
            "query": query,
            "results": data.get("results", []),
            "answer": data.get("answer"),
        }

    @staticmethod
    def normalize_results(search_payload: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        items = []
        for entry in search_payload.get("results", []):
            items.append(
                {
                    "title": entry.get("title"),
                    "url": entry.get("url"),
                    "summary": entry.get("content") or entry.get("snippet"),
                }
            )
            if len(items) >= limit:
                break
        return items


class MarketingBriefBuilder:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def build_brief(
        self,
        brand: str,
        city: str,
        brand_research: Dict[str, Any],
        events: List[Dict[str, Any]],
        visual_style: str,
    ) -> Dict[str, Any]:
        prompt = f"""
You are a partnerships marketing strategist. Create a concise JSON brief for a poster that pairs the brand with city events.
Brand: {brand}
City: {city}
Brand research (marketing perspective): {json.dumps(brand_research, ensure_ascii=False)}
Candidate events: {json.dumps(events, ensure_ascii=False)}
Visual style: {visual_style}

Return a JSON object with keys:
- brand_summary: 2 sentences on voice, audience, and iconic product to feature.
- events: list of events (title, hook, partnership_angle, url) limited to {len(events)} items.
- visual_prompt: <=120 words prompt for an image generator, must mention the brand logo placement, the iconic product, the city atmosphere, and blend event themes.
- overlay_copy: short copy (headline + CTA) to render on the poster.
No prose outside the JSON.
"""
        message = self.llm.invoke(prompt)
        try:
            return json.loads(message.content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM did not return valid JSON: {message.content}") from exc


class GeminiImageGenerator:
    """
    Wrapper around Gemini image generation (Nano Banana / Pro) using the official google-genai SDK.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3-pro-image-preview",
        aspect_ratio: str = "16:9",
        image_size: str = "1K",
    ):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.image_config = types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        )
        self.config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=self.image_config,
        )

    def generate_image_bytes(self, prompt: str) -> bytes:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt],
            config=self.config,
        )
        for part in response.parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                data = inline_data.data
                if isinstance(data, str):
                    return base64.b64decode(data)
                return data
        raise ValueError("Gemini response missing inline image data")

    def save_image(self, image_bytes: bytes, brand: str, city: str) -> str:
        assets_dir = Path("marketing_assets")
        assets_dir.mkdir(exist_ok=True)
        safe_brand = brand.lower().replace(" ", "_")
        safe_city = city.lower().replace(" ", "_")
        filename = f"{safe_brand}_{safe_city}_{uuid.uuid4().hex[:8]}.png"
        path = assets_dir / filename
        path.write_bytes(image_bytes)
        return str(path)

class QueryRequest(BaseModel):
    query: str
    thread_id: str = None


def ensure_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise HTTPException(status_code=500, detail=f"{var_name} not set")
    return value


def build_marketing_assets(request: MarketingRequest) -> Dict[str, Any]:
    tavily_api_key = ensure_env("TAVILY_API_KEY")
    openai_api_key = ensure_env("OPENAI_API_KEY")
    gemini_api_key = ensure_env("GEMINI_API_KEY")

    tavily_client = TavilyClient(api_key=tavily_api_key)
    brand_research = tavily_client.search(
        query=f"{request.brand} brand voice, marketing positioning, product heroes, logo usage, partnership history",
        max_results=6,
    )
    events_raw = tavily_client.search(
        query=f"upcoming events in {request.city} suitable for {request.brand} partnerships or collaborations",
        max_results=max(request.num_events * 2, 6),
    )
    events = tavily_client.normalize_results(events_raw, limit=request.num_events)

    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", api_key=openai_api_key)
    brief_builder = MarketingBriefBuilder(llm)
    brief = brief_builder.build_brief(
        brand=request.brand,
        city=request.city,
        brand_research=brand_research,
        events=events,
        visual_style=request.visual_style,
    )

    image_prompt = (
        f"{brief.get('visual_prompt', '')}\n"
        f"Include {request.brand} logo and signature product prominently. "
        f"Blend {request.city} landmarks. Text overlay: {brief.get('overlay_copy', '')}."
    )
    gemini = GeminiImageGenerator(api_key=gemini_api_key)
    image_bytes = gemini.generate_image_bytes(image_prompt)
    poster_path = gemini.save_image(image_bytes, brand=request.brand, city=request.city)

    return {
        "brand": request.brand,
        "city": request.city,
        "events": brief.get("events", events),
        "brand_summary": brief.get("brand_summary"),
        "visual_prompt": image_prompt,
        "overlay_copy": brief.get("overlay_copy"),
        "poster_path": poster_path,
    }


@app.post("/partnership-campaign")
async def run_partnership_agent(request: MarketingRequest):
    """
    Build a partnership/marketing campaign poster for a brand in a target city.
    """
    try:
        payload = build_marketing_assets(request)
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/run")
async def run_agent(request: QueryRequest):
    """
    Execute the agent on a query string.
    """
    try:
        tavily_api_key = ensure_env("TAVILY_API_KEY")
        openai_api_key = ensure_env("OPENAI_API_KEY")
        
        # Use the llm of your choice
        llm = ChatOpenAI(model="gpt-5.1-2025-11-13", api_key=openai_api_key)
        
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
