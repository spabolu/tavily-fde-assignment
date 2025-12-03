import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.genai as genai
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


def get_country_from_city(city: str) -> Optional[str]:
    """
    Map city name to country for Tavily's country parameter.
    Returns country name in lowercase format expected by Tavily API.
    """
    # Common city-to-country mappings for major cities
    city_country_map = {
        # United States
        "new york": "united states",
        "new york city": "united states",
        "nyc": "united states",
        "los angeles": "united states",
        "chicago": "united states",
        "san francisco": "united states",
        "boston": "united states",
        "miami": "united states",
        "seattle": "united states",
        "austin": "united states",
        "atlanta": "united states",
        "dallas": "united states",
        "houston": "united states",
        "philadelphia": "united states",
        "washington": "united states",
        "washington dc": "united states",
        # United Kingdom
        "london": "united kingdom",
        "manchester": "united kingdom",
        "birmingham": "united kingdom",
        "edinburgh": "united kingdom",
        # Canada
        "toronto": "canada",
        "vancouver": "canada",
        "montreal": "canada",
        # Australia
        "sydney": "australia",
        "melbourne": "australia",
        # Other major cities
        "paris": "france",
        "berlin": "germany",
        "madrid": "spain",
        "rome": "italy",
        "amsterdam": "netherlands",
        "tokyo": "japan",
        "singapore": "singapore",
        "hong kong": "china",
        "dubai": "united arab emirates",
        "mumbai": "india",
        "delhi": "india",
        "bangalore": "india",
        "sao paulo": "brazil",
        "rio de janeiro": "brazil",
        "mexico city": "mexico",
    }
    
    city_lower = city.lower().strip()
    # Try exact match first
    if city_lower in city_country_map:
        return city_country_map[city_lower]
    
    # Try partial match (e.g., "New York" contains "new york")
    for city_key, country in city_country_map.items():
        if city_key in city_lower or city_lower in city_key:
            return country
    
    return None  # Return None if no match found


def normalize_tavily_results(results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """
    Normalize Tavily search results to a consistent format.
    Prioritizes raw_content if available for better detail extraction.
    
    Args:
        results: List of Tavily result dictionaries
        limit: Maximum number of results to return
        
    Returns:
        List of normalized event dictionaries
    """
    items = []
    for entry in results[:limit]:
        # Prioritize raw_content for full page content, fallback to content/snippet
        raw_content = entry.get("raw_content", "")
        content = entry.get("content") or entry.get("snippet") or ""
        
        # Use raw_content if available (more detailed), otherwise use content
        full_content = raw_content if raw_content else content
        
        items.append(
            {
                "title": entry.get("title", ""),
                "url": entry.get("url", ""),
                "summary": content,  # Keep summary as the snippet
                "full_content": full_content,  # Full content for LLM extraction
                "raw_content": raw_content,  # Explicitly include raw_content if available
            }
        )
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
- events: list of events limited to {len(events)} items. Each event must include:
  - title: event name
  - hook: compelling one-liner about the event
  - partnership_angle: why this brand-event pairing works
  - url: event URL
  - date: specific event date (extract from event content, format as "Month Day, Year" or "MM/DD/YYYY")
  - time: event start time if available (format as "HH:MM AM/PM" or "HH:MM")
  - location: specific venue name and address if available, otherwise neighborhood/area
- visual_prompt: <=120 words prompt for an image generator, must mention the brand logo placement, the iconic product, the city atmosphere, event dates/times/locations, and blend event themes.
- overlay_copy: short copy (headline + CTA) to render on the poster. Include event dates and locations prominently.

IMPORTANT: Extract specific dates, times, and locations from the event content. If details are missing, indicate "TBD" or "Check website".
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


def get_brand_domains(brand: str, llm: ChatOpenAI) -> List[str]:
    """
    Use LLM to get the top 5 actual domain names for a brand to exclude from search results.
    Returns a list of domain patterns to exclude (e.g., ['nike.com', 'www.nike.com']).
    """
    prompt = f"""What are the top 5 most important official website domains for the brand "{brand}"?
    
Return ONLY a JSON array of domain strings (without www prefix, just the base domains).
Include the main corporate website and any major brand-specific domains.
Do not include social media domains or third-party sites.

Example format for "Nike":
["nike.com", "nike.net", "nike.org"]

Return only the JSON array, no other text:"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        # Parse JSON array
        domains = json.loads(content)
        
        # Validate it's a list of strings
        if not isinstance(domains, list):
            raise ValueError("LLM did not return a list")
        
        # Limit to top 5 and add www variants
        domains = domains[:5]
        
        # Add www variants for each domain
        result = []
        for domain in domains:
            domain = domain.strip().lower()
            # Remove www if present
            if domain.startswith("www."):
                domain = domain[4:]
            result.append(domain)
            result.append(f"www.{domain}")
        
        return result
    except (json.JSONDecodeError, ValueError, AttributeError):
        # Fallback: use simple heuristic if LLM fails
        normalized = brand.lower().strip()
        for suffix in [' inc', ' inc.', ' llc', ' ltd', ' ltd.', ' corp', ' corp.', ' company', ' co', ' co.']:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        if ' ' in normalized:
            base_domains = [normalized.replace(' ', '-'), normalized.replace(' ', '')]
        else:
            base_domains = [normalized]
        
        result = []
        for base in base_domains[:2]:  # Limit to 2 variants
            result.append(f"{base}.com")
            result.append(f"www.{base}.com")
        
        return result[:10]  # Return up to 10 domains


def ensure_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise HTTPException(status_code=500, detail=f"{var_name} not set")
    return value


def build_marketing_assets(request: MarketingRequest) -> Dict[str, Any]:
    tavily_api_key = ensure_env("TAVILY_API_KEY")
    openai_api_key = ensure_env("OPENAI_API_KEY")
    gemini_api_key = ensure_env("GEMINI_API_KEY")

    # Initialize LLM for domain extraction
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", api_key=openai_api_key)
    
    # Get brand domains to exclude from search results using LLM
    brand_domains = get_brand_domains(request.brand, llm)
    
    # Get country for geographic filtering
    country = get_country_from_city(request.city)
    
    # Use TavilySearch from langchain-tavily with optimized parameters for brand research
    brand_search = TavilySearch(
        tavily_api_key=tavily_api_key,
        search_depth="advanced",  # Advanced search for better relevance (2 credits)
        max_results=6,
        exclude_domains=brand_domains,
        include_answer="advanced",  # Use advanced answer for more detailed insights
        include_raw_content="markdown",  # Get full page content in markdown format
        chunks_per_source=3,  # Maximum chunks per source for comprehensive content
    )
    brand_research_result = brand_search.invoke(
        f"{request.brand} brand voice, marketing positioning, product heroes, logo usage, partnership history"
    )
    # TavilySearch returns a dict with 'results' and 'answer' keys
    brand_research = {
        "query": f"{request.brand} brand voice, marketing positioning, product heroes, logo usage, partnership history",
        "results": brand_research_result.get("results", []) if isinstance(brand_research_result, dict) else [],
        "answer": brand_research_result.get("answer", "") if isinstance(brand_research_result, dict) else "",
    }
    
    # Configure event search with optimized parameters for finding upcoming events
    events_search_params = {
        "tavily_api_key": tavily_api_key,
        "search_depth": "advanced",  # Advanced search for better event discovery (2 credits)
        "max_results": max(request.num_events * 2, 6),
        "exclude_domains": brand_domains,
        "include_answer": "advanced",  # Advanced answer for better event summaries
        "include_raw_content": "markdown",  # Full page content for extracting dates/times/locations
        "chunks_per_source": 3,  # Maximum chunks for comprehensive event details
        "time_range": "month",  # Focus on events in the next month
    }
    
    # Add country filter if we can map the city to a country
    if country:
        events_search_params["country"] = country
    
    events_search = TavilySearch(**events_search_params)
    events_result = events_search.invoke(
        f"upcoming events in {request.city} with dates, times, and locations suitable for {request.brand} partnerships or collaborations. Include specific event details like event date, start time, venue address, and location."
    )
    # Extract results list from TavilySearch response
    events_raw = events_result.get("results", []) if isinstance(events_result, dict) else []
    events = normalize_tavily_results(events_raw, limit=request.num_events)

    # Reuse the same LLM instance for brief building
    brief_builder = MarketingBriefBuilder(llm)
    brief = brief_builder.build_brief(
        brand=request.brand,
        city=request.city,
        brand_research=brand_research,
        events=events,
        visual_style=request.visual_style,
    )

    # Build image prompt with event details
    events_info = ""
    if brief.get("events"):
        event_details = []
        for event in brief.get("events", []):
            detail_parts = [event.get("title", "")]
            if event.get("date"):
                detail_parts.append(f"Date: {event['date']}")
            if event.get("time"):
                detail_parts.append(f"Time: {event['time']}")
            if event.get("location"):
                detail_parts.append(f"Location: {event['location']}")
            event_details.append(" | ".join(detail_parts))
        events_info = "\nEvent Details:\n" + "\n".join(event_details)
    
    image_prompt = (
        f"{brief.get('visual_prompt', '')}\n"
        f"Include {request.brand} logo and signature product prominently. "
        f"Blend {request.city} landmarks. "
        f"{events_info}\n"
        f"Text overlay: {brief.get('overlay_copy', '')}."
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
