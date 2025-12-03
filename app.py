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
from langchain_tavily import TavilySearch
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Partnership Campaign API")


class MarketingRequest(BaseModel):
    brand: str
    city: str
    num_events: int = 3
    visual_style: str = "bright illustration style with clear readable text"


def get_country_from_city(city: str) -> Optional[str]:
    """Map city name to country for geographic filtering."""
    city_country_map = {
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
        "london": "united kingdom",
        "manchester": "united kingdom",
        "birmingham": "united kingdom",
        "edinburgh": "united kingdom",
        "toronto": "canada",
        "vancouver": "canada",
        "montreal": "canada",
        "sydney": "australia",
        "melbourne": "australia",
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
    if city_lower in city_country_map:
        return city_country_map[city_lower]
    
    for city_key, country in city_country_map.items():
        if city_key in city_lower or city_lower in city_key:
            return country
    
    return None


def normalize_tavily_results(results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Normalize Tavily results, prioritizing raw_content when available."""
    items = []
    for entry in results[:limit]:
        raw_content = entry.get("raw_content", "")
        content = entry.get("content") or entry.get("snippet") or ""
        full_content = raw_content if raw_content else content
        
        items.append({
            "title": entry.get("title", ""),
            "url": entry.get("url", ""),
            "summary": content,
            "full_content": full_content,
            "raw_content": raw_content,
        })
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
        print(f"[debug] Building marketing brief for brand='{brand}' in city='{city}' with {len(events)} events")
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
        print("[debug] LLM returned brief content")
        try:
            return json.loads(message.content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM did not return valid JSON: {message.content}") from exc


class GeminiImageGenerator:
    """Generate images using Google Gemini."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3-pro-image-preview", # nano banana pro
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
        """Generate image bytes from text prompt."""
        print("[debug] Generating image with Gemini")
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
        """Save generated image to marketing_assets directory."""
        assets_dir = Path("marketing_assets")
        assets_dir.mkdir(exist_ok=True)
        safe_brand = brand.lower().replace(" ", "_")
        safe_city = city.lower().replace(" ", "_")
        filename = f"{safe_brand}_{safe_city}_{uuid.uuid4().hex[:8]}.png"
        path = assets_dir / filename
        path.write_bytes(image_bytes)
        return str(path)


def get_brand_domains(brand: str, llm: ChatOpenAI) -> List[str]:
    """Get top 5 brand domains using LLM, with fallback heuristic."""
    prompt = f"""What are the top 5 most important official website domains for the brand "{brand}"?
    
Return ONLY a JSON array of domain strings (without www prefix, just the base domains).
Include the main corporate website and any major brand-specific domains.
Do not include social media domains or third-party sites.

Example format for "Nike" assuming the brand has the following domains:
["nike.com", "nike.net", "nike.org"]

Return only the JSON array, no other text:"""

    try:
        print(f"[debug] Requesting brand domains for '{brand}' from LLM")
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        domains = json.loads(content)
        if not isinstance(domains, list):
            raise ValueError("LLM did not return a list")
        
        domains = domains[:5]
        result = []
        for domain in domains:
            domain = domain.strip().lower()
            if domain.startswith("www."):
                domain = domain[4:]
            result.append(domain)
            result.append(f"www.{domain}")
        print(f"[debug] LLM provided domains: {result}")
        return result
    except (json.JSONDecodeError, ValueError, AttributeError):
        print(f"[debug] Falling back to heuristic domain generation for '{brand}'")
        # Fallback to simple heuristic
        normalized = brand.lower().strip()
        for suffix in [' inc', ' inc.', ' llc', ' ltd', ' ltd.', ' corp', ' corp.', ' company', ' co', ' co.']:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        base_domains = [normalized.replace(' ', '-'), normalized.replace(' ', '')] if ' ' in normalized else [normalized]
        
        result = []
        for base in base_domains[:2]:
            result.append(f"{base}.com")
            result.append(f"www.{base}.com")
        
        return result[:10]


def ensure_env(var_name: str) -> str:
    """Get environment variable or raise error."""
    value = os.getenv(var_name)
    if not value:
        raise HTTPException(status_code=500, detail=f"{var_name} not set")
    print(f"[debug] Loaded environment variable: {var_name}")
    return value


def build_marketing_assets(request: MarketingRequest) -> Dict[str, Any]:
    """Build marketing campaign assets: research, brief, and poster."""
    print(f"[debug] Starting build_marketing_assets for brand='{request.brand}', city='{request.city}'")
    tavily_api_key = ensure_env("TAVILY_API_KEY")
    openai_api_key = ensure_env("OPENAI_API_KEY")
    gemini_api_key = ensure_env("GEMINI_API_KEY")

    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", api_key=openai_api_key)
    brand_domains = get_brand_domains(request.brand, llm)
    print(f"[debug] Brand domains determined: {brand_domains}")
    country = get_country_from_city(request.city)
    print(f"[debug] Country derived from city '{request.city}': {country or 'not found'}")
    
    # Brand research search
    brand_search = TavilySearch(
        tavily_api_key=tavily_api_key,
        search_depth="advanced",
        max_results=6,
        exclude_domains=brand_domains,
        include_answer="advanced",
        include_raw_content="markdown",
        chunks_per_source=3,
    )
    brand_result = brand_search.invoke(
        f"{request.brand} brand voice, marketing positioning, product heroes, logo usage, partnership history"
    )
    print(f"[debug] Brand research results fetched: {len(brand_result.get('results', [])) if isinstance(brand_result, dict) else 0} items")
    brand_research = {
        "query": f"{request.brand} brand voice, marketing positioning, product heroes, logo usage, partnership history",
        "results": brand_result.get("results", []) if isinstance(brand_result, dict) else [],
        "answer": brand_result.get("answer", "") if isinstance(brand_result, dict) else "",
    }
    
    # Event search
    events_params = {
        "tavily_api_key": tavily_api_key,
        "search_depth": "advanced",
        "max_results": max(request.num_events * 2, 6),
        "exclude_domains": brand_domains,
        "include_answer": "advanced",
        "include_raw_content": "markdown",
        "chunks_per_source": 3,
        "time_range": "month",
    }
    if country:
        events_params["country"] = country
    
    events_search = TavilySearch(**events_params)
    events_result = events_search.invoke(
        f"upcoming events in {request.city} with dates, times, and locations suitable for {request.brand} partnerships or collaborations. Include specific event details like event date, start time, venue address, and location."
    )
    events_raw = events_result.get("results", []) if isinstance(events_result, dict) else []
    events = normalize_tavily_results(events_raw, limit=request.num_events)
    print(f"[debug] Events gathered: requested {request.num_events}, normalized {len(events)}")

    # Generate brief
    brief_builder = MarketingBriefBuilder(llm)
    brief = brief_builder.build_brief(
        brand=request.brand,
        city=request.city,
        brand_research=brand_research,
        events=events,
        visual_style=request.visual_style,
    )
    print(f"[debug] Brief generated with {len(brief.get('events', [])) if brief.get('events') else 0} events")

    # Build image prompt with event details
    events_info = ""
    if brief.get("events"):
        event_details = []
        for event in brief.get("events", []):
            parts = [event.get("title", "")]
            if event.get("date"):
                parts.append(f"Date: {event['date']}")
            if event.get("time"):
                parts.append(f"Time: {event['time']}")
            if event.get("location"):
                parts.append(f"Location: {event['location']}")
            event_details.append(" | ".join(parts))
        events_info = "\nEvent Details:\n" + "\n".join(event_details)
    
    image_prompt = (
        f"{brief.get('visual_prompt', '')}\n"
        f"Include {request.brand} logo and signature product prominently. "
        f"Blend {request.city} landmarks. "
        f"{events_info}\n"
        f"Text overlay: {brief.get('overlay_copy', '')}."
    )
    print("[debug] Image prompt prepared")

    # Generate and save poster
    gemini = GeminiImageGenerator(api_key=gemini_api_key)
    image_bytes = gemini.generate_image_bytes(image_prompt)
    poster_path = gemini.save_image(image_bytes, brand=request.brand, city=request.city)
    print(f"[debug] Poster saved to {poster_path}")

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
    """Generate partnership marketing campaign poster."""
    try:
        return build_marketing_assets(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
