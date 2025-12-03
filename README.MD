# Fortune 500 Marketing Agent: Partnerships/Collaborations

A FastAPI service that generates partnership marketing campaign posters by researching brands and events using Tavily, creating marketing briefs with OpenAI, and generating visual posters with Google Gemini. The service excludes brand-owned domains from search results to focus on new partnership opportunities and includes specific event details (dates, times, locations) in the generated posters.

## Features

- **Brand Research**: Uses Tavily to research brand voice, positioning, and partnership history (excluding brand-owned domains)
- **Event Discovery**: Finds upcoming events in target cities with specific dates, times, and locations
- **LLM-Powered Domain Exclusion**: Uses OpenAI to identify brand domains for exclusion from search results
- **Marketing Brief Generation**: Creates structured marketing briefs with partnership angles and visual prompts
- **Poster Generation**: Generates partnership campaign posters using Google Gemini image generation
- **Web Research Agent**: Includes a general-purpose web research agent endpoint using LangGraph

## Prerequisites

- Python 3.10+
- API keys:
  - **Tavily API Key**: For web search and event discovery
  - **OpenAI API Key**: For LLM-powered domain extraction and brief generation
  - **Gemini API Key**: For image generation w/ Nano Banana Pro

## Setup

1. Clone the repository and navigate to the project directory:
```bash
cd tavily-fde-assignment
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the repo root with your API keys:
```env
TAVILY_API_KEY=your_tavily_api_key
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Run

Start the FastAPI server:
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### POST `/partnership-campaign`

Generate a partnership marketing campaign poster for a brand in a target city.

**Request Body:**
```json
{
  "brand": "Nike",
  "city": "New York City",
  "num_events": 1,
  "visual_style": "bright illustration style with clear readable text"
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/partnership-campaign" \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Nike",
    "city": "New York City",
    "num_events": 1,
    "visual_style": "bright illustration style with clear readable text"
  }' | jq .
```

**Response:**
```json
{
  "brand": "Nike",
  "city": "New York City",
  "events": [
    {
      "title": "New York Fashion Week",
      "hook": "Premier fashion event in the heart of NYC",
      "partnership_angle": "Nike's athletic innovation meets high fashion",
      "url": "https://example.com/event",
      "date": "September 6, 2024",
      "time": "7:00 PM",
      "location": "Lincoln Center, New York, NY"
    }
  ],
  "brand_summary": "Nike embodies athletic excellence and innovation...",
  "visual_prompt": "A vibrant poster featuring...",
  "overlay_copy": "Join Nike at New York Fashion Week | Get Tickets",
  "poster_path": "marketing_assets/nike_new_york_city_abcd1234.png"
}
```

The generated poster PNG is saved to the `marketing_assets/` directory.

### POST `/run` (existing code from Tavily take-home)

Execute a general-purpose web research agent query using LangGraph and Tavily tools.

**Request Body:**
```json
{
  "query": "What are the latest developments in AI?",
  "thread_id": "optional-thread-id"
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in AI?",
    "thread_id": null
  }'
```

**Response:**
```json
{
  "query": "What are the latest developments in AI?",
  "output": "Based on recent research...",
  "thread_id": "generated-thread-id"
}
```

## How It Works

1. **Domain Exclusion**: The service uses OpenAI to identify the top 5 official domains for the brand, which are then excluded from Tavily search results to focus on third-party sources and new partnership opportunities.

2. **Brand Research**: Tavily performs advanced searches for brand information (voice, positioning, products) from external sources with:
   - **Full page content extraction** (`include_raw_content="markdown"`) for comprehensive brand insights
   - **Maximum content chunks** (`chunks_per_source=3`) for detailed information per source
   - **Advanced answers** for better brand summaries
   - Brand-owned domains automatically excluded

3. **Event Discovery**: Tavily searches for upcoming events in the target city with optimized parameters:
   - **Geographic filtering** using `country` parameter to focus on the target city's country
   - **Time-based filtering** (`time_range="month"`) to ensure events are upcoming
   - **Full page content** for accurate extraction of dates, times, and locations
   - **Advanced search depth** for better event discovery and relevance
   - Brand-owned domains excluded to find new partnership opportunities

4. **Brief Generation**: OpenAI generates a structured marketing brief including:
   - Brand summary
   - Event details with dates, times, and locations (extracted from full page content)
   - Visual prompt for image generation
   - Overlay copy for the poster with event details prominently featured

5. **Poster Generation**: Google Gemini w/ Nano Banana generates a visual poster based on the brief, incorporating brand elements, city landmarks, and event details.

## Technologies Used

- **FastAPI**: Web framework for the API
- **`langchain-tavily`**: Tavily integration for web search and event discovery (Tavily Search)
- **`langchain-openai`**: OpenAI integration for LLM-powered operations
- **google-genai**: Google Gemini integration for image generation
- **LangGraph**: For the web research agent workflow

## Project Structure

```
tavily-fde-assignment/
├── app.py                  # Main FastAPI application
├── requirements.txt        # Python dependencies
├── README.MD               # This file
├── .env                    # API keys (populate this)
└── marketing_assets/       # Store generated poster images
```

## Tavily Search Optimizations

The service uses several Tavily API optimizations for better search results:

- **Advanced Search Depth**: Uses `search_depth="advanced"` (2 API credits) for better relevance and content quality
- **Full Content Extraction**: `include_raw_content="markdown"` retrieves complete page content for accurate event detail extraction
- **Maximum Content Chunks**: `chunks_per_source=3` provides up to 3 content chunks (500 chars each) per source
- **Geographic Filtering**: Automatically maps cities to countries using `country` parameter to focus results geographically
- **Time-Based Filtering**: `time_range="month"` ensures events are upcoming and relevant
- **Advanced Answers**: Uses `include_answer="advanced"` for more comprehensive summaries

### API Credit Usage

- **Brand Research**: 2 credits (advanced search)
- **Event Search**: 2 credits (advanced search)
- **Total per request**: ~4 Tavily API credits (excluding OpenAI and Gemini API calls)

## Notes

- Generated posters are saved in the `marketing_assets/` directory
- The service automatically creates the `marketing_assets/` directory if it doesn't exist
- Event details (dates, times, locations) are extracted from full page content and included in the poster
- Brand domains are dynamically identified using LLM to ensure accurate exclusion
- Geographic filtering supports 30+ major cities across multiple countries
- The service prioritizes `raw_content` when available for better detail extraction

## Troubleshooting

### API Rate Limits

If you encounter errors like `503 UNAVAILABLE` or `model is overloaded`:
- **Gemini API**: Wait a few minutes and retry, or check your API quota in Google Cloud Console
- **OpenAI API**: Check your rate limits and usage in the OpenAI dashboard
- **Tavily API**: Monitor your credit usage at [app.tavily.com](https://app.tavily.com)

### Common Issues

- **Missing event details**: Ensure your Tavily API key has sufficient credits and the search is finding relevant events
- **Domain exclusion not working**: Verify the LLM is correctly identifying brand domains (check logs)
- **Geographic filtering**: If a city isn't recognized, the service will still work but without country-specific filtering
