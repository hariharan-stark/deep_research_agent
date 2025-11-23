from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from src.config import config

import json

def get_google_search_tool():
    search = GoogleSearchAPIWrapper(
        google_api_key=config.GOOGLE_API_KEY,
        google_cse_id=config.GOOGLE_CSE_ID,
    )
    
    def search_with_links(query: str):
        """
        Performs a Google search and returns structured results with titles and links.
        """
        results = search.results(query, num_results=5)
        structured_results = []
        for res in results:
            structured_results.append({
                "title": res.get("title"),
                "link": res.get("link"),
                "snippet": res.get("snippet")
            })
        return json.dumps(structured_results, indent=2)
    
    return Tool(
        name="google_search",
        description="Search Google for recent results. Returns JSON with title, link, and snippet.",
        func=search_with_links,
    )
