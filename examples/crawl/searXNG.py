import os
import requests

class SearXNGClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.searxng_api_key = os.getenv('SEARXNG_API_KEY')

    async def search(self, query, categories=None, engines=None, language=None, pageno=1, time_range=None, output_format='json'):
        params = {
            'q': query,
            'categories': categories,
            'engines': engines,
            'language': language,
            'pageno': pageno,
            'time_range': time_range,
            'format': output_format
        }
        response = requests.get(f"{self.base_url}/search", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def search2(self, query):
        params = {"q": query, "format": "json"}
        custom_URL = os.getenv('SEARXNG_API_URL')
        headers = (
            {"Authorization": f"Bearer {self.searxng_api_key}"}
            if self.searxng_api_key
            else {}
        )
        response = requests.get(
            custom_URL, headers=headers, params=params
        )
        results = response.json()
        return results
    
if __name__ == "__main__":
    base_url = os.getenv('SEARXNG_URL')
    client = SearXNGClient(base_url)
    
    query = "SearXNG"
    categories = "general"
    engines = "google,bing"
    language = "en"
    pageno = 1
    time_range = "year"
    output_format = "json"
    
    results = client.search2(query)
    print(results)