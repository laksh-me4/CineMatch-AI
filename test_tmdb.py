import requests

API_KEY = "10c55f2fbb7fc92c35de78875a82b3f8"

url = "https://api.themoviedb.org/3/search/movie"
params = {
    "api_key": API_KEY,
    "query": "Heat"
}

response = requests.get(url, params=params)
print(response.status_code)
print(response.json())
