import requests
import json

url = 'http://localhost:5000/api'
r = requests.post(url, json={'Price': 2, 'Rating': 9, 'Reviews': 6})
print(r.json())

