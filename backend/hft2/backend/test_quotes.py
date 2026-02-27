import os
import requests
import json
from dotenv import load_dotenv

load_dotenv(os.path.join("..", "env"))
client_id = os.environ.get("FYERS_APP_ID")
access_token = os.environ.get("FYERS_ACCESS_TOKEN")

headers = {
    "Authorization": f"{client_id}:{access_token}",
    "Content-Type": "application/json"
}

urls = [
    ("v3_docs", "https://api.fyers.in/data-rest/v3/quotes?symbols=NSE:TATACOMM-EQ"),
    ("t1_data", "https://api-t1.fyers.in/data/quotes?symbols=NSE:TATACOMM-EQ"),
    ("t1_v3_data", "https://api-t1.fyers.in/api/v3/data/quotes?symbols=NSE:TATACOMM-EQ"),
    ("t1_data_rest", "https://api-t1.fyers.in/data-rest/v3/quotes?symbols=NSE:TATACOMM-EQ")
]

results = {}
for name, url in urls:
    try:
        res = requests.get(url, headers=headers)
        results[name] = {"status": res.status_code, "text": res.text[:200]}
    except Exception as e:
        results[name] = {"error": str(e)}

with open("quotes_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
