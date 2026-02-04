"""
Test script to verify API endpoints work correctly
"""

import requests
import json
import sys
from pathlib import Path

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    print("Testing Stock Analysis API...")
    print(f"Base URL: {base_url}")
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test 3: Stock data endpoint
    print("\n3. Testing stock data endpoint...")
    try:
        response = requests.get(f"{base_url}/stocks/TCS.NS", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Symbol: {data.get('symbol')}")
            print(f"   Price: {data.get('current_price')}")
            print(f"   Date: {data.get('date')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 4: Prediction endpoint
    print("\n4. Testing prediction endpoint...")
    try:
        response = requests.post(
            f"{base_url}/tools/predict",
            json={"symbols": ["TCS.NS"], "horizon": "intraday"},
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Predictions count: {data.get('metadata', {}).get('count', 0)}")
            predictions = data.get('predictions', [])
            if predictions:
                pred = predictions[0]
                print(f"   Symbol: {pred.get('symbol')}")
                print(f"   Current Price: {pred.get('current_price')}")
                print(f"   Predicted Price: {pred.get('predicted_price')}")
                print(f"   Action: {pred.get('action')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("\nAPI test complete!")
    return True

if __name__ == "__main__":
    test_api()