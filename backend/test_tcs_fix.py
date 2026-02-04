#!/usr/bin/env python3
"""
Test TCS.NS prediction after cache cleanup
"""

import requests
import json
import time

def test_tcs_prediction():
    """Test TCS.NS prediction to verify the datetime error is fixed"""
    
    base_url = "http://127.0.0.1:8000"
    
    print("TESTING TCS.NS PREDICTION AFTER CACHE CLEANUP")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n[TEST 1] Backend Health Check...")
    try:
        response = requests.get(f"{base_url}/tools/health", timeout=10)
        if response.status_code == 200:
            print("[OK] Backend is healthy")
        else:
            print(f"[ERROR] Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Cannot connect to backend: {e}")
        return False
    
    # Test 2: Force fetch fresh data for TCS.NS
    print("\n[TEST 2] Force fetching fresh data for TCS.NS...")
    try:
        fetch_payload = {
            "symbols": ["TCS.NS"],
            "period": "2y",
            "refresh": True,  # Force refresh
            "include_features": True
        }
        
        response = requests.post(
            f"{base_url}/tools/fetch_data",
            json=fetch_payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            tcs_result = result['results'][0]
            print(f"[OK] Data fetch status: {tcs_result['status']}")
            print(f"     Rows: {tcs_result.get('rows', 'N/A')}")
            print(f"     Date range: {tcs_result.get('date_range', 'N/A')}")
            print(f"     Latest price: {tcs_result.get('latest_price', 'N/A')}")
        else:
            print(f"[ERROR] Data fetch failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Data fetch error: {e}")
        return False
    
    # Test 3: Try prediction
    print("\n[TEST 3] Testing TCS.NS prediction...")
    try:
        predict_payload = {
            "symbols": ["TCS.NS"],
            "horizon": "intraday"
        }
        
        response = requests.post(
            f"{base_url}/tools/predict",
            json=predict_payload,
            timeout=120  # Longer timeout for prediction
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['predictions'][0]
            
            if 'error' in prediction:
                print(f"[ERROR] Prediction failed: {prediction['error']}")
                return False
            else:
                print("[SUCCESS] Prediction completed!")
                print(f"   Symbol: {prediction['symbol']}")
                print(f"   Action: {prediction['action']}")
                print(f"   Current Price: {prediction['current_price']}")
                print(f"   Predicted Price: {prediction['predicted_price']}")
                print(f"   Predicted Return: {prediction['predicted_return']:.2f}%")
                print(f"   Confidence: {prediction['confidence']:.4f}")
                return True
        else:
            print(f"[ERROR] Prediction request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return False

if __name__ == "__main__":
    success = test_tcs_prediction()
    
    if success:
        print("\n" + "=" * 50)
        print("[SUCCESS] TCS.NS datetime error has been FIXED!")
        print("The backend is now working correctly.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("[FAILED] TCS.NS still has issues.")
        print("You may need to restart the backend server.")
        print("=" * 50)