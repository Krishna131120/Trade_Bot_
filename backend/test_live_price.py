"""
Test Live Price Validation
"""

import sys
sys.path.append('.')

from live_price_validator import LivePriceValidator
import json

def test_live_price():
    """Test live price fetching for TCS.NS"""
    print("Testing Live Price Validation for TCS.NS")
    print("="*50)
    
    validator = LivePriceValidator()
    
    # Test TCS.NS
    live_data = validator.get_live_price_data("TCS.NS")
    
    print(f"Symbol: {live_data.get('symbol')}")
    print(f"Current Price: Rs.{live_data.get('current_price', 0):.2f}")
    print(f"Price Source: {live_data.get('price_source')}")
    print(f"Exchange: {live_data.get('exchange')}")
    print(f"Currency: {live_data.get('currency')}")
    print(f"Market State: {live_data.get('market_state')}")
    print(f"Is Delayed: {live_data.get('is_delayed')}")
    print(f"Delay Minutes: {live_data.get('delay_minutes', 0)}")
    print(f"Fetch Time: {live_data.get('fetch_timestamp')}")
    
    validation = live_data.get('validation', {})
    print(f"\nValidation:")
    print(f"  Valid: {validation.get('is_valid')}")
    print(f"  Warnings: {validation.get('warnings', [])}")
    print(f"  Errors: {validation.get('errors', [])}")
    
    # Compare with cached data
    print(f"\n" + "="*50)
    print("COMPARING WITH CACHED DATA")
    print("="*50)
    
    try:
        with open('data/cache/TCS.NS_all_data.json', 'r') as f:
            cached_data = json.load(f)
        
        cached_price = cached_data.get('info', {}).get('currentPrice', 0)
        live_price = live_data.get('current_price', 0)
        
        if cached_price > 0 and live_price > 0:
            diff_pct = ((live_price - cached_price) / cached_price) * 100
            print(f"Cached Price: Rs.{cached_price:.2f}")
            print(f"Live Price: Rs.{live_price:.2f}")
            print(f"Difference: {diff_pct:+.2f}%")
            
            if abs(diff_pct) > 1:
                print(f"🔴 SIGNIFICANT DIFFERENCE DETECTED!")
                print(f"   This confirms the data integrity issue.")
            else:
                print(f"✅ Prices are aligned (difference < 1%)")
        else:
            print(f"Could not compare prices (cached={cached_price}, live={live_price})")
            
    except Exception as e:
        print(f"Error loading cached data: {e}")

if __name__ == "__main__":
    test_live_price()