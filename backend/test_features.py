#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import json

# Test script to debug the timezone issue

def test_feature_calculation():
    print("Testing feature calculation...")
    
    # Load the AAPL data
    with open('data/cache/AAPL_all_data.json', 'r') as f:
        cached_data = json.load(f)
    
    print(f"Loaded data with {len(cached_data['price_history'])} records")
    
    # Create DataFrame
    df = pd.DataFrame(cached_data['price_history'])
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check Date column
    if 'Date' in df.columns:
        print(f"First few dates: {df['Date'].head().tolist()}")
        print(f"Date column type: {type(df['Date'].iloc[0])}")
        
        # Try different approaches to handle dates
        try:
            # Approach 1: Simple parsing
            df['Date'] = pd.to_datetime(df['Date'])
            print("Approach 1 worked - simple parsing")
        except Exception as e:
            print(f"Approach 1 failed: {e}")
            
            try:
                # Approach 2: UTC parsing
                df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
                print("Approach 2 worked - UTC parsing")
            except Exception as e2:
                print(f"Approach 2 failed: {e2}")
                
                # Approach 3: Manual cleaning
                date_strings = []
                for date_str in df['Date']:
                    if isinstance(date_str, str):
                        # Remove timezone info
                        clean_date = date_str.split('+')[0].split('-05:00')[0].split('-04:00')[0]
                        date_strings.append(clean_date)
                    else:
                        date_strings.append(str(date_str))
                
                df['Date'] = pd.to_datetime(date_strings)
                print("Approach 3 worked - manual cleaning")
        
        df.set_index('Date', inplace=True)
    
    print(f"Final DataFrame shape: {df.shape}")
    print(f"Index type: {type(df.index)}")
    print(f"Index timezone: {getattr(df.index, 'tz', None)}")
    
    # Test basic technical indicator
    try:
        rsi = ta.rsi(df['Close'], length=14)
        print(f"RSI calculation successful: {len(rsi)} values")
        return True
    except Exception as e:
        print(f"RSI calculation failed: {e}")
        return False

if __name__ == "__main__":
    test_feature_calculation()