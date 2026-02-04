import subprocess
import re
import sys

def test_single_option(option_num, symbol='TCS.NS'):
    """Test a single menu option for news content"""
    try:
        proc = subprocess.Popen([sys.executable, 'stock_analysis_complete.py'], 
                               stdin=subprocess.PIPE, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        
        # Send option and symbol
        stdin_input = f'{option_num}\n{symbol}\n9\n'  # 9 to exit after the test
        stdout, stderr = proc.communicate(input=stdin_input, timeout=20)
        
        # Look for news-related content in output
        has_news = bool(re.search(r'(news|headline|article|sentiment|event|announcement|press|coverage)', stdout.lower()))
        
        print(f'\n--- OPTION {option_num} ---')
        print(f'Status: {"✅ HAS NEWS" if has_news else "❌ NO NEWS FOUND"}')
        
        if has_news:
            print("News-related content found in output:")
            # Print lines containing news-related terms
            for i, line in enumerate(stdout.split('\n')):
                if re.search(r'(news|headline|article|sentiment|event|announcement|press|coverage)', line.lower()):
                    print(f'  Line {i+1}: {line.strip()}')
        
        return has_news, stdout
    except Exception as e:
        print(f'OPTION {option_num}: FAILED - {str(e)}')
        return False, ""

# Test all options that accept symbols
options_to_test = [1, 2, 3, 4, 5, 7, 8]
results = {}

print('Testing each menu option for news content...')
print('='*60)

for opt in options_to_test:
    has_news, output = test_single_option(opt)
    results[opt] = has_news

print('\n' + '='*60)
print('FINAL SUMMARY:')
print('='*60)
for opt in sorted(results.keys()):
    status = "✅ HAS NEWS" if results[opt] else "❌ NO NEWS"
    print(f'Option {opt}: {status}')

# Identify which option(s) have news
news_options = [opt for opt, has_news in results.items() if has_news]
if news_options:
    print(f'\nOptions with news content: {", ".join(map(str, news_options))}')
    print('These options likely display news along with stock data.')
else:
    print('\nNo options found that display news content.')
    print('News may be stored in cache but not displayed in menu output.')