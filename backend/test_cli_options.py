import subprocess
import sys
import time

def test_menu_option(option, input_data=None):
    """Test a specific menu option and return the result"""
    print(f"\n{'='*60}")
    print(f"TESTING MENU OPTION {option}")
    print(f"{'='*60}")
    
    try:
        # Start the process
        process = subprocess.Popen(
            [sys.executable, 'stock_analysis_complete.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        # Send menu choice
        inputs = [str(option)]
        if input_data:
            inputs.extend(input_data)
        inputs.append('9')  # Always exit after test
        
        full_input = '\n'.join(inputs) + '\n'
        
        # Get output
        stdout, stderr = process.communicate(input=full_input, timeout=30)
        
        print("STDOUT:")
        print(stdout)
        
        if stderr:
            print("\nSTDERR:")
            print(stderr)
            
        return {
            'success': process.returncode == 0,
            'stdout': stdout,
            'stderr': stderr,
            'returncode': process.returncode
        }
        
    except subprocess.TimeoutExpired:
        process.kill()
        print("Process timed out")
        return {'success': False, 'error': 'timeout'}
    except Exception as e:
        print(f"Error: {e}")
        return {'success': False, 'error': str(e)}

# Test each menu option
results = {}

# Test Option 1: Fetch Stock Data
print("Testing Option 1: Fetch Stock Data")
results['1'] = test_menu_option(1, ['TCS.NS'])

# Test Option 2: Calculate Technical Indicators  
print("\nTesting Option 2: Calculate Technical Indicators")
results['2'] = test_menu_option(2, ['TCS.NS'])

# Test Option 3: View Financial Statements
print("\nTesting Option 3: View Financial Statements")
results['3'] = test_menu_option(3, ['TCS.NS'])

# Test Option 4: View Technical Indicators
print("\nTesting Option 4: View Technical Indicators")
results['4'] = test_menu_option(4, ['TCS.NS'])

# Test Option 5: Complete Analysis
print("\nTesting Option 5: Complete Analysis")
results['5'] = test_menu_option(5, ['TCS.NS'])

# Test Option 7: Train ALL 4 Models
print("\nTesting Option 7: Train ALL 4 Models")
results['7'] = test_menu_option(7, ['TCS.NS'])

# Test Option 8: Predict with Ensemble
print("\nTesting Option 8: Predict with Ensemble")
results['8'] = test_menu_option(8, ['TCS.NS'])

# Test Option 9: Exit
print("\nTesting Option 9: Exit")
results['9'] = test_menu_option(9)

# Summary
print(f"\n{'='*60}")
print("TEST SUMMARY")
print(f"{'='*60}")

working = []
not_working = []

for option, result in results.items():
    if result.get('success', False):
        working.append(option)
        print(f"✅ Option {option}: WORKING")
    else:
        not_working.append(option)
        print(f"❌ Option {option}: NOT WORKING")
        if 'error' in result:
            print(f"   Error: {result['error']}")
        if 'stderr' in result and result['stderr']:
            print(f"   stderr: {result['stderr'][:200]}...")

print(f"\nWorking options: {', '.join(working)}")
print(f"Not working options: {', '.join(not_working)}")

# Save detailed results
with open('cli_test_results.txt', 'w') as f:
    f.write("CLI TOOL TEST RESULTS\n")
    f.write("="*50 + "\n\n")
    for option, result in results.items():
        f.write(f"Option {option}:\n")
        f.write(f"  Success: {result.get('success', False)}\n")
        if 'error' in result:
            f.write(f"  Error: {result['error']}\n")
        if result.get('stdout'):
            f.write(f"  Output: {result['stdout'][:500]}...\n")
        f.write("\n")