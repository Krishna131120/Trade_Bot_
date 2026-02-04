import pexpect
import sys

def test_option_8():
    print("Testing Option 8: Predict with Ensemble")
    print("="*60)
    
    try:
        # Spawn the process
        child = pexpect.spawn(sys.executable + ' stock_analysis_complete.py')
        child.logfile = sys.stdout.buffer  # Show output in real-time
        
        # Wait for main menu
        child.expect('Enter your choice \(1-9\):')
        print("Sending option 8...")
        child.sendline('8')
        
        # Wait for symbol prompt
        child.expect('Symbol \(e.g., INFY.NS, RPOWER.NS, AAPL\):')
        print("Sending symbol TCS.NS...")
        child.sendline('TCS.NS')
        
        # Wait for a bit to see the error
        child.expect(pexpect.EOF, timeout=10)
        
    except pexpect.exceptions.TIMEOUT:
        print("Process timed out")
        child.terminate()
    except Exception as e:
        print(f"Error occurred: {e}")
        try:
            child.terminate()
        except:
            pass

if __name__ == "__main__":
    test_option_8()