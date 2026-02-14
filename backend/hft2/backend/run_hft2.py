#!/usr/bin/env python3
"""
Single entry point to run the full HFT2 stack:
- Fyers data service (port 8002) - with FYERS_ALLOW_MOCK=true if Fyers API unavailable
- MCP server (API on 8003, monitoring 8004)
- Web backend (port 5000) - requires pandas_market_calendars, testindia
- Simple HFT2 API (port 5001) - runs in foreground; Ctrl+C stops all

Usage: from backend/hft2/backend run:  python run_hft2.py
"""
import os
import sys
import time
import signal
import subprocess
import atexit
import socket
from pathlib import Path

# Run from backend/hft2/backend
BACKEND_DIR = Path(__file__).resolve().parent
os.chdir(BACKEND_DIR)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load env from backend/hft2/env
ENV_FILE = BACKEND_DIR.parent / "env"
if ENV_FILE.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
    except ImportError:
        pass

# Child process handles; killed on exit
children = []


def is_port_in_use(port):
    """Check if a port is already in use - Windows compatible"""
    import subprocess
    # Method 1: Try binding to the port (most reliable)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                s.close()
                return False  # Port is free
            except OSError:
                return True  # Port is in use
    except Exception:
        pass
    
    # Method 2: Fallback - check via netstat on Windows
    try:
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            timeout=2
        )
        # Check if port appears in LISTENING state
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                return True
    except Exception:
        pass
    
    return False


def kill_children():
    for p in children:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    children.clear()


def start_subprocess(name, cmd, env_extra=None, wait_ready_sec=0):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        # Use PIPE for both stdout and stderr so we can capture errors, but don't block on them
        p = subprocess.Popen(
            cmd,
            cwd=BACKEND_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            creationflags=creationflags,
        )
        children.append(p)
        if wait_ready_sec:
            time.sleep(wait_ready_sec)
        # Check if process exited
        exit_code = p.poll()
        if exit_code is not None:
            # Process exited - read output
            msg = ""
            try:
                stdout, _ = p.communicate(timeout=2)
                msg = (stdout or b"").decode(errors="replace")[:800]
                print(f"[run_hft2] {name} exited with code {exit_code}:\n{msg}")
            except subprocess.TimeoutExpired:
                p.kill()
                print(f"[run_hft2] {name} exited immediately (timeout reading output)")
                msg = "timeout"
            
            if msg and ("sqlalchemy" in msg.lower() or "No module named" in msg or "NameError" in msg):
                print("[run_hft2] Missing dependencies! Install: pip install -r requirements-minimal.txt")
            elif msg and "DeprecationWarning" in msg and exit_code == 0:
                # Just warnings, process might still be running - check again
                if p.poll() is None:
                    print(f"[run_hft2] Started {name} (PID {p.pid}) - warnings ignored")
                    return p
            children.remove(p)
            return None
        print(f"[run_hft2] Started {name} (PID {p.pid})")
        return p
    except Exception as e:
        print(f"[run_hft2] Failed to start {name}: {e}")
        return None


def main():
    atexit.register(kill_children)

    def sig_handler(signum, frame):
        kill_children()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, sig_handler)

    # 1) Fyers data service on 8002 (mock mode so it starts without fyers_apiv3)
    start_subprocess(
        "fyers_data_service (8002)",
        [sys.executable, "fyers_data_service.py", "--port", "8002"],
        env_extra={"FYERS_ALLOW_MOCK": "true"},
        wait_ready_sec=2,
    )

    # 2) MCP server (use 8004 for monitoring so 8002 stays for Fyers)
    start_subprocess(
        "start_mcp_server",
        [sys.executable, "start_mcp_server.py"],
        env_extra={"MCP_MONITORING_PORT": "8004", "MCP_API_PORT": "8003"},
        wait_ready_sec=3,
    )

    # 3) Web backend on 5000 (may fail if pandas_market_calendars / testindia missing)
    start_subprocess(
        "web_backend (5000)",
        [sys.executable, "web_backend.py", "--port", "5000"],
        wait_ready_sec=2,
    )

    # 4) Optional: Main backend proxy on 8000 (if MAIN_BACKEND_PORT env is set)
    main_backend_port = os.environ.get("MAIN_BACKEND_PORT", "")
    if main_backend_port:
        try:
            main_backend_port_int = int(main_backend_port)
            if not is_port_in_use(main_backend_port_int):
                # Start main backend with proxy to hft2
                main_backend_dir = BACKEND_DIR.parent.parent  # backend/ directory
                api_server_path = main_backend_dir / "api_server.py"
                if api_server_path.exists():
                    start_subprocess(
                        f"main_backend ({main_backend_port_int})",
                        [sys.executable, str(api_server_path)],
                        env_extra={"HFT2_BACKEND_URL": f"http://127.0.0.1:5001"},
                        wait_ready_sec=3,
                    )
                    print(f"[run_hft2] Main backend proxy started on port {main_backend_port_int}")
                else:
                    print(f"[run_hft2] Main backend not found at {api_server_path}")
            else:
                print(f"[run_hft2] Port {main_backend_port_int} already in use - skipping main backend")
        except Exception as e:
            print(f"[run_hft2] Failed to start main backend: {e}")

    # 5) Simple HFT2 API on 5001 in foreground (this blocks until Ctrl+C)
    port = 5001
    if is_port_in_use(port):
        print(f"[run_hft2] ERROR: Port {port} is already in use!")
        print(f"[run_hft2] Kill the existing process or use a different port.")
        print(f"[run_hft2] On Windows: netstat -ano | findstr :{port}  then  taskkill /PID <PID> /F")
        kill_children()
        sys.exit(1)
    
    print(f"[run_hft2] Starting HFT2 API on port {port} (Ctrl+C stops all)...")
    print(f"[run_hft2] Frontend should call: http://127.0.0.1:{port}/api/bot-data (direct)")
    if main_backend_port:
        print(f"[run_hft2] OR via proxy: http://127.0.0.1:{main_backend_port}/api/bot-data")
    import uvicorn
    from simple_app import app
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    finally:
        kill_children()


if __name__ == "__main__":
    main()
