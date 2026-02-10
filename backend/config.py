"""
Configuration Management
Loads settings from environment variables
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Authentication Configuration
# JWT authentication permanently disabled - open access API
ENABLE_AUTH = False

# JWT Configuration - kept for backward compatibility (auth is disabled but auth.py imports these)
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'default-key-for-optional-auth')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))

# Admin Credentials - kept for backward compatibility (auth is disabled but auth.py imports these)
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')

# Rate Limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '10'))
RATE_LIMIT_PER_HOUR = int(os.getenv('RATE_LIMIT_PER_HOUR', '100'))

# API Limits
MAX_SYMBOLS_PER_REQUEST = int(os.getenv('MAX_SYMBOLS_PER_REQUEST', '10'))
MAX_SCAN_SYMBOLS = int(os.getenv('MAX_SCAN_SYMBOLS', '50'))

# FastAPI Configuration
API_TITLE = "Stock Prediction MCP API"
API_VERSION = "3.0"
API_DESCRIPTION = "Secure MCP-style REST API with JWT auth, rate limiting, and validation"
UVICORN_HOST = os.getenv('UVICORN_HOST', '0.0.0.0')
UVICORN_PORT = int(os.getenv('PORT', os.getenv('UVICORN_PORT', '8000')))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# CORS: comma-separated list of allowed frontend origins (e.g. https://trade-bot-dashboard-llb8.onrender.com)
# Set CORS_ALLOW_ALL=true to allow all origins (useful for debugging)
CORS_ALLOW_ALL = os.getenv('CORS_ALLOW_ALL', 'false').lower() == 'true'
CORS_ORIGINS_EXTRA = [x.strip() for x in os.getenv('CORS_ORIGINS', '').split(',') if x.strip()]

# Auto-detect Render domains
if os.getenv('RENDER'):
    # When running on Render, automatically allow common Render domains
    CORS_ORIGINS_EXTRA.extend([
        "https://trade-bot-dashboard-c9x3.onrender.com",
        "https://trade-bot-dashboard-llb8.onrender.com",
        "https://trade-bot-frontend-halb.onrender.com",
        "https://trade-bot-api.onrender.com",
    ])

# Directories
DATA_DIR = Path("data")
DATA_CACHE_DIR = DATA_DIR / "cache"
FEATURE_CACHE_DIR = DATA_DIR / "features"
LOGS_DIR = DATA_DIR / "logs"
MODEL_DIR = Path("models")

# Ensure directories exist with proper error handling
# Note: exist_ok=True prevents race conditions when multiple processes
# try to create directories simultaneously. Each mkdir is atomic at the OS level.
directories_to_create = [
    ('Data Cache', DATA_CACHE_DIR),
    ('Feature Cache', FEATURE_CACHE_DIR),
    ('Logs', LOGS_DIR),
    ('Models', MODEL_DIR)
]

for dir_name, directory in directories_to_create:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f'[ERROR] No permission to create {dir_name} directory: {directory}', file=sys.stderr)
        print(f'[ERROR] Please ensure write permissions for: {directory.parent}', file=sys.stderr)
        print(f'[ERROR] Current user: {os.getenv("USER", os.getenv("USERNAME", "unknown"))}', file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f'[ERROR] Cannot create {dir_name} directory: {directory}', file=sys.stderr)
        print(f'[ERROR] OS Error: {e}', file=sys.stderr)
        print(f'[ERROR] Check disk space and file system permissions', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'[ERROR] Unexpected error creating {dir_name} directory: {directory}', file=sys.stderr)
        print(f'[ERROR] {e}', file=sys.stderr)
        sys.exit(1)
