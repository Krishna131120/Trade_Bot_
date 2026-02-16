"""MongoDB connection using MONGODB_URI from env."""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)
_client = None


def _load_mongo_env():
    """Load env from backend/hft2/env so MONGODB_URI is set regardless of cwd."""
    uri = os.getenv("MONGODB_URI")
    if uri:
        return uri
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        base = Path(__file__).resolve()
        # Try: backend/hft2/backend/db -> backend/hft2/env
        for parent in [base.parents[2], base.parents[1].parent, Path.cwd(), Path.cwd().parent]:
            env_file = parent / "env"
            if env_file.exists():
                load_dotenv(env_file)
                uri = os.getenv("MONGODB_URI")
                if uri:
                    logger.debug(f"MONGODB_URI loaded from {env_file}")
                    return uri
        load_dotenv()  # .env in cwd
        return os.getenv("MONGODB_URI")
    except Exception as e:
        logger.debug(f"dotenv load: {e}")
    return os.getenv("MONGODB_URI")


def get_mongo_client():
    """Return a PyMongo MongoClient. Requires pymongo and MONGODB_URI in env."""
    global _client
    if _client is not None:
        return _client
    uri = _load_mongo_env()
    if not uri:
        raise ValueError("MONGODB_URI not set. Add it to backend/hft2/env or set the environment variable.")
    try:
        from pymongo import MongoClient
        # 10s timeout so we don't hang; Atlas can be slow on first connect
        _client = MongoClient(uri, serverSelectionTimeoutMS=10000)
        _client.admin.command("ping")
        logger.info("MongoDB connected")
        return _client
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise


def get_mongo_db(name: Optional[str] = None):
    """Return database. If name is None, uses default from URI or 'trading'."""
    client = get_mongo_client()
    if name:
        return client[name]
    # default db from URI or 'trading'
    return client.get_database()
