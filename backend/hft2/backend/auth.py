"""JWT authentication: login, token creation/verification, user storage in MongoDB."""
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Load env early
try:
    from dotenv import load_dotenv
    from pathlib import Path
    _env_path = Path(__file__).resolve().parents[1] / "env"
    load_dotenv(_env_path)
except Exception:
    pass

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production-use-long-secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))


def _get_users_collection():
    try:
        from db.mongo_client import get_mongo_db
        db = get_mongo_db("trading")
        return db["users"]
    except Exception as e:
        logger.warning(f"MongoDB users collection unavailable: {e}")
        return None


def hash_password(password: str) -> str:
    """Hash password with bcrypt. Bcrypt has a 72-byte limit, so truncate if needed."""
    import bcrypt
    # Bcrypt has a 72-byte limit - truncate password BEFORE hashing
    if isinstance(password, str):
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            # Truncate to 72 bytes
            password_bytes = password_bytes[:72]
    else:
        password_bytes = password
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    """Verify password with bcrypt. Truncate plain password if longer than 72 bytes."""
    import bcrypt
    try:
        # Bcrypt has a 72-byte limit - truncate password BEFORE verification
        if isinstance(plain, str):
            plain_bytes = plain.encode('utf-8')
            if len(plain_bytes) > 72:
                # Truncate to 72 bytes
                plain_bytes = plain_bytes[:72]
        else:
            plain_bytes = plain
        # Verify password - hash should be bytes or string
        if isinstance(hashed, str):
            hashed_bytes = hashed.encode('utf-8')
        else:
            hashed_bytes = hashed
        return bcrypt.checkpw(plain_bytes, hashed_bytes)
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False


def create_token(sub: str, extra: Optional[dict] = None) -> str:
    import jwt
    payload = {
        "sub": sub,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES),
        "iat": datetime.utcnow(),
    }
    if extra:
        payload.update(extra)
    out = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return out if isinstance(out, str) else out.decode("utf-8")


def decode_token(token: str) -> Optional[dict]:
    import jwt
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        return None


def get_user_by_username(username: str) -> Optional[dict]:
    """Get user by username (email). Username is normalized to lowercase."""
    col = _get_users_collection()
    if col is None:
        logger.warning("MongoDB collection unavailable - cannot get user")
        return None
    normalized_username = username.lower().strip()
    logger.debug(f"Looking up user: '{normalized_username}'")
    u = col.find_one({"username": normalized_username})
    if u:
        if "_id" in u:
            u["id"] = str(u["_id"])
        logger.debug(f"User found: {u.get('username')}")
        return u
    else:
        logger.warning(f"User not found: {normalized_username}")
        return None


def create_user(username: str, password: str) -> Optional[dict]:
    """Create a new user. Username is normalized to lowercase."""
    try:
        col = _get_users_collection()
        if col is None:
            logger.warning("MongoDB collection unavailable - cannot create user")
            return None
        normalized_username = username.lower().strip()
        logger.debug(f"Creating user: '{normalized_username}'")
        
        # Check if user already exists
        existing = col.find_one({"username": normalized_username})
        if existing:
            logger.warning(f"Username already exists: {normalized_username}")
            return None  # Username already exists
        
        # Hash password and create user document
        password_hash = hash_password(password)
        doc = {
            "username": normalized_username,
            "password_hash": password_hash,
            "created_at": datetime.utcnow(),
        }
        r = col.insert_one(doc)
        doc["id"] = str(r.inserted_id)
        logger.info(f"User created successfully in MongoDB: {normalized_username} (ID: {doc['id']})")
        
        # Verify the user was actually saved
        verify_user = col.find_one({"username": normalized_username})
        if not verify_user:
            logger.error(f"CRITICAL: User {normalized_username} was not saved to MongoDB after insert_one!")
            return None
        
        logger.info(f"User verified in database: {normalized_username}")
        return doc
    except Exception as e:
        logger.error(f"Failed to create user in MongoDB: {e}", exc_info=True)
        # Return None instead of raising - let web_backend handle the 503
        return None


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate user by username and password. Returns user dict if successful, None otherwise."""
    try:
        normalized_username = username.lower().strip()
        logger.info(f"Attempting to authenticate user: {normalized_username}")
        user = get_user_by_username(normalized_username)
        if not user:
            logger.warning(f"User not found: {normalized_username}")
            return None
        password_hash = user.get("password_hash", "")
        if not password_hash:
            logger.warning(f"User {normalized_username} has no password hash")
            return None
        if not verify_password(password, password_hash):
            logger.warning(f"Password verification failed for user: {normalized_username}")
            return None
        logger.info(f"User authenticated successfully: {normalized_username}")
        return user
    except Exception as e:
        logger.error(f"Failed to authenticate user from MongoDB: {e}", exc_info=True)
        return None
