import jwt
import datetime
import os

# JWT Secret Key (use environment variable in production)
JWT_SECRET = os.getenv("JWT_SECRET", "f541169f57693ac288720476cdcba19c669dc7caadbeb077572093a94d6d3514")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY = datetime.timedelta(hours=1)

USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

def authenticate(username, password):
    user = USERS.get(username)
    if user and user["password"] == password:
        # Generate JWT token
        payload = {
            "user": username,
            "role": user["role"],
            "exp": datetime.datetime.utcnow() + TOKEN_EXPIRY,
            "iat": datetime.datetime.utcnow()
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return True, user["role"], token
    return False, None, None

def verify_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["user"], payload["role"]
    except jwt.ExpiredSignatureError:
        return None, None
    except jwt.InvalidTokenError:
        return None, None

def check_permission(token, required_role=None):
    user, role = verify_token(token)
    if not user:
        return False
    if required_role and role != required_role:
        return False
    return True