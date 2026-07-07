with open('server.py', 'r') as f:
    content = f.read()

# Make sure we import Security and APIKeyHeader
if "from fastapi import Depends" in content and "Security" not in content:
    content = content.replace("from fastapi import Depends", "from fastapi import Depends, Security, Header")
if "from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials" in content and "APIKeyHeader" not in content:
    content = content.replace("from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials", "from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader")

old_verify = """def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = os.environ.get("TTS_API_KEY")
    if not api_key:
        return None
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing API Key")
    if not hmac.compare_digest(credentials.credentials, api_key):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials"""

new_verify = """api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_api_key: str = Security(api_key_header)
):
    api_key = os.environ.get("TTS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="TTS_API_KEY is not configured on the server. Access denied.")

    token = x_api_key
    if not token and credentials and credentials.credentials:
        token = credentials.credentials

    if not token:
        raise HTTPException(status_code=401, detail="Missing API Key")

    if not hmac.compare_digest(token, api_key):
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return token"""

content = content.replace(old_verify, new_verify)

with open('server.py', 'w') as f:
    f.write(content)
