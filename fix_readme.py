with open("README.md", "r") as f:
    content = f.read()

new_docs = """
### API Authentication and Security Defaults
For safe deployment, the following security defaults are enabled:
- `ENABLE_WEB_UI` is `false` by default.
- `ENABLE_MANAGEMENT_ENDPOINTS` is `false` by default.
- API endpoints (`/tts`, `/api/tts/speak`, `/v1/audio/speech`, `/api/unload`, etc) require a configured `TTS_API_KEY`. If it is unset, these endpoints fail securely.

Provide your configured key via `Authorization: Bearer <your_api_key>` or `X-API-Key: <your_api_key>`.
Only `/health` and `/api/tts/health` are public without auth.

### CORS Configuration
CORS origins should be specified via the `ALLOWED_ORIGINS` environment variable (comma-separated, e.g. `http://localhost:3000,http://localhost:5173`). Wildcard `*` origins are no longer permitted with credentials in production.

### MP3 Support
To ensure compatibility in lightweight server environments (like Railway), MP3 generation is **intentionally unsupported** in this deployment. Please request `wav` (default) or `opus`.
"""
if "### API Authentication and Security Defaults" not in content:
    content = content.replace("### API Authentication", new_docs)
    with open("README.md", "w") as f:
        f.write(content)
