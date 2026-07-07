import re

with open('server.py', 'r') as f:
    content = f.read()

# Fix pydantic model
content = content.replace('response_format: Literal["wav", "opus", "mp3"] = "wav"  # Add "mp3"',
                          'response_format: Literal["wav", "opus"] = "wav"')
content = content.replace('response_format: Literal["wav", "opus", "mp3"] = "wav"',
                          'response_format: Literal["wav", "opus"] = "wav"')

# Add explicit MP3 rejection logic inside openai_speech_endpoint
openai_endpoint_start = """    request_start = time.perf_counter()"""
openai_rejection = """    if request.response_format == "mp3":
        raise HTTPException(status_code=400, detail="MP3 output is not supported in this deployment. Use wav.")
    request_start = time.perf_counter()"""

content = content.replace(openai_endpoint_start, openai_rejection)

with open('server.py', 'w') as f:
    f.write(content)
