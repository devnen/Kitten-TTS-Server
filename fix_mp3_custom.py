import re
with open('server.py', 'r') as f:
    content = f.read()

# Replace the format validation in custom_tts_endpoint
old_validation = """    if output_format_str not in ["wav", "opus"]:
        raise HTTPException(status_code=400, detail=f"Invalid format: {output_format_str}. Only 'wav' or 'opus' are supported.")"""

new_validation = """    if output_format_str == "mp3":
        raise HTTPException(status_code=400, detail="MP3 output is not supported in this deployment. Use wav.")
    if output_format_str not in ["wav", "opus"]:
        raise HTTPException(status_code=400, detail=f"Invalid format: {output_format_str}. Only 'wav' or 'opus' are supported.")"""

content = content.replace(old_validation, new_validation)

with open('server.py', 'w') as f:
    f.write(content)
