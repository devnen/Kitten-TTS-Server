with open('server.py', 'r') as f:
    content = f.read()

old_val1 = """    if output_format_str not in ["wav", "opus"]:
        raise HTTPException(status_code=400, detail=f"Invalid format: {output_format_str}. Only 'wav' or 'opus' are supported.")"""

new_val1 = """    if output_format_str == "mp3":
        raise HTTPException(status_code=400, detail="MP3 output is not supported in this deployment. Use wav.")
    if output_format_str not in ["wav", "opus"]:
        raise HTTPException(status_code=400, detail=f"Invalid format: {output_format_str}. Only 'wav' or 'opus' are supported.")"""

content = content.replace(old_val1, new_val1)

# In openai_speech_endpoint, the default format is mp3 (OpenAI default). If they request mp3, reject.
# Wait, OpenAI spec defaults to mp3. Let's see how it's handled.
