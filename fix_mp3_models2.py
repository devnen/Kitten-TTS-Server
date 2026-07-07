with open('models.py', 'r') as f:
    content = f.read()

content = content.replace('output_format: Optional[Literal["wav", "opus"]]', 'output_format: Optional[Literal["wav", "opus", "mp3"]]')

with open('models.py', 'w') as f:
    f.write(content)

with open('server.py', 'r') as f:
    content = f.read()

content = content.replace('response_format: Literal["wav", "opus"] = "wav"', 'response_format: Literal["wav", "opus", "mp3"] = "wav"')

with open('server.py', 'w') as f:
    f.write(content)
