with open('server.py', 'r') as f:
    content = f.read()

content = content.replace('if request.response_format == "wav" and chunk_index > 0:', 'if request.response_format == "wav" and chunk_index > 1:')

with open('server.py', 'w') as f:
    f.write(content)
