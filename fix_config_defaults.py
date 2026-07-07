import re

with open('config.py', 'r') as f:
    content = f.read()

# I will update defaults in config.py if they exist there. But they are env vars, so mostly server.py
pass

with open('config.yaml', 'r') as f:
    content = f.read()

# Clean config.yaml
content = re.sub(r'last_text:.*?\n(?=  [a-zA-Z_]+:)', 'last_text: ""\n', content, flags=re.DOTALL)
content = content.replace(r'logs\tts_server.log', 'logs/tts_server.log')

with open('config.yaml', 'w') as f:
    f.write(content)
