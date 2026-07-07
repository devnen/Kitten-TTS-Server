import re

with open('server.py', 'r') as f:
    content = f.read()

content = content.replace('os.environ.get("ENABLE_WEB_UI", "true")', 'os.environ.get("ENABLE_WEB_UI", "false")')
content = content.replace('os.environ.get("ENABLE_MANAGEMENT_ENDPOINTS", "true")', 'os.environ.get("ENABLE_MANAGEMENT_ENDPOINTS", "false")')

with open('server.py', 'w') as f:
    f.write(content)
