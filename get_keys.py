import re

with open('references.bib', 'r', encoding='utf-8') as f:
    content = f.read()

entries = re.findall(r'@\w+\{(.+?),', content)
print(','.join(entries))
