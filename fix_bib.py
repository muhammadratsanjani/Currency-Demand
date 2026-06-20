import re

with open('references.bib', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix empty keys like @article{,
count = 1
def replace_empty_key(match):
    global count
    new_key = f'@article{{MissingKey{count},'
    count += 1
    return new_key

fixed_content = re.sub(r'@article\{,\s*', replace_empty_key, content)

with open('references.bib', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

entries = re.findall(r'@\w+\{(.+?),', fixed_content)
print(f'Total References Fixed and Found: {len(entries)}')
