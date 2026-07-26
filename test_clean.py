from utils.clean import clean_text

sample = """
<h1>Breaking News!</h1>

Visit https://bbc.com

Artificial Intelligence is changing the world rapidly!!!

"""

print(clean_text(sample))