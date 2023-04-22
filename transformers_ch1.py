#%%
from transformers import pipeline

text = """I wish I could get my girlfriend to come to Baltimore more often. She's a great cook and I love her company."""

pipe = pipeline('text-generation')
pipe(text)
#%%
pipe = pipeline('ner')
pipe(text)
#%%
pipe = pipeline('question-answering')
text = """given that my girlfriend is a great cook, what types of chores should I do?"""
pipe(text)
# %%
pipe = pipeline("fill-mask")
#%%
pipe = pipeline("summarization")

#%%
pipe = pipeline("translation_en_to_spa")
#%%