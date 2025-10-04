#3) Perform Part of Speech tagging
from nltk import pos_tag, word_tokenize
text = "The data set given satisfies the requirement for model generation.This is used in Data Science Lab"
tokens = word_tokenize(text)
tags = pos_tag(tokens)
print(tags)
