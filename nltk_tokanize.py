#1) perform word and sentence tokenization.
from nltk import word_tokenize, sent_tokenize
sent = " The data set given satisfies the requirement for model generation.This is used in Data Science Lab"
print(word_tokenize(sent))
print(sent_tokenize(sent))