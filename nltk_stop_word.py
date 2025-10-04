#2) Remove the stop words from the given text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
sample_text = " The data set given satisfies the requirement for model generation.This is used in Data Science Lab"
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(sample_text.lower())
filtered_list = [word for word in tokens if word not in stop_words]
print(filtered_list)