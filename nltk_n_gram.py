#4) create n-grams for different values of n=2,4.
def generate_ngrams(text, n):
    tokens = text.split()
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return ngrams
text = "The data set given satisfies the requirement for model generation.This is used in Data Science Lab"
bigrams = generate_ngrams(text, 2)
four_grams = generate_ngrams(text, 4)
print("Bigrams (n=2):", bigrams)
print("4-grams (n=4):", four_grams)