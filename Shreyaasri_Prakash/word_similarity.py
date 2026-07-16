"""
pip install spacy

"""
import spacy
import numpy as np

nlp = spacy.load('en_core_web_lg')

doc = nlp("sitting in the morning sun")

w1 = nlp("coffee")
w2 = nlp("espresso")
w3 = nlp("tea")
w4 = nlp("water")
w5 = nlp("giraffe")
words = [w1, w2, w3, w4, w5]

def l2_norm(a):
    return np.sqrt((a**2).sum())

def cosim(a, b):
    return np.dot(a.vector, b.vector) / (l2_norm(a.vector) * l2_norm(b.vector))

# all words against all
for word1 in words:
    for word2 in words:
        print(f"{str(word1):10} {str(word2):10} {cosim(word1, word2):5.2f}")


# document vectors
doc = nlp("sitting in the morning sun")
docvec = sum(word.vector for word in doc) / len(doc)
sum(doc.vector - docvec)

"""
task: search for similar lyrics

1. define a query sentence q
   'morning sunshine bright day'
   'we are living in a material world and I am a material girl'
   'the night is dark and full of stars'

2. loop over all song lyrics ly from lyrics.json
"""