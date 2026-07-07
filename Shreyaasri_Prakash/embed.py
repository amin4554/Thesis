"""
ollama pull nomic-embed-text
"""
#Example: Compute similarity between embeddings
import ollama
import numpy as np

def get_embedding(text):
    return ollama.embeddings(model="nomic-embed-text", prompt=text).embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Get embeddings' for various texts

# # Sentences about the sky and ocean — both talk about “blue things”
sky_sentence = get_embedding("The sky is blue.")
ocean_sentence = get_embedding("The ocean is blue.")

# # A completely different sentence (about food)
pizza_sentence = get_embedding("I love eating pizza.")

# Two sentences with similar meaning (both describe a cat sleeping)
cat_sentence = get_embedding("The cat is sleeping on the mat.")
kitty_sentence = get_embedding("A kitty is taking a nap on the rug.")

# Two very similar phrases with only a plural difference
blue_sky = get_embedding("Blue Sky")
blue_skies = get_embedding("Blue Skies")

# Two sentences that mean almost the same thing (identifying as Ollama)
name_sentence = get_embedding("Fuel is flammable.")
identity_sentence = get_embedding("Fuel is flammable.")


# Compare similarities and print results

print("Sky vs Ocean:", cosine_similarity(sky_sentence, ocean_sentence))
print("Sky vs Pizza:", cosine_similarity(sky_sentence, pizza_sentence))
print("Blue Sky vs Blue Skies:", cosine_similarity(blue_sky, blue_skies))
print("Cat vs Kitty:", cosine_similarity(cat_sentence, kitty_sentence))
print("Fuel vs Fuel", cosine_similarity(name_sentence, identity_sentence))
