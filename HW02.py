HW03.py#2
import nltk
from nltk.corpus import gutenberg
persuasion_words = gutenberg.words('austen-persuasion.txt')
num_tokens = len(persuasion_words)
num_types = len(set(persuasion_words))
print("Number of word tokens:", num_tokens)
print("Number of word types:", num_types)


#3
import nltk
brown_news_words = nltk.corpus.brown.words(categories='news')
print("Sample text from the Brown Corpus (news genre):")
print(' '.join(brown_news_words[:50]))
webtextchatwords = nltk.corpus.webtext.words('firefox.txt')
print("\nSample text from the Web Text Corpus (chat genre):")
print(' '.join(webtextchatwords[:50]))

#4
import nltk
state_union = nltk.corpus.state_union
target_words = ["men", "women", "people"]
word_counts = {word: {} for word in target_words}
for fileid in state_union.fileids():
    year = fileid[:4]
    words = state_union.words(fileid)
    for word in target_words:
        count = words.count(word)
        word_counts[word][year] = count
for word, counts in word_counts.items():
    print(f"Occurrences of '{word}' over time:")
    for year, count in sorted(counts.items()):
        print(f"{year}: {count}")
    print()


#5
import nltk
from nltk.corpus import wordnet
def print_relations(noun):
    synsets = wordnet.synsets(noun)
    if not synsets:
        print(f"No synsets found for '{noun}'.")
        return

    for synset in synsets:
        print(f"Synset: {synset.name()} - Definition: {synset.definition()}")

        member_meronyms = synset.member_meronyms()
        if member_meronyms:
            print("Member Meronyms:", ', '.join(m.name().split('.')[0] for m in member_meronyms))


        part_meronyms = synset.part_meronyms()
        if part_meronyms:
            print("Part Meronyms:", ', '.join(m.name().split('.')[0] for m in part_meronyms))


        substance_meronyms = synset.substance_meronyms()
        if substance_meronyms:
            print("Substance Meronyms:", ', '.join(m.name().split('.')[0] for m in substance_meronyms))


        part_holonyms = synset.part_holonyms()
        if part_holonyms:
            print("Part Holonyms:", ', '.join(m.name().split('.')[0] for m in part_holonyms))


        substance_holonyms = synset.substance_holonyms()
        if substance_holonyms:
            print("Substance Holonyms:", ', '.join(m.name().split('.')[0] for m in substance_holonyms))

        print()



nouns = ['car', 'tree', 'water', 'hand']

for noun in nouns:
    print_relations(noun)

#11
import nltk

brown_corpus = nltk.corpus.brown
genres = brown_corpus.categories()
modal_verbs = ['can', 'could', 'may', 'might', 'will', 'would', 'shall', 'should', 'must']
modal_distributions = {}
for genre in genres:
    genre_words = brown_corpus.words(categories=genre)
    genre_modal_counts = nltk.FreqDist(w.lower() for w in genre_words if w.lower() in modal_verbs)
    modal_distributions[genre] = genre_modal_counts

print("Modal Distributions across Genres:")
for genre, modal_counts in modal_distributions.items():
    print(f"{genre}: {modal_counts}")

closed_classes = ['DT', 'IN', 'CC']

closed_class_distributions = {}


for closed_class in closed_classes:
    closed_class_counts = nltk.ConditionalFreqDist(
        (genre, word)
        for genre in genres
        for (word, tag) in nltk.pos_tag(brown_corpus.words(categories=genre))
        if tag == closed_class
    )
    closed_class_distributions[closed_class] = closed_class_counts


print("\nDistributions of Closed Classes across Genres:")
for closed_class, class_counts in closed_class_distributions.items():
    print(f"{closed_class}:")
    class_counts.tabulate(title=closed_class)


#17
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def most_frequent_words(text):
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    freq_dist = nltk.FreqDist(filtered_words)

    most_frequent = freq_dist.most_common(50)

    return most_frequent


text = "This is an example sentence. It demonstrates how to find the most frequent words in a text that are not stopwords. Stopwords are common words like 'the', 'and', 'is', 'are', etc."

result = most_frequent_words(text)

print("50 Most Frequent Words (excluding stopwords):")
for word, frequency in result:
    print(word, "-", frequency)

#23
import nltk
import matplotlib.pyplot as plt


def plot_zipf(text):
    tokens = nltk.word_tokenize(text.lower())

    freq_dist = nltk.FreqDist(tokens)

    sorted_freq = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)

    ranks = list(range(1, len(sorted_freq) + 1))
    frequencies = [freq for _, freq in sorted_freq]
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, frequencies, marker='o', linestyle='-', color='b')
    plt.title("Zipf's Law")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


large_text = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')

plot_zipf(large_text)

import random
import string


def generate_random_text(length):
    return ''.join(random.choice(string.ascii_lowercase + ' ') for _ in range(length))


random_text = generate_random_text(1000000)

plot_zipf(random_text)


#24
import nltk
import random

def generate_text(n):
    text = nltk.corpus.genesis.words('english-kjv.txt')
    bigrams = nltk.bigrams(text)
    cfd = nltk.ConditionalFreqDist(bigrams)

    most_common = cfd['God'].most_common(n)
    words = [word for word, _ in most_common]

    chosen_word = random.choice(words)

    return chosen_word

def generate_text_genre(genre, start_word, num_words):
    words = []
    for i in range(num_words):
        words.append(start_word)
        text = nltk.corpus.brown.words(categories=genre)
        bigrams = nltk.bigrams(text)
        cfd = nltk.ConditionalFreqDist(bigrams)
        start_word = cfd[start_word].max()
    return ' '.join(words)

def generate_text_hybrid_genre(genre1, genre2, start_word, num_words):
    words = []
    for i in range(num_words):
        words.append(start_word)
        text1 = nltk.corpus.brown.words(categories=genre1)
        text2 = nltk.corpus.brown.words(categories=genre2)
        text = text1 + text2
        bigrams = nltk.bigrams(text)
        cfd = nltk.ConditionalFreqDist(bigrams)
        start_word = cfd[start_word].max()
    return ' '.join(words)

n = 10
chosen_word = generate_text(n)
print("Randomly chosen word from the most likely:", chosen_word)

# b
genre = 'news'
start_word = 'The'
num_words = 50
random_text_genre = generate_text_genre(genre, start_word, num_words)
print("\nRandom text generated for the genre:", random_text_genre)

#c
genre1 = 'news'
genre2 = 'romance'
start_word = 'She'
num_words = 50
random_text_hybrid_genre = generate_text_hybrid_genre(genre1, genre2, start_word, num_words)
print("\nRandom text generated for the hybrid genre:", random_text_hybrid_genre)


#25
import nltk

def find_language(word):
    languages = []
    for fileid in nltk.corpus.udhr.fileids():
        if fileid.endswith('-Latin1') and word in nltk.corpus.udhr.words(fileid):
            languages.append(fileid[:-7])  # Remove '-Latin1' suffix to get language name
    return languages


word = 'freedom'
result = find_language(word)
if result:
    print(f"The word '{word}' is found in the following languages in the UDHR corpus:")
    for language in result:
        print(language)
else:
    print(f"The word '{word}' is not found in any language in the UDHR corpus.")
