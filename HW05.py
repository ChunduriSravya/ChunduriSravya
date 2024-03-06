#11
import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg

text = gutenberg.words('austen-persuasion.txt')

brown_sentences = brown.sents(categories='news')
brown_tagged_sentences = brown.tagged_sents(categories='news')
affix_tagger = nltk.AffixTagger(train=brown_tagged_sentences, affix_length=1, min_stem_length=3)
print(affix_tagger.tag(text))


#14
import nltk
nltk.download('brown')
tagged_words = nltk.corpus.brown.tagged_words()
tags = set(tag for word, tag in tagged_words)
sorted_tags = sorted(tags)

print(sorted_tags)

#15
#a
import nltk
nltk.download('brown')

tagged_words = nltk.corpus.brown.tagged_words()

plural_nouns = [word.lower() for word, tag in tagged_words if tag == 'NNS']

plural_counts = nltk.FreqDist(plural_nouns)

singular_forms = [word[:-1] for word in plural_counts.keys() if word.endswith('s')]

more_common_in_plural = [(singular, plural_counts[singular + 's']) for singular in singular_forms if plural_counts[singular] < plural_counts[singular + 's']]

print("Nouns more common in plural form:")
for singular, count in more_common_in_plural:
    print(f"{singular}s: {count} occurrences")

#b
import nltk

nltk.download('brown')
tagged_words = nltk.corpus.brown.tagged_words()

word_tag_counts = {}
for word, tag in tagged_words:
    word_tag_counts[word] = word_tag_counts.get(word, set())
    word_tag_counts[word].add(tag)

word_with_most_tags = max(word_tag_counts, key=lambda word: len(word_tag_counts[word]))
most_tags = word_tag_counts[word_with_most_tags]

print("Word with the greatest number of distinct tags:")
print(f"Word: {word_with_most_tags}")
print("Distinct tags:")
for tag in most_tags:
    print(tag)


#c
import nltk

nltk.download('brown')

tagged_words = nltk.corpus.brown.tagged_words()

tag_counts = nltk.FreqDist(tag for word, tag in tagged_words)

sorted_tags = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)

print("Tags in order of decreasing frequency:")
for tag, count in sorted_tags[:20]:
    print(f"{tag}: {count} occurrences")

#d
import nltk

nltk.download('brown')

tagged_words = nltk.corpus.brown.tagged_words()

noun_following_tags = [tag for (prev_word, prev_tag), (word, tag) in nltk.bigrams(tagged_words)
                       if prev_tag.startswith('NN') and tag.startswith('N')]

following_tag_counts = nltk.FreqDist(noun_following_tags)

print("Tags most commonly found after nouns:")
for tag, count in following_tag_counts.most_common(5):
    print(f"{tag}: {count} occurrences")
