#6
import nltk


test_strings = [
    "Hello World",
    "Abcdefg",
    "pot",
    "pt",
    "paeiot",
    "123.45",
    "0.789",
    "bcdb",
    "xyz",
]

regex_patterns = [
    r'[a-zA-Z]+',
    r'[A-Z][a-z]*',
    r'p[aeiou]{,2}t',
    r'\d+(\.\d+)?',
    r'([^aeiou][aeiou][^aeiou])*',
    r'\w+|[^\w\s]+',
]

for regex in regex_patterns:
    print("Regular Expression:", regex)
    for string in test_strings:
        nltk.re_show(regex, string)
    print("\n---\n")

#7
import re

determiner_pattern = r'\b(a|an|the)\b'

test_string = "I have a pen and the book is on the table."

matches = re.findall(determiner_pattern, test_string)

print(matches)

import re

arithmetic_expression_pattern = r'\d+(\*\d+|\+\d+)*'

test_string = "The result of 2*3+8 is 14."

matches = re.findall(arithmetic_expression_pattern, test_string)

print(matches)


#21
import re
import requests
import nltk


def unknown(url):
    response = requests.get(url)
    html_content = response.text

    lowercase_words = re.findall(r'\b[a-z]+\b', html_content.lower())

    nltk.download('words')
    english_words = set(nltk.corpus.words.words())

    unknown_words = [word for word in lowercase_words if word not in english_words]

    return unknown_words


url = "https://blackboard.umbc.edu/ultra/courses/_78819_1/cl/outline"
unknown_words = unknown(url)
print("Unknown words on the webpage:", unknown_words)

#30
from nltk.stem import PorterStemmer, LancasterStemmer

tokenized_text = ["running", "flies", "happier", "cats", "going", "wonderful", "unhappy", "meeting"]

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

porter_normalized = [porter_stemmer.stem(word) for word in tokenized_text]

lancaster_normalized = [lancaster_stemmer.stem(word) for word in tokenized_text]

print("Original Tokenized Text:", tokenized_text)
print("Porter Stemmer Normalized:", porter_normalized)
print("Lancaster Stemmer Normalized:", lancaster_normalized)


#39
class Soundex:
    def __init__(self):
        self.soundex_mapping = {
            'b': 1, 'f': 1, 'p': 1, 'v': 1,
            'c': 2, 'g': 2, 'j': 2, 'k': 2, 'q': 2, 's': 2, 'x': 2, 'z': 2,
            'd': 3, 't': 3,
            'l': 4,
            'm': 5, 'n': 5,
            'r': 6
        }

    def soundex(self, word):
        word = word.lower()

        word = ''.join(filter(str.isalpha, word))
        
        soundex_code = word[0]

        for char in word[1:]:
                if char in self.soundex_mapping:
                    digit = self.soundex_mapping[char]
                    if digit != soundex_code[-1]:
                        soundex_code += str(digit)
        
        soundex_code = soundex_code.replace('a', '').replace('e', '').replace('i', '').replace('o', '').replace('u', '')
        soundex_code = soundex_code.replace('h', '').replace('w', '').replace('y', '')

        soundex_code += '000'
        soundex_code =  soundex_code[:4]
        return soundex_code


s = Soundex()
print(s.soundex("Washington"))




#chapter-4
#14
def novel10(text):
    words = text.split()
    last_10_percent_index = int(len(words) * 0.9)
    unique_words = set()

    for word in words[last_10_percent_index:]:
        if word not in unique_words:
            print(word)
            unique_words.add(word)

text = "This is a sample text. We will test the novel10 function on this text. This function prints words that appeared in the last 10% of the text and had not been encountered earlier."
novel10(text)

#20
def sort_by_frequency(words):
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1

    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    sorted_words = [word for word, _ in sorted_word_count]

    return sorted_words


input_words = ["table", "chair", "table", "chair", "table", "table", "chair", "chair", "table", "table"]
sorted_unique_words = sort_by_frequency(input_words)
print(sorted_unique_words)

#25
import nltk

str1 = "kitten"
str2 = "sitting"

edit_distance = nltk.edit_distance(str1, str2)
print("Edit distance between '{}' and '{}' is: {}".format(str1, str2, edit_distance))

#note: The Levenshtein distance is in fact implemented using dynamic programming in the nltk.edit_dist() function.
# In order to efficiently compute the edit distance, a bottom-up approach is used. The technique achieves better temporal complexity by avoiding unnecessary calculations by first calculating the distances for smaller substrings, and then using these results to solve bigger subproblems.

#26
#a
def catalan_recursive(n):
    if n <= 1:
        return 1
    else:
        res = 0
        for i in range(n):
            res += catalan_recursive(i) * catalan_recursive(n - i - 1)
        return res
#b
def catalan_dynamic(n):
    if n <= 1:
        return 1

    catalan = [0] * (n + 1)
    catalan[0], catalan[1] = 1, 1

    for i in range(2, n + 1):
        for j in range(i):
            catalan[i] += catalan[j] * catalan[i - j - 1]

    return catalan[n]

#c
import timeit

def compare_performance():
    for n in range(10, 21):
        print(f"n = {n}:")
        print("Recursive approach:", timeit.timeit(lambda: catalan_recursive(n), number=1000))
        print("Dynamic programming approach:", timeit.timeit(lambda: catalan_dynamic(n), number=1000))
        print()

compare_performance()

