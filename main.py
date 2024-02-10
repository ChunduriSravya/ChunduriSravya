# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


#6
import nltk
from nltk.book import *
text2.dispersion_plot(['Elinor', 'Marianne', 'Edward', 'Willoughby'])



#14
import nltk
from nltk.book import text3

sent3 = text3[:11]

indexes = [i for i, word in enumerate(sent3) if word == 'the']
print("Indexes of 'the' in sent3:", indexes)

#17
from nltk.book import text9

sunset_index = text9.index('sunset')

start_index = sunset_index
end_index = sunset_index

while start_index > 0 and text9[start_index] != '.':
    start_index -= 1

while end_index < len(text9) - 1 and text9[end_index] != '.':
    end_index += 1

start_index += 1
end_index += 1

complete_sentence = ' '.join(text9[start_index:end_index])

print("Complete sentence containing 'sunset':", complete_sentence)

#20
#w.isupper(): This method checks whether all characters in the string w are uppercase letters.
# If all characters are uppercase, it returns True; otherwise, it returns False.
#not w.islower(): This expression checks whether the string w contains at least one character that is not a lowercase letter.
# If w contains at least one character that is not lowercase, it returns True; otherwise, it returns False.
# The not operator negates the result of w.islower(), so if w.islower() returns False (i.e., if there is at least one non-lowercase character), then not w.islower() will be True.

#w.isupper() returns True if all characters in w are uppercase.
#not w.islower() returns True if w contains at least one character that is not lowercase.

#28
def percent(word, text):

    word_count = text.count(word)

    total_words = len(text)

    percentage = (word_count / total_words) * 100

    return percentage



sample_text = ["apple", "banana", "apple", "orange", "apple", "grape"]

apple_percentage = percent("apple", sample_text)

print("Percentage of occurrences of 'apple' in the text:", apple_percentage)


#29

sent3 = {'the', 'sun', 'is', 'setting'}
text1 = {'the', 'sun', 'is', 'setting', 'in', 'the', 'west'}

is_subset = set(sent3) < set(text1)

print("Is sent3 a proper subset of text1?", is_subset)


