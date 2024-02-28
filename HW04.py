from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


stop_words_list = stopwords.words('english')

document = fetch_20newsgroups().data
tokens = []

for doc in document:
    words = word_tokenize(doc.lower())  # Tokenize and convert to lowercase
    words_filtered = [word for word in words if word.isalpha() and word not in stop_words_list]
    tokens.extend(words_filtered)


frequency_distribution_T = FreqDist(tokens)
total_documents = len(document)

# Identifying words appearing in more than 97.5% of documents
high_freq_words = {word: freq for word, freq in frequency_distribution_T.items() if freq / total_documents > 0.975}
# Identifying words appearing in less than 2.5% of documents
low_freq_words = {word: freq for word, freq in frequency_distribution_T.items() if freq / total_documents < 0.025}


high_freq_words_list = list(high_freq_words.keys())
low_freq_words_list = list(low_freq_words.keys())
identified_freq_words = high_freq_words_list + low_freq_words_list


data = pd.DataFrame({'doc': fetch_20newsgroups().data, 'target': fetch_20newsgroups().target})


train_data, test_data = train_test_split(data)


vectorizer = CountVectorizer(stop_words=identified_freq_words)
pipeline = Pipeline([('vectorizer', vectorizer),
                     ('classifier', MultinomialNB())])
pipeline.fit(train_data['doc'], train_data['target'])
print('Training accuracy:', accuracy_score(train_data['target'], pipeline.predict(train_data['doc'])))
print('Testing accuracy:', accuracy_score(test_data['target'], pipeline.predict(test_data['doc'])))