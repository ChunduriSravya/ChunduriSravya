#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument  
from nltk.corpus import brown

# Load the Brown Corpus
brown_corpus = brown.sents(categories=brown.categories())

# Prepare data for training
tagged_data = []
for category in brown.categories():
    for doc in brown_corpus:
        tagged_data.append(TaggedDocument(words=doc, tags=[category]))

# Initialize and train Doc2Vec model
model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Classify documents from each category
for category in brown.categories():
    print("Category:", category)
    category_docs = [doc for doc in brown_corpus if category in brown.categories(doc)]
    for doc in category_docs:
        vector = model.infer_vector(doc)
        print("Inferred Vector:", vector)
    print("--------------")


# In[ ]:


pip install scikit-learn


# In[ ]:


pip install --upgrade scikit-learn


# In[ ]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import FunctionTransformer

# Fetch the 20 newsgroups dataset
data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

# Function to remove stop words from text
def remove_stopwords(text_list):
    stop_words = set(stopwords.words('english'))
    filtered_text_list = []
    for text in text_list:
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        filtered_text_list.append(' '.join(filtered_text))
    return filtered_text_list

# Define a function to compute TF-IDF vectors
def compute_tfidf_vectors(data):
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(data)
    return tfidf_vectors, vectorizer

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Compute TF-IDF vectors for both training and testing data
X_train_tfidf, vectorizer = compute_tfidf_vectors(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Fit a logistic regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_tfidf)

# Calculate accuracy without stop word removal
accuracy_without_stopwords = accuracy_score(y_test, y_pred)
print("Accuracy without stop word removal:", accuracy_without_stopwords)

# Apply stop word removal
stopword_transformer = FunctionTransformer(remove_stopwords)
X_train_processed = stopword_transformer.fit_transform(X_train)
X_test_processed = stopword_transformer.transform(X_test)

# Compute TF-IDF vectors for processed data
X_train_tfidf_processed = vectorizer.transform(X_train_processed)
X_test_tfidf_processed = vectorizer.transform(X_test_processed)

# Fit a logistic regression classifier on processed data
classifier_processed = LogisticRegression(max_iter=1000)
classifier_processed.fit(X_train_tfidf_processed, y_train)

# Make predictions on processed testing data
y_pred_processed = classifier_processed.predict(X_test_tfidf_processed)

# Calculate accuracy with stop word removal
accuracy_with_stopwords = accuracy_score(y_test, y_pred_processed)
print("Accuracy with stop word removal:", accuracy_with_stopwords)


# In[ ]:




