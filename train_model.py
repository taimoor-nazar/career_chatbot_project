# Chatbot Model Training

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
import nltk
import joblib

df = pd.read_csv('career_guidance_dataset.csv')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

role_names = [re.escape(role.lower()) for role in df['role'].unique()]
role_pattern = re.compile(r'\b(' + '|'.join(role_names) + r')\b')

def preprocess_text(text, remove_roles=False, role_pattern=None):
    text = text.lower()
    if remove_roles and role_pattern is not None:
        text = role_pattern.sub('', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    # Lemmatize each word and remove stopwords
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(lemmatized_words)


#Encoding the roles
le_role = LabelEncoder()
df['roles_encoded'] = le_role.fit_transform(df['role'])

x = df['question']
y = df['roles_encoded']

#Train-Test Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(df['question'], df['roles_encoded']):
    x_train = df.loc[train_idx, 'question']
    x_test = df.loc[test_idx, 'question']
    y_train = df.loc[train_idx, 'roles_encoded']
    y_test = df.loc[test_idx, 'roles_encoded']

#Preprocessing

x_train = x_train.apply(lambda x: preprocess_text(x, remove_roles=True, role_pattern=role_pattern))
x_test = x_test.apply(lambda x: preprocess_text(x, remove_roles=True, role_pattern=role_pattern))

#Converting text to numerical format using TF-IDF Vectorizer

vectorizer = TfidfVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

#Training 
model = MultinomialNB()
model.fit(x_train_vectorized, y_train)
y_pred = model.predict(x_test_vectorized)

#Save model
joblib.dump(model, 'intent_model.pkl')
#Save vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')
# Save label encoder
joblib.dump(le_role, 'label_encoder.pkl')