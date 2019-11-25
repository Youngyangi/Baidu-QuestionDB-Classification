from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import random
import os
import config
import pandas as pd

history = pd.read_csv(os.path.join(config.output_dir, 'history.csv'), encoding='utf8')
data = list(history.itertuples(index=False))
random.shuffle(data)
x, y = zip(*data)
x_train, x_test, y_train, y_test = train_test_split(x, y)
print(len(x_train))
print(len(y_train))
bow = CountVectorizer(analyzer='word', ngram_range=(1, 1))
tf_idf = TfidfVectorizer()
metrics = bow.fit_transform(x_train)
# metrics = tf_idf.fit_transform(x_train)
print(metrics.toarray().shape)
model = MultinomialNB()
model.fit(metrics, y_train)
print(model.predict(bow.transform(x_test[:1])))
print(model.score(bow.transform(x_test), y_test))
# print(model.score(tf_idf.transform(x_test), y_test))

