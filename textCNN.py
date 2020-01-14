from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv1D, MaxPooling1D, concatenate
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.layers import Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import os.path
import config

max_len = 300
batch_size = 64
embedding_dim = 100
max_feature = 20000


def text_cnn(maxlen=max_len, max_features=max_feature, embed_size=embedding_dim):
    conment_seq = Input(shape=[maxlen], name='x_seq')
    emb_comment = Embedding(max_features, embed_size)(conment_seq)
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.3)(merge)
    output = Dense(32, activation='relu')(out)
    # activity_regularizer = regularizers.l1(0.01),kernel_regularizer = regularizers.l2(0.01)

    output = Dense(units=3, activation='softmax')(output)
    model = Model([conment_seq], output)
    a = optimizers.Adam(lr=2e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss="categorical_crossentropy", optimizer=a, metrics=['accuracy'])
    return model


def tokenize(lang, max_len):
    # 这里max_features是要挑选最常用的词汇
    tokenizer = Tokenizer(filters='', lower=False, num_words=max_feature)
    tokenizer.fit_on_texts(lang)
    # 114k 的词典
    # 注意：keras API的word_index是从1开始的，0要预留出来，因为进行了补零操作
    word_index = tokenizer.word_index
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = pad_sequences(tensor, padding='post', maxlen=max_len)
    return tensor, tokenizer, word_index


history = pd.read_csv(os.path.join(config.output_dir, 'history.csv'), encoding='utf8')
data = list(history.itertuples(index=False))
x, y = zip(*data)
y_tag = []
x, tokenizer, word_index = tokenize(x, max_len)
for tag in y:
    if tag == 'gd':
        y_tag.append([1, 0, 0])
    elif tag == 'xd':
        y_tag.append([0, 1, 0])
    else:
        y_tag.append([0, 0, 1])
y_tag = np.array(y_tag)
x_train, x_test, y_train, y_test = train_test_split(x, y_tag)


model_class = text_cnn(max_len, max_feature, embedding_dim)
model_class.summary()

history = model_class.fit(x_train, y_train,
                          batch_size=batch_size, epochs=5, validation_split=0.1)
y_pre = model_class.predict(x_test)

print(model_class.evaluate(x_test, y_test))

# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test, y_pre))