#! -*- coding:utf-8 -*-
# 句子对分类任务
# label说明： 0表示两句话修改后幽默程度一样，1表示第一句话更幽默，2表示第二句话更幽默

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizer import Tokenizer
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator


maxlen = 50
batch_size = 16
config_path = 'E:/Pretrained_Model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'E:/Pretrained_Model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'E:/Pretrained_Model/chinese_L-12_H-768_A-12/vocab.txt'


def label_passer(labels):
    i = 0
    label_dict = {}
    for label in labels:
        if label not in label_dict:
            label_dict[label] = i
            i += 1
    return label_dict


def load_data(filename):
    D = []
    df = pd.read_csv(filename, encoding='utf8')
    text = df['question'].tolist()
    label = df['tag'].tolist()
    label_dict = label_passer(label)
    for x, y in zip(text, label):
        D.append((x, label_dict[y]))
    return D, label_dict


# 加载数据集
train_data, label_dict = load_data("D:/Workstations/Baidu-QuestionDB-Classification/Data/Output/history.csv")
# valid_data = load_data('datasets/lcqmc/lcqmc.valid.data')
# test_data = load_data('datasets/lcqmc/lcqmc.test.data')

text_train, text_valid = train_test_split(train_data, random_state=2019, test_size=0.1)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=False)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text1, text2, label = self.data[i]
            token_ids, segment_ids = tokenizer.encode(text1, text2, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=True
)

output = tf.keras.layers.Dropout(rate=0.1)(bert.output)
output = tf.keras.layers.Dense(units=3, activation='softmax', name='classifier')(output)

model = keras.models.Model(bert.input, output)
model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=Adam(2e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)


def embedding(data):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for i in range(len(data)):
        text, label = data[i]
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    batch_labels = sequence_padding(batch_labels)
    return batch_token_ids, batch_segment_ids, batch_labels


# 转换数据集
# train_generator = data_generator(text_train, batch_size)
# valid_generator = data_generator(text_valid, batch_size)
# test_generator = data_generator(test_data, batch_size)

train = embedding(text_train)
train = tf.data.Dataset.from_tensor_slices(((train[0], train[1]), train[2])).batch(batch_size, drop_remainder=True)

valid = embedding(text_train)
valid = tf.data.Dataset.from_tensor_slices(((valid[0], valid[1]), valid[2])).batch(batch_size, drop_remainder=True)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_train_begin(self, logs=None):
        with open('Data/Output/BERT/label_dict.txt', 'w') as f:
            for label, id in label_dict.items():
                f.write(str(label)+'\t'+str(id))

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('Data/Output/BERT/best_model.weights')
        # test_acc = evaluate(test_generator)
        print(u'val_acc: %05f, best_val_acc: %05f, test_acc: None\n'
              % (val_acc, self.best_val_acc))


evaluator = Evaluator()
tensorboard = tf.keras.callbacks.TensorBoard("Data/Output/BERT/logs", update_freq='batch')
model.fit(train, epochs=2, validation_data=valid, callbacks=[evaluator,tensorboard])

model.load_weights('Data/Output/BERT/best_model.weights')
# print(u'final test acc: %05f\n' % (evaluate(test_generator))
