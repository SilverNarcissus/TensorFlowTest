import random
import re
import json
import jieba

import legal_instrument.system_path as constant
import pickle
import tensorflow as tf
import numpy as np

# param
embedding_size = 128
num_sampled = 64
vocabulary_size = 10000
batch_size = 128


##
## change linux file to windows file
def change_new_line():
    file = open('C:\\Users\\njdx\\Desktop\\文书\\windows_law.txt', "w", encoding="UTF-8")
    with open('C:\\Users\\njdx\\Desktop\\文书\\law.txt', "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            line = line.replace('\n', '\r\n')
            file.write(line)
            line = f.readline()


def read_accu():
    accu_dict = dict()
    with open(constant.ACCU_FILE, "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            accu_dict[line.strip()] = len(accu_dict)
            line = f.readline()
    reverse_accu_dict = dict(zip(accu_dict.values(), accu_dict.keys()))

    return accu_dict, reverse_accu_dict


def get_dictionary_and_embedding():
    with open("dump_embedding.txt", "rb") as f:
        embedding = pickle.load(f)
    with open("dump_dict.txt", "rb") as f:
        word_dictionary = pickle.load(f)

    return word_dictionary, embedding, dict(zip(word_dictionary.values(), word_dictionary.keys()))


def change_fact_to_vector(fact, embedding, dictionary):
    result = np.zeros(embedding_size)
    for word in list(jieba.cut(fact, cut_all=False)):
        if word in dictionary:
            result += embedding[dictionary[word]]

    return result


# label : 数据列，one-hot编码之后非零的列
# max : 总类数
def change_label_to_one_hot(label, max):
    # n_sample = label.shape[0]
    # one_hot = np.zeros((n_sample, max))  # 3个样本，4个类别
    # one_hot[:, label] = 1  # 非零列赋值为1
    return np.eye(max)[label]


# read file into memory
def read_data():
    data = []
    # control data size
    i = 0
    with open(constant.DATA_TRAIN, "r", encoding="UTF-8") as f:
        line = f.readline()
        while line and i < 100:
            i = i + 1
            data.append(json.loads(line))
            line = f.readline()

    return data


def generate_accu_batch(batch_size, data, accu_size, embedding, dictinary, accu_dict):
    x = np.ndarray([batch_size, embedding_size])
    y = np.ndarray([batch_size], dtype=int)
    i = 0
    while i < batch_size:
        obj = data[random.randint(0, len(data) - 1)]
        l = obj['meta']['accusation']
        for index, accusation in enumerate(l):
            if accusation in accu_dict:
                x[i] = change_fact_to_vector(obj['fact'], embedding, dictinary)
                y[i] = accu_dict[accusation]
                i = i + 1
                if i >= batch_size:
                    break

    y = change_label_to_one_hot(y, accu_size)

    return x, y


## execute begin here
accu_dict, reverse_accu_dict = read_accu()
acuu_size = len(accu_dict)
word_dict, embedding, reverse_dictionary = get_dictionary_and_embedding()
print("reading...")
data = read_data()
# test_fact = '公诉机关指控：2016年3月28日20时许，被告人颜某在本市洪山区马湖新村足球场马路边捡拾到被害人谢某的VIVOX5手机一部，并在同年3月28日21时起，分多次通过支付宝小额免密支付功能，秘密盗走被害人谢某支付宝内人民币3723元。案发后，被告人颜某家属已赔偿被害人全部损失，并取得谅解。公诉机关认为被告人颜某具有退赃、取得谅解、自愿认罪等处罚情节，建议判处被告人颜某一年以下××、××或者××，并处罚金。'

x, y = generate_accu_batch(batch_size, data, acuu_size, embedding, word_dict, accu_dict)
print(len(y))