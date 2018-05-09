import json

import collections
import random
import numpy as np
import jieba

# param
batch_size = 128
embedding_size = 128
skip_window = 2
num_skips = 1
valid_size = 9  # 切记这个数字要和len(valid_word)对应，要不然会报错哦
valid_window = 100
num_sampled = 64  # Number of negative examples to sample.
vocabulary_size = 10000
##

# global
data_index = 0


##

# get fact from the data and write to one file
def get_fact():
    # 读取停用词
    stop_words = []
    with open('stop_words.txt', "r", encoding="UTF-8") as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

    result = []
    with open('/Users/SilverNarcissus/Documents/法院文书/good/data_train.json', "r", encoding="UTF-8") as f:
        line = f.readline()
        for i in range(10):
            obj = json.loads(line)
            raw_words = list(jieba.cut(obj['fact'], cut_all=False))
            for word in raw_words:
                if word not in stop_words:
                    result.append(word)
            result.append('$$')
            line = f.readline()

    return result


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print("count", len(count))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    buffer.append(dictionary['$$'])
    while dictionary['$$'] in buffer:
        fill_buffer(buffer, span)

    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

        while dictionary['$$'] in buffer:
            fill_buffer(buffer, span)

    return batch, labels

def fill_buffer(buffer, span):
    global data_index
    buffer.clear()
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

# execute part
words = get_fact()
data, count, dictionary, reverse_dictionary = build_dataset(words)

del words  # 删除words节省内存
print('Most common words (+UNK)', count[:20])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

x, y = generate_batch(batch_size, num_skips, skip_window)
for i in range(batch_size):
    print(i)
    print(reverse_dictionary[x[i]], reverse_dictionary[y[i][0]])

# print(get_fact())
