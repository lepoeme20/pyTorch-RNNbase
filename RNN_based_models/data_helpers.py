import numpy as np
import itertools
import re
import codecs
import json
import torch
import time
import sys

####################################################################################
####################################################################################
#############                                                    ###################
#############       make data set scaling unbalanced or not      ###################
#############                                                    ###################
####################################################################################
####################################################################################


def load_json(json_path):
    data_from_json = []
    for line in codecs.open(json_path, 'rb', encoding='utf-8'):
        data_from_json.append(json.loads(line))

    data = make_data(data_from_json)
    return data


# positive_labels = [[0, 1] for _ in positive_examples]
# negative_labels = [[1, 0] for _ in negative_examples]
def make_data(data_from_json):
    x_text = []
    y = []
    for i, x in enumerate(data_from_json):
        if x['overall'] != 3.:
            x_text.append(x['reviewText'])
            if x['overall'] == 1. or x['overall'] == 2.:
                y_tmp = [1, 0]
                y.append(y_tmp)
            elif x['overall'] == 4. or x['overall'] == 5.:
                y_tmp = [0, 1]
                y.append(y_tmp)
    return [x_text, y]

####################################################################################
####################################################################################
#############                                                    ###################
#############                     basic tokenizer                ###################
#############                                                    ###################
####################################################################################
####################################################################################


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


####################################################################################
####################################################################################
#############                                                    ###################
#############     calculate max length of tokenized sentence     ###################
#############                                                    ###################
####################################################################################
####################################################################################


def max_len(sentence_list):
    sen_len = np.empty((1,len(sentence_list)), int)
    for i, x in enumerate(sentence_list):
        clean_sen = clean_str(x)
        word_list = clean_sen.split(" ")
        sen_len[0][i] = len(word_list)
    return np.max(list(sen_len.flat)), list(sen_len.flat)

####################################################################################
####################################################################################
#############                                                    ###################
#############        convert word to index and fill zeros        ###################
#############                                                    ###################
####################################################################################
####################################################################################

def word2idx_array(sentence_list, length):
    word_to_idx = {}
    idx_array = np.zeros((len(sentence_list), length))
    count = 0
    start = time.time()
    for i, x in enumerate(sentence_list):
        idx_tmp = np.empty((0, 1), int)
        clean_sen = clean_str(x)
        word_list = clean_sen.split(" ")

        for word in word_list:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx) + 1  # +1 to leave out zero for padding
            idx_tmp = np.vstack((idx_tmp, int(word_to_idx[word])))

        if length > len(idx_tmp):
            num_zeros = length - len(idx_tmp)
            zeros_array = np.zeros((1, num_zeros))
            sen_max_len = np.hstack((idx_tmp.T, zeros_array))
            idx_array[i] = sen_max_len
        else:
            idx_array[i] = idx_tmp[:length].T

        count += 1
        if count % 1000 == 0:
            end = time.time()
            sys.stdout.write(
                "\rI'm working at word2idx FN %({}/{}, {})".format(count,
                                                                   len(sentence_list),
                                                                   end - start))
            start = end
    return idx_array, word_to_idx


def sorting_sequence(x_data, y_data, sequence, args):
    sorted_x = torch.LongTensor(len(x_data), args.seq_len).zero_()
    sorted_y = torch.LongTensor(len(y_data), args.num_classes).zero_()
    sorted_seq = []
    index = [i[0] for i in sorted(enumerate(sequence), key=lambda x: x[1])]
    index.reverse()
    for i, x in enumerate(index):
        sorted_x[i] = x_data[x]
        sorted_y[i] = y_data[x]
        sorted_seq.append(sequence[x])

    return [sorted_x, sorted_y, sorted_seq]


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def default_collate(batch):
    seq_list = []
    data, labels, seq = zip(*batch)
    # print("data", data)
    # print("labels", labels)
    data = torch.stack(data, dim=0)
    # labels = torch.stack(torch.LongTensor( [lbl.tolist() for lbl in labels] ), dim=0)
    labels = torch.stack(labels, dim=0)
    seq_list.append(list(seq))
    return data, labels, list(itertools.chain.from_iterable(seq_list))


def lr_decay(loss, args):
    _loss = loss.data.cpu().numpy()[0]

    if len(args.lr_decay) <= 10:
        args.lr_decay.append(_loss)
    else:
        args.lr_decay.pop(0)
        args.lr_decay.append(_loss)
        if args.lr_decay[0] * .75 > args.lr_decay[len(args.lr_decay) - 3]\
                and args.lr_decay[1] * .75 > args.lr_decay[len(args.lr_decay) - 2]\
                and args.lr_decay[2] * .75 > args.lr_decay[len(args.lr_decay) - 1]:
            args.lr = args.lr * 0.1
            print("")
            print('LR is set to {}'.format(args.lr))

    return args.lr
