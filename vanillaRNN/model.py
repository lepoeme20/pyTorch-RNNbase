import sys
import os
sys.path.append(os.getcwd()+'/vanillaRNN')
import torch
import argparse
import datetime
import numpy as np
import time
from torch.autograd import Variable
import torch.nn as nn
import data_helpers
from RNN import RNN
#from vanillaRNN import data_helpers
#from vanillaRNN.RNN import RNN
import math
from sklearn.metrics import confusion_matrix, auc, roc_curve

parser = argparse.ArgumentParser(description='Vanilla RNN')

# Model Hyperparameters
parser.add_argument('-lr', type=float, default=1e-5, help='setting learning rate')
parser.add_argument('-hidden-size', type=int, default=128, help='setting hidden size [default : 128]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-num-layers', type=int, default=2, help='setting number of layers [default : 1]')

# Training parameters
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-num-epochs', type=int, default=100, help='number of epochs for train [default: 200]')
parser.add_argument('-dev-interval', type=int, default=1000, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-list4ES', type=list,  default=[], help='Empty list for appending dev-acc')
parser.add_argument('-corrects-index', type=list,  default=[], help='Empty list for appending dev-acc')

# Data Set
parser.add_argument('-json-path', type=str, default="../data/amazon/Video_Games_5.json", help='Data source')
parser.add_argument('-vocab-size', type=int, default=0 , help='Vocab size')
parser.add_argument('-max-len', type=int, default=0 , help='max length among all of sentences')
parser.add_argument('-data-size', type=int, default=0, help='Data size')
parser.add_argument('-num-classes', type=int, default=2, help='Number of classes')
parser.add_argument('-trn-sample-percentage', type=float, default=.5, help='Percentage of the data to use for training')
parser.add_argument('-dev-sample-percentage', type=float, default=.2, help='Percentage of the data to use for validation')
parser.add_argument('-test-sample-percentage', type=float, default=.3, help='Percentage of the data to use for testing')
parser.add_argument('-seq-len', type=int, default=0, help='setting input size')

# saver
parser.add_argument('-iter', type=int, default=0, help='For checking iteration')
parser.add_argument('-save-dir', type=str, default='../RUNS/', help='Data size')
parser.add_argument('-final-model-dir', type=str, default='../Final_model/', help='Dir to saving learned model')
parser.add_argument('-snapshot', type=str, default='../RUNS/Final_model/', help='dir learned model')
parser.add_argument('-model-name', type=str, default='vanillaRNN', help='Model name')
parser.add_argument('-data-name', type=str, default='Video_Games_5', help='Data name')


args, unknown = parser.parse_known_args()
# Instantiate RNN model



print("Loading data...")
x_text, y = data_helpers.load_json(args.json_path)
#max_len, seq_num = data_helpers.max_len(x_text)
median_len, seq_num = data_helpers.median_len(x_text)
x, vocab_dic = data_helpers.word2idx_array(x_text, median_len)
x = np.array(x)
y = np.array(y)
seq_num = np.array(seq_num)

# Randomly shuffle data
np.random.seed(int(time.time()))
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
seq_num_shuffled = seq_num[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
trn_sample_index = -1 * int(args.trn_sample_percentage * float(len(y)))
test_sample_index = -1 * int(args.test_sample_percentage * float(len(y)))
x_train, x_dev, x_test = x_shuffled[:trn_sample_index], x_shuffled[trn_sample_index:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_dev, y_test = y_shuffled[:trn_sample_index], y_shuffled[trn_sample_index:test_sample_index], y_shuffled[test_sample_index:]
seq_train, seq_dev, seq_test = seq_num_shuffled[:trn_sample_index], seq_num_shuffled[trn_sample_index:test_sample_index], seq_num_shuffled[test_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_dic)))
print("Train/Dev split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))

####################################################################################################################################
print("Start to saving data")
import pickle
input_path = '{}/{}_{}'.format('./data/inputs', args.model_name, args.data_name)
if not os.path.isdir(input_path): os.makedirs(input_path)
pickle.dump(x_train, open('{}/{}.p'.format(input_path,'x_train'), 'wb'),protocol=4)
pickle.dump(x_dev, open('{}/{}.p'.format(input_path, 'x_dev'), 'wb'))
pickle.dump(x_test, open('{}/{}.p'.format(input_path, 'x_test'), 'wb'))
pickle.dump(y_train, open('{}/{}.p'.format(input_path, 'y_train'), 'wb'))
pickle.dump(y_dev, open('{}/{}.p'.format(input_path, 'y_dev'), 'wb'))
pickle.dump(y_test, open('{}/{}.p'.format(input_path, 'y_test'), 'wb'))
pickle.dump(seq_train, open('{}/{}.p'.format(input_path, 'seq_train'), 'wb'))
pickle.dump(seq_dev, open('{}/{}.p'.format(input_path, 'seq_dev'), 'wb'))
pickle.dump(seq_test, open('{}/{}.p'.format(input_path, 'seq_test'), 'wb'))
pickle.dump(vocab_dic, open('{}/{}.p'.format(input_path, 'vocab_dic'), 'wb'))
pickle.dump(median_len, open('{}/{}.p'.format(input_path, 'median_len'), 'wb'))
print("Saving is over")
####################################################################################################################################
'''
import pickle
x_train = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/x_train.p', 'rb'))
x_test = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/x_test.p', 'rb'))
x_dev = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/x_dev.p', 'rb'))
y_train = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/y_train.p', 'rb'))
y_test = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/y_test.p', 'rb'))
y_dev = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/y_dev.p', 'rb'))
seq_train = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/seq_train.p', 'rb'))
seq_test = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/seq_test.p', 'rb'))
seq_dev = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/seq_dev.p', 'rb'))
max_len = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/max_len.p', 'rb'))
vocab_dic = pickle.load(open('../data/inputs/vanillaRNN_Video_Games_5/vocab_dic.p', 'rb'))
'''



# update args and print
args.embed_num = len(vocab_dic)
args.seq_len = int(median_len)
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.snapshot = os.path.join(args.snapshot, '{}_{}.{}'.format(args.model_name, args.data_name,'pt'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))



rnn = RNN(args)
if torch.cuda.is_available():
    rnn.cuda()
    print("model will use GPU")


def train_step(x_train, y_train, x_dev, y_dev, x_test, y_test,  model, args):

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted


    model.train()
    model.zero_grad()

    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), args.batch_size, args.num_epochs, seq_train)

    print("")
    x_batch_list = []
    y_batch_list = []
    seq_list = []
    for batch, seq in batches:
        x_batch, y_batch = zip(*batch)


        # x_batch_list.append(x_batch)
        # y_batch_list.append(y_batch)
        # seq_list.append(seq)
        # x_batch = x_batch_list[1]
        # y_batch = y_batch_list[1]
        # seq = seq_list[1]


        sorted_x_batch, sorted_y_batch, sorted_seq = data_helpers.sorting_sequence(x_batch, y_batch, seq)

        x_batch_Tensor, y_batch_Tensor = data_helpers.tensor4batch(sorted_x_batch, sorted_y_batch, args)


        # x_batch_Tensor, y_batch_Tensor = data_helpers.tensor4batch(x_batch, y_batch, args)
        x_batch_Variable, y_batch_Variable = Variable(x_batch_Tensor).cuda(), Variable(y_batch_Tensor).cuda()
        #x_batch_Variable, y_batch_Variable = Variable(x_batch_Tensor), Variable(y_batch_Tensor)


        logit = model(x_batch_Variable, sorted_seq)
        logit = logit[0]
        #loss = F.cross_entropy(logit, torch.max(y_batch_Variable, 1)[1])
        loss = loss_func(logit, torch.max(y_batch_Variable, 1)[1])
        loss.backward()
        optimizer.step()

        args.iter += 1

        if args.iter % args.log_interval == 0:
            pred_y = torch.max(logit, 1)[1].data.cpu().numpy().squeeze()
            real_y = torch.max(y_batch_Variable, 1)[1].data.cpu().numpy().squeeze()
            corrects = sum(pred_y == real_y)
            accuracy = 100.0 * corrects/args.batch_size
            sys.stdout.write(
                '\rTrn||Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{} y_pred: {})'.format(args.iter,
                                                                         loss.data[0],
                                                                         accuracy,
                                                                         corrects,
                                                                         args.batch_size,
                                                                         pred_y))

        # if args.iter % args.dev_interval == 0:
        #     dev_step(x_dev, y_dev, x_test, y_test, model, args)
        #
        # if args.iter % args.save_interval == 0:
        #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
        #     save_prefix = os.path.join(args.save_dir, 'snapshot')
        #     save_path = '{}_steps{}.pt'.format(save_prefix, args.iter)
        #     torch.save(model, save_path)

    print("training is over")
    if not os.path.isdir(args.final_model_dir): os.makedirs(args.final_model_dir)
    save_prefix = os.path.join(args.final_model_dir, args.model_name)
    save_path = '{}_{}.pt'.format(save_prefix, args.data_name)
    torch.save(model, save_path)
    test(x_test, y_test, args, save_prefix)




print("make dev_step def")
def dev_step(x_dev, y_dev, x_test, y_test, model, args, Dev = True):
    model.eval()
    corrects_dev, avg_loss, iter_dev, avg_auc = 0, 0, 0, 0
    loss_func = nn.CrossEntropyLoss()

    batches_dev = data_helpers.batch_iter(
        list(zip(x_dev, y_dev)), args.batch_size, 1, seq_dev)

    print("")
    for batch, seq in batches_dev:
        x_dev_batch, y_dev_batch = zip(*batch)
        x_dev_Tensor, y_dev_Tensor = data_helpers.tensor4batch(x_dev_batch, y_dev_batch, args)


        x_dev_Variable, y_dev_Variable = Variable(x_dev_Tensor).cuda(), Variable(y_dev_Tensor).cuda()

        logit = model(x_dev_Variable, seq)

        iter_dev += 1

        loss = loss_func(logit, torch.max(y_dev_Variable, 1)[1])
        loss_tmp = loss.data.cpu().numpy()[0]

        pred_y = torch.max(logit, 1)[1].data.cpu().numpy().squeeze()
        real_y = torch.max(y_dev_Variable, 1)[1].data.cpu().numpy().squeeze()
        corrects = sum(pred_y == real_y)
        accuracy = 100.0 * corrects / args.batch_size
        sys.stdout.write(
            '\rDev||Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{} )'.format(iter_dev,
                                                                     loss.data[0],
                                                                     accuracy,
                                                                     corrects,
                                                                     args.batch_size
                                                                     ))


        avg_loss += loss_tmp
        corrects_dev += corrects

    size = len(y_dev)
    avg_loss = avg_loss/iter_dev
    accuracy = 100.0 * corrects_dev/size



    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects_dev,
                                                                       size
                                                                       ))
    if Dev:
        args.list4ES.append(accuracy)
        if len(args.list4ES) > 10:
            sub = args.list4ES[len(args.list4ES) - 1] - args.list4ES[len(args.list4ES) - 5]
            if abs(sub) < 0.001:
                print("training is over")
                if not os.path.isdir(args.final_model_dir): os.makedirs(args.final_model_dir)
                save_prefix = os.path.join(args.final_model_dir, args.model_name)
                save_path = '{}_{}.pt'.format(save_prefix, args.data_name)
                torch.save(model, save_path)
                test(x_test, y_test, args, save_prefix)


def test(x_test, y_test, args, path):
    rnn = torch.load(args.snapshot)
    if torch.cuda.is_available():
        rnn.cuda()
    print("Test started")

    rnn.eval()
    corrects_test, avg_loss, iter_test, avg_auc = 0, 0, 0, 0
    tn, fn, tp, fp = 0, 0, 0, 0
    AUROC_list, BCR_list = [], []

    loss_func = nn.CrossEntropyLoss()

    batches_test = data_helpers.batch_iter(
        list(zip(x_test, y_test)), args.batch_size, 1, seq_test)

    print("")
    for batch in batches_test:
        x_test_batch, y_test_batch = zip(*batch)
        x_test_Tensor, y_test_Tensor = data_helpers.tensor4batch(x_test_batch, y_test_batch, args)

        x_test_Variable, y_test_Variable = Variable(x_test_Tensor).cuda(), Variable(y_test_Tensor).cuda()

        logit = rnn(x_test_Variable)

        iter_test += 1

        loss = loss_func(logit, torch.max(y_test_Variable, 1)[1])
        loss_tmp = loss.data.cpu().numpy()[0]
        corrects_data = (torch.max(logit, 1)[1] == torch.max(y_test_Variable, 1)[1]).data

        corrects = corrects_data.sum()
        accuracy = 100.0 * corrects / args.batch_size

        y_pred = torch.max(logit, 1)[1].data.cpu().numpy()
        y_true = torch.max(y_test_Variable, 1)[1].data.cpu().numpy()

        batch_fpr, batch_tpr, batch_thresholds = roc_curve(y_true, y_pred, pos_label=1)
        batch_tn, batch_fp, batch_fn, batch_tp = confusion_matrix(y_true, y_pred).ravel()
        batch_TPR = batch_tp / (batch_tp + batch_fn)
        batch_TNR = batch_tn / (batch_tn + batch_fp)
        batch_AUROC = auc(batch_fpr, batch_tpr)
        batch_BCR = math.sqrt(batch_TPR * batch_TNR)

        tn += batch_tn
        fp += batch_fp
        fn += batch_fn
        tp += batch_tp
        AUROC_list.append(batch_AUROC)
        BCR_list.append(batch_BCR)

        sys.stdout.write(
            '\rTEST||Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{}  TPR: {:.2f}  TNR: {:.2f}  AUROC: {:.2f}  BCR: {:.2f})'.
                format(iter_test,
                       loss.data[0],
                       accuracy,
                       corrects,
                       args.batch_size,
                       batch_TPR,
                       batch_TNR,
                       batch_AUROC,
                       batch_BCR
                       ))
        avg_loss += loss_tmp
        corrects_test += corrects

    size = len(y_test)
    avg_loss = avg_loss / iter_test
    accuracy = 100.0 * corrects_test / size
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    AUROC = sum(AUROC_list[0:len(AUROC_list) - 1]) / len(AUROC_list)
    BCR = sum(BCR_list[0:len(BCR_list) - 1]) / len(BCR_list)

    print('\nTEST - loss: {:.6f}  acc: {:.4f}%({}/{}) TPR: {:.2f}  TNR: {:.2f}  AUROC: {:.2f}  BCR: {:.2f})'.
          format(avg_loss,
                 accuracy,
                 corrects_test,
                 size,
                 TPR,
                 TNR,
                 AUROC,
                 BCR
                 ))

    indicators = ['accuracy', 'TPR', 'TNR', 'AUROC', 'BCR']
    result_list = [accuracy, TPR, TNR, AUROC, BCR]

    import pandas as pd
    results = pd.DataFrame(result_list, columns=['{}_{}'.format(args.model_name, args.data_name)], index=indicators)
    save_path = '{}_{}.csv'.format(path, args.data_name)
    results.to_csv(save_path)


train_step(x_train, y_train, x_dev, y_dev, x_test, y_test, rnn, args)

'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

x = Variable(torch.randn(10, 20, 30)).cuda()
lens = range(1,11)

x = pack_padded_sequence(x, lens[::-1], batch_first=True)

lstm = nn.LSTM(30, 50, batch_first=True).cuda()
h0 = Variable(torch.zeros(1, 10, 50)).cuda()
c0 = Variable(torch.zeros(1, 10, 50)).cuda()

packed_h, (packed_h_t, packed_c_t) = lstm(x, (h0, c0))
h, _ = pad_packed_sequence(packed_h)
print h.size() # Size 20 x 10 x 50 instead of 10 x 20 x 50
'''