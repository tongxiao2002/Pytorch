import os
import sys
import time
import logging
import argparse
import gensim
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from gensim.models import KeyedVectors
from collections import OrderedDict

sys.path.append('../')

from utils import data_helpers as dh
from utils import param_parser as parser

logging.getLogger('Pytorch').disabled = True

class RNN(nn.Module):
    def __init__(self, args, device):
        super(RNN, self).__init__()
        self.num_layers = 6
        self.num_directions = 1
        self.hidden_size = args.fc_dim
        self.output_size = args.pad_seq_len * args.batch_size * self.num_directions * self.hidden_size
        self.lstm = nn.LSTM(input_size=args.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=False,
                            batch_first=True)
        self.fc = nn.Linear(self.output_size * 2, 2)
        self.hidden = self.init_hidden()        #初始化隐藏层
        self.out = self.init_hidden()

    def init_hidden(self):
        return torch.randn(self.num_layers * self.num_directions, args.batch_size, self.hidden_size).to(device)

    def forward(self, x):
        '''
        向前传播
        '''
        out, (middle_hidden, middle_out) = self.lstm(x, (self.hidden, self.out))
        print("out_shape: {0}\thidden_shape: {1}\t".format(out.size(), middle_hidden.size()))
        # ? 这里为什么结果大小比预计的大了 1 倍 ?, 因为在 getitem() 中我将 front 和 behind 连到了一起, 因此是 seq_len 增大了一倍
        shape_out = torch.reshape(out, (1, self.output_size * 2))
        fc_out = self.fc(shape_out)
        final_out = F.log_softmax(fc_out, dim=1)

        return final_out

class TextData(torch.utils.data.Dataset):
    def __init__(self, args, input_file, word2idx, embedding_matrix):	# read input_file
        if not input_file.endswith('.json'):
            raise IOError("[Error] The research record is not a json file. "
                          "Please preprocess the research record into the json file.")

        def _token_to_index(x: list):
            result = []
            for item in x:
                if item not in word2idx.keys():
                    result.append(word2idx['_UNK'])		#	unknown
                else:
                    word_idx = word2idx[item]
                    result.append(word_idx)		# get value
            return result
        self.length = 0
        self.Data = dict()
        with open(input_file) as fin:
            self.Data['f_id'] = []
            self.Data['b_id'] = []
            self.Data['f_content_index'] = []
            self.Data['b_content_index'] = []
            self.Data['labels'] = []

            print("Tokens to Vectors ...")
            for eachline in fin:	# type(eachline) = str
                record = json.loads(eachline)	# type(record) = dict
                f_id = record['front_testid']
                b_id = record['behind_testid']
                f_content = record['front_features']
                b_content = record['behind_features']
                labels = record['label']

                self.Data['f_id'].append(f_id)
                self.Data['b_id'].append(b_id)
                self.Data['f_content_index'].append(_token_to_index(f_content))
                self.Data['b_content_index'].append(_token_to_index(b_content))
                self.Data['labels'].append(labels)
                self.length += 1

            # 下面进行截断/补全
            for idx, content in enumerate(self.Data['f_content_index']):
                if len(content) < args.pad_seq_len:
                    while len(content) < args.pad_seq_len:
                        content.append(int(0))	# 补 0
                elif len(content) > args.pad_seq_len:	# 截断
                    while len(content) > args.pad_seq_len:
                        content.pop()
                else:
                    pass

            for idx, content in enumerate(self.Data['b_content_index']):
                if len(content) < args.pad_seq_len:
                    while len(content) < args.pad_seq_len:
                        content.append(int(0))	# 补 0
                elif len(content) > args.pad_seq_len:	# 截断
                    while len(content) > args.pad_seq_len:
                        content.pop()
                else:
                    pass

            # 下面将通过 embedding_matrix 将索引列表转化为向量列表, 将每个索引替换为一个 1xN 向量
            def _index_to_vector(x, embedding_matrix):	# x 为一行单词对应的索引列表
                result = []
                for index in x:
                    result.append(embedding_matrix[index])
                return result

            self.Data['f_content_vector'] = []
            self.Data['b_content_vector'] = []
            self.Data['2_labels'] = []
            for content in self.Data['f_content_index']:
                self.Data['f_content_vector'].append(_index_to_vector(content, embedding_matrix))	# 新建一个列表, 用于存向量列表
            for content in self.Data['b_content_index']:
                self.Data['b_content_vector'].append(_index_to_vector(content, embedding_matrix))	# 新建一个列表, 用于存向量列表
            for label in self.Data['labels']:	#将 labels 转化为一个二元组
                result = [0., 0.]
                result[label] = 1.
                self.Data['2_labels'].append(result)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.Tensor(self.Data['f_content_vector'][idx] + self.Data['b_content_vector'][idx]), \
               torch.Tensor(self.Data['2_labels'][idx])	#合并两个文本列表, 作为神经网络输入

def load_word2vec_matrix(word2vec_file):	#导入训练数据
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    wv = KeyedVectors.load(word2vec_file, mmap='r')

    word2idx = OrderedDict({"_UNK": 0})	# 有序字典
    embedding_size = wv.vector_size
    for k, v in wv.vocab.items():
        word2idx[k] = v.index + 1
    vocab_size = len(word2idx)

    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in word2idx.items():
        if key == "_UNK":
            embedding_matrix[value] = [0. for _ in range(embedding_size)]
        else:
            embedding_matrix[value] = wv[key]
    return word2idx, embedding_matrix

def train(args, model, train_loader, device, epoch):
    model.train()	# ?
    for batch_idx, (vectors, label) in enumerate(train_loader):
        vectors = vectors.to(device)
        label = label.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
        optimizer.zero_grad()
        output = model(vectors)	# 待修改

        loss_func = nn.MSELoss()
        loss = loss_func(output, label)
        loss.backward(retain_graph=True)

        optimizer.step()
        # 打印进度条

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(vectors), len(train_loader.dataset),	#为什么 len(train_loader.dataset) 恒为 128？
                   100. * batch_idx / len(train_loader), loss.item()))
    print("\n\n\n")

def train_RNN(args, device):

    
    # Load word2vec model
    print("Loading data...")
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    # Load sentences, labels, and training parameters
    print("Data processing...")
    train_data = TextData(args, args.train_file, word2idx, embedding_matrix)
    #	test_data = TextData(args, args.test_file, word2idx, embedding_matrix)
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True, num_workers=1)
    #	test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False, num_workers=1)

    model = RNN(args, device).to(device)
    print(model)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, device, epoch)

#torch.save(model.state_dict(), "../data/TextCNN.pt")


if __name__ == '__main__':
    args = parser.parameter_parser()	# add parser by using argparse module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	#使用 gpu
    train_RNN(args, device)
    x = print("Press any key to continue...")