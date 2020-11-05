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

class CNN(nn.Module):
	def __init__(self, args):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(2, 64, 5, 1)
		self.conv2 = nn.Conv2d(64, 32, 5, 1)
		#Expected 4-dimensional input for 4-dimensional weight [32, 1, 5, 5], but got 3-dimensional input of size [64, 320, 100] instead
		self.max_pool = nn.MaxPool2d(2, 2)
		self.dropout = nn.Dropout(args.dropout_rate)
		self.fc1 = nn.Linear((32 * 37 * 22), 1000)
		# ((args.pad_seq_len - 4) / 2 - 4) / 2
		self.fc2 = nn.Linear(1000, 100)
		self.fc3 = nn.Linear(100, 2)
	
	def forward(self, x):
		'''
		向前传播
		'''
		x = self.conv1(x)
		#print(x.size())
		x = F.relu(x)
		x = self.max_pool(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.max_pool(x)
		x = torch.flatten(x, 1)
		x = self.dropout(x)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		x = F.log_softmax(x, dim=1)

		return x

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

			print("Tokens to Vectors...")
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
		return torch.Tensor([self.Data['f_content_vector'][idx], self.Data['b_content_vector'][idx]]), \
			   torch.Tensor(self.Data['2_labels'][idx])

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
		optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.8)
		optimizer.zero_grad()
		output = model(vectors)	# 待修改

		loss_func = nn.MSELoss()
		loss = loss_func(output, label)
		loss.backward()

		optimizer.step()
		# 打印进度条
		
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * args.batch_size + len(vectors), len(train_loader.dataset),
			(100. * batch_idx * args.batch_size + len(vectors)) / len(train_loader.dataset), loss.item()))
	print("\n\n\n")

def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss_func = nn.MSELoss(reduction='sum')
			test_loss += loss_func(output, target)  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
			target = target.argmax(dim=1, keepdim=False)
			for idx in range(len(pred)):
				if pred[idx] == target[idx]:
					correct += 1
				else:
					pass

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def train_CNN(args):
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	#使用 gpu
	# Load word2vec model
	print("Loading data...")
	word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file)

	# Load sentences, labels, and training parameters
	print("Data processing...")
	train_data = TextData(args, args.train_file, word2idx, embedding_matrix)
	test_data = TextData(args, args.test_file, word2idx, embedding_matrix)
	train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True, num_workers=1)
	test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False, num_workers=1)
	
	model = CNN(args).to(device)
	#print(model)

	for epoch in range(1, args.epochs + 1):
		train(args, model, train_loader, device, epoch)
		test(model, device, test_loader)
	
	#torch.save(model.state_dict(), "../data/TextCNN.pt")


if __name__ == '__main__':
	args = parser.parameter_parser()	# add parser by using argparse module
	train_CNN(args)
	x = print("Press any key to continue...")