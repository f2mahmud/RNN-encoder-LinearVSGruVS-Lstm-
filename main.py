from __future__ import unicode_literals, print_function, division
import random

import glob
import string
import unicodedata
from io import open

import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import math

import matplotlib.pyplot as plt


def findFiles(path): return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []
category_lines_values = {}


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    x = (int)(len(lines) * .7)
    category_lines[category] = lines[:x]
    category_lines_values[category] = lines[x:]

n_categories = len(all_categories)


def letterToIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.LSTMCell(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hc):
        hidden, context = self.i2h(input, hc)
        output = self.out(hidden)
        output = self.softmax(output)
        return output, (hidden, context)

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size)), Variable(torch.zeros(1, self.hidden_size))


rnn_lstm = RNN_LSTM(n_letters, n_hidden, n_categories)


class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_GRU, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.GRUCell(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = self.i2h(input, hidden)
        output = self.out(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


rnn_gru = RNN_GRU(n_letters, n_hidden, n_categories)


def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor


criterion = nn.NLLLoss()


def train(category_tensor, line_tensor, model):
    hidden = model.initHidden()

    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]


n_iters = 100000
print_every = 5000
plot_every = 1000


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def evaluate(model):
    total_loss = 0
    count = 0
    for category in all_categories:
        for line in category_lines_values[category]:
            category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
            line_tensor = Variable(lineToTensor(line))
            hidden = model.initHidden()
            for i in range(line_tensor.size()[0]):
                output, hidden = model(line_tensor[i], hidden)

            total_loss += len(line_tensor) * criterion(output, category_tensor).data
            count += 1

    avg_loss = total_loss / count
    return avg_loss


start = time.time()
model_types = [rnn, rnn_gru, rnn_lstm]
model_labels = ['linear_rnn(given)', 'gru', 'lstm']

losses = []
plots = []

for i in range(len(model_types)):

    # Keep track of losses for plotting
    current_loss = 0
    value_loss = []

    model_type = model_types[i]
    model_label = model_labels[i]

    print('Now running RNN Type-------------> %s <----------------\n' % model_label)
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor, model_type)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            value_loss.append(evaluate(model_type))
            current_loss = 0

    losses.append(value_loss)

plt.figure()

for i in range(len(model_types)):
    plt.plot(losses[i], label=model_labels[i])


plt.xlabel("Number of iterations (thousand)")
plt.ylabel("test(validation) negative log likelihood")
plt.legend()
plt.title('Linear vs GRU vs LSTM RNNs')
plt.savefig('Linear_v_GRU_v_LSTM.png')
