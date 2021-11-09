from collections import Counter
import os
import re
from string import punctuation
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from LSTM import LSTM
import matplotlib.pyplot as plt
import time
import csv
from sklearn.model_selection import train_test_split

descriptions = []
tokenized_descriptions = []
normalized_salaries = []
words_train = set()
max_sentence_size = 0 

counter = 4000

with open('./data/Train_rev1.csv', newline='', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        if(row[0]!='Id'):
            descriptions.append(row[2].lower())
            tokens = re.split('\s+',row[2].lower())
            tokenized_descriptions.append(tokens)
            words_train.update(tokens)
            normalized_salaries.append(int(row[10]))
            size = len(tokens)
            if(size > max_sentence_size):
                max_sentence_size = size
            counter -=1
            if(counter <= 0):
                break

csvfile.close()

#converting the words into indices
vocab_to_int_train = {}
counter = 1
for elem in words_train:
    vocab_to_int_train[elem] = counter
    counter += 1

descriptions_int_train = []

#having the unknown word token as the max index
unk_token = len(set(vocab_to_int_train)) + 1

#converting the sentences into list of word indices
for review in tokenized_descriptions:

    r = [vocab_to_int_train[w] if w in vocab_to_int_train else unk_token for w in review]
    descriptions_int_train.append(r)

#creating the training and testing labels

max_salary = max(normalized_salaries)
min_salary = min(normalized_salaries)
diff = max_salary - min_salary

normalized_salaries = [ (salary - min_salary) / diff for salary in normalized_salaries]
#normalized_salaries = [ ((salary - min_salary) * 2.0 / diff ) - 1 for salary in normalized_salaries]

train_y = np.array(normalized_salaries)

#method to truncate and pad the sentences based on the sequence limit
def pad_features(reviews_int, seq_length):

    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    
    return features

train_x = pad_features(descriptions_int_train,max_sentence_size)

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y,test_size=0.2,random_state=42)

train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 50

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

vocab_size = len(vocab_to_int_train)+2
output_size = 1
embedding_dim = 100
hidden_dim = 100
n_layers = 3

net = LSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, None)

lr=0.001

criterion = nn.MSELoss()
l1_loss = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

epochs = 150
clip=5 # gradient clipping

train_accuracy = []
train_losses = []

#training the model on the dataset for the given number of epochs
for e in range(epochs):

    net.train()

    # initialize hidden state
    h = net.init_hidden(batch_size)

    total_loss = 0

    for inputs, labels in train_loader:

        # Creating new variables for the hidden state
        h = tuple([each.data.cuda() for each in h])
        #h  = torch.stack(list(h), dim=0).cuda()

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.LongTensor).cuda()
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float().cuda())
        loss.backward()
        total_loss += loss.item()

        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        pred = (output.squeeze()).float()

    avg_train_loss = total_loss / len(train_loader)

    train_losses.append(avg_train_loss)

    print("Epoch: {}/{}...".format(e+1, epochs),
                  "Train Loss: {:.6f}...".format(avg_train_loss))


net.eval()

total_error = 0

#evaluating the model on the test set
for test_inputs, test_labels in test_loader:

    h = tuple([each.data.cuda() for each in h])
    
    # get predicted outputs
    test_inputs = test_inputs.type(torch.LongTensor).cuda()
    output, h = net(test_inputs, h)

    pred = (output.squeeze()).float()

    loss = l1_loss(output.squeeze(), test_labels.float().cuda())
    total_error += loss.item()

avg_test_accuracy = total_error * diff / (len(test_loader))
#avg_test_accuracy = total_error * diff / (2.0 * len(test_loader))

print("Test Accuracy: {:.6f}...".format(avg_test_accuracy))
