'''
NOTE: This is a very spart representation of comments.
If you would like to see a fully commented out explanation of all code

Please see the model code found in 'models/LSTM/Training.py'.
'''
# Import necessary modules needed for the project
from re import S
import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score
import random
import matplotlib as mpl
import os
import gc
from torch.utils.tensorboard import SummaryWriter
from datetime import date
mpl.rcParams['figure.dpi'] = 300


# writer is just a summary writer of the training
writer = SummaryWriter()
writer = SummaryWriter("training_tensorboard")
writer = SummaryWriter(comment="400_3000")

x = np.arange(3000)
x = np.reshape(x,(-1,1))
y = x 
X, x_test, Y, y_test = train_test_split( x, y, test_size=0.04, random_state=10)
x_train, x_valid, y_train, y_valid = train_test_split( X, Y, test_size=0.04, random_state=100)
n_samples_train = np.shape(x_train)[0] 
n_samples_test = np.shape(x_test)[0]
n_samples_valid = np.shape(x_valid)[0]
print('Number of train samples:', n_samples_train)
print('Number of valid samples:', n_samples_valid)
print('Number of test samples:', n_samples_test)

# saving the test sample set for future use
np.save('x_test', x_test)
np.save('x_train', x_train)
np.save('x_valid', x_valid)



# Hyper-parameter setting
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



# Dataset preparation
class BetaDataset(Dataset) :
    def __init__(self,x,y, n_samples) :
        self.x = x
        self.y = y 
        self.n_samples = n_samples
        
        
    def __getitem__(self,index) :
        return self.x[index], self.y[index]

    def __len__(self) :    
        return self.n_samples      

train_dataset = BetaDataset(x_train,y_train,n_samples_train)
test_dataset = BetaDataset(x_test,y_test,n_samples_test)
valid_dataset = BetaDataset(x_valid,y_valid,n_samples_valid)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=1)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=1)



# Shortened LSTM Model (not as deep, only 1 bnn, 2 biderectional LSTM, 2 nn to out
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, seq_len,num_classes=1):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.seq_len = seq_len

        self.bnn = nn.Linear(input_size, 1024) #go from 1308 to 1024

        self.lstm1 = nn.LSTM(1024, hidden_size1, num_layers, batch_first=True, bidirectional=True, dropout=0.5) #go from 1024 to 512

        self.lstm2 = nn.LSTM(2*hidden_size1, 64, num_layers, batch_first=True, bidirectional=True, dropout=0.5) #go from 1024 to 128 (bidirectional so 2x output 

        self.nn1 = nn.Linear(2*hidden_size2,hidden_size2) #go from 128 to 64
        self.nn2 = nn.Linear(hidden_size2, 1) #go from 64 to 1 for output
    
        self.relu = nn.ReLU()

        
    def forward(self, x, array_lengths):
        inital_seq_len = x.size(1)
        x = Variable(x.float()).to(device)

        x = torch.reshape(x, (x.size(0)*x.size(1), x.size(2)))

        out = self.bnn(x)
        out = self.relu(out)

        out = torch.reshape(out, (-1, inital_seq_len, out.size(1)))

        pack = nn.utils.rnn.pack_padded_sequence(out, array_lengths, batch_first=True, enforce_sorted=False)
        h0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size1).to(device))
        c0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size1).to(device))
        h1 = Variable(torch.zeros(2*self.num_layers, self.hidden_size1, self.hidden_size2).to(device))
        c1 = Variable(torch.zeros(2*self.num_layers, self.hidden_size1, self.hidden_size2).to(device))
        
        out, _ = self.lstm1(pack, (h0,c0))
        del(h0)
        del(c0)
        out, _ = self.lstm2(out, (h1,c1))
        del(h1)
        del(c1)
        gc.collect()
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        this_batch_len = unpacked.size(1)
        out = unpacked
        out = torch.reshape(out, (out.size(0)*out.size(1), out.size(2)))

        out = self.nn1(out)
        out = self.relu(out)
        out = self.nn2(out)
        
        out = torch.reshape(out, (-1, this_batch_len, 1))

        return out


#Actual running of the model
input_size = 1308
hidden_size1 = 512
hidden_size2 = 64
num_layers = 1
init_lr = 0.001
num_epochs = 400
seq_len = 500


model = RNN(input_size, hidden_size1, hidden_size2,  num_layers, seq_len).to(device)
print(model)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters()))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

all_x = np.load('/home/tjdean2/research_data/CHBE594/class_project/dataset/x_full_3000_fixed' + '.npy', allow_pickle=True)
all_y = np.load('/home/tjdean2/research_data/CHBE594/class_project/dataset/y_3000' + '.npy', allow_pickle=True)

for epoch in range(num_epochs):
  loop = 0
  avg_loss = 0
  for i, (parameters, no_req) in enumerate(train_loader):
    input_x = torch.from_numpy(all_x[parameters,:,:]).to(device)
    output_y = torch.from_numpy(all_y[parameters,:,:]).to(device)
    input_x = torch.reshape(input_x, (input_x.size(0), input_x.size(2), input_x.size(3)))
    output_y = torch.reshape(output_y, (output_y.size(0), output_y.size(2), output_y.size(3)))
    
    array_lengths = input_x[:,0,1308]
    array_lengths = array_lengths.int()
    array_lengths = array_lengths.tolist()
    outputs = model(input_x[:,:,0:1308], array_lengths)
    outputs = torch.reshape(outputs, (-1,int(max(array_lengths))) )
    output_y = torch.reshape(output_y[:,0:int(max(array_lengths)), 0], (-1,int(max(array_lengths))))
    weight_matrix = torch.ones(output_y.size()).to(device)

    loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True).to(device)
    for k in range(input_x.size()[0]):
      prot_len = int(input_x[k,0,1308].item())
      vector_mul = torch.ones((1, int(max(array_lengths)))).to(device)
      zero_vector = torch.zeros((1, int(max(array_lengths)))).to(device)
      vector_mul[prot_len:int(max(array_lengths))] = 0.00
      weighted_loss = (torch.mul(torch.reshape(outputs[k,:], (1,-1)), vector_mul)-  torch.reshape(output_y[k,:], (1,-1))).to(device)
      weighted_loss = torch.mul( weighted_loss, weight_matrix[k,:]).float()
      loss = loss + criterion( weighted_loss, zero_vector)

    loss = loss/input_x.size()[0]

    avg_loss = (avg_loss*i + loss.item())/(i+1)


    
    del(input_x)
    del(output_y)
    del(vector_mul)
    del(outputs)
    gc.collect()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  
  if (epoch%1 == 0):
    torch.save(model, f'/home/tjdean2/research_data/CHBE594/class_project/models/LSTM_smaller/epoch_{epoch+1}.pth')
    writer.add_scalar("Loss per epoch/train", avg_loss, epoch)

    with torch.no_grad():
        valid_loss = 0.0
        for j, (param_valid, no_req) in enumerate(valid_loader):
            input_x = torch.from_numpy(all_x[param_valid,:,:]).to(device)
            output_y = torch.from_numpy(all_y[param_valid,:,:]).to(device)
            input_x = torch.reshape(input_x, (input_x.size(0), input_x.size(2), input_x.size(3)))
            output_y = torch.reshape(output_y, (output_y.size(0), output_y.size(2), output_y.size(3)))
            
            array_lengths = input_x[:,0,1308]
            array_lengths = array_lengths.int()
            array_lengths = array_lengths.tolist()
            outputs = model(input_x[:,:,0:1308], array_lengths)
            outputs = torch.reshape(outputs, (-1,int(max(array_lengths))) )
            output_y = torch.reshape(output_y[:,0:int(max(array_lengths)), 0], (-1,int(max(array_lengths))))
            weight_matrix = torch.ones(output_y.size()).to(device)

            loss = torch.tensor(0.0, dtype=torch.float32).to(device)
            for s in range(input_x.size()[0]):
                prot_len = int(input_x[s,0,1308].item())
                vector_mul = torch.ones((1, int(max(array_lengths)))).to(device)
                zero_vector = torch.zeros((1, int(max(array_lengths)))).to(device)
                vector_mul[prot_len:int(max(array_lengths))] = 0.00
                weighted_loss = (torch.mul(torch.reshape(outputs[s,:], (1,-1)), vector_mul)-  torch.reshape(output_y[s,:], (1,-1))).to(device)
                weighted_loss = torch.mul( weighted_loss, weight_matrix[s,:]).float()
                loss = loss + criterion( weighted_loss, zero_vector)
            loss = loss/input_x.size()[0]         

            valid_loss = (valid_loss*j + loss.item())/(j+1)
        writer.add_scalar("Loss per epoch/Valid", valid_loss, epoch)

  print('done epoch:', epoch+1)

