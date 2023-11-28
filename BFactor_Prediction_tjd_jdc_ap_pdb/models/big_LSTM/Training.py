'''
NOTE: This class project was inspired by the paper of Pandey et. al.

You can find this paper at:
    Pandey et. al, "B-factor prediction in proteins using a sequence-based deep learning model," Patterns, 2022
    DOI: https://doi.org/10.1016/j.patter.2023.100805
'''

# Import necessary modules needed for the project

from re import S # Allow the dot character to match any charater 'DOTALL flag'
import torch
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader # To load the dataset and split data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import r2_score
import random
import matplotlib as mpl
import os
import gc # Garbage collector to ensure that variables no longer needed can be erased
from torch.utils.tensorboard import SummaryWriter
from datetime import date
mpl.rcParams['figure.dpi'] = 300 # Set preliminary DPI for matplotlib figures


# writer is just a summary writer of the training
writer = SummaryWriter()
writer = SummaryWriter("training_tensorboard")
writer = SummaryWriter(comment="400_3000")

# Due to GPU constraints, our dataset is constrained to 3,000 proteins of padded sequence 500 each for 1.5 million residues total
x = np.arange(3000)
x = np.reshape(x,(-1,1))
y = x 
X, x_test, Y, y_test = train_test_split( x, y, test_size=0.04, random_state=10) # Split the data for an initial 4% for testing
x_train, x_valid, y_train, y_valid = train_test_split( X, Y, test_size=0.04, random_state=100) # Take another 4% for validation during training
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



# hyper-parameters
batch_size = 128 # Performs well with GPU, don't want to push it when not necessary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(device) # Ensure GPU can be found and used by pytorch



# Build the data loader for our dataset
class BetaDataset(Dataset):
    def __init__(self,x,y, n_samples):
        # data loading
        self.x = x
        self.y = y 
        self.n_samples = n_samples
        
        
    def __getitem__(self,index):
        return self.x[index], self.y[index]

    def __len__(self):    
        return self.n_samples      

# Separate the dataset into the three sections for train, validate, and test
train_dataset = BetaDataset(x_train,y_train,n_samples_train)
test_dataset = BetaDataset(x_test,y_test,n_samples_test)
valid_dataset = BetaDataset(x_valid,y_valid,n_samples_valid)

# For the training, we want to shuffle the dataset when loading between epochs so as to avoid order bias
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



# LSTM Model (Build the parts and define the model below)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, seq_len,num_classes=1):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.seq_len = seq_len

        # Start defining the layer types and keep them named differently 
        # bnn[1-4] = (before neural network), lstm1 and bn_lstm1, then nn[1-8] (nn = neural netowrk)
        self.bnn1 = nn.Linear(input_size, 32)
        self.bnn2 = nn.Linear(32,64)
        self.bnn3 = nn.Linear(64,64)
        self.bnn4 = nn.Linear(64,64)

        # Bidirectional LSTM with batch norm 1D, what this means is it gathers dependencies in data
        # forwards to backwards and backwards to forwards at the same time. (the batchnorm increases
        # training speed and stabilization ((but isn't needed))).
        self.lstm1 = nn.LSTM(64, hidden_size1, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.bn_lstm1 = nn.BatchNorm1d(2*hidden_size1,device=device)

        # Now the rest of the basic Linear Neural Network layers
        self.nn1 = nn.Linear(2*hidden_size1, 2*hidden_size1)
        self.nn2 = nn.Linear(2*hidden_size1, 512)
        self.nn3 = nn.Linear(512, 512)
        self.nn4 = nn.Linear(512, 256)
        self.nn5 = nn.Linear(256, 256)
        self.nn6 = nn.Linear(256, 128)
        self.nn7 = nn.Linear(128, 32)
        self.nn8 = nn.Linear(32, 1)

        # Now define the activation function needed (just ReLU outperforms tanh)
        self.relu = nn.ReLU()

        
    # Now we need to define the forward function and return its outputs
    def forward(self, x, array_lengths):
        # First, set the initial length of the sequences
        inital_seq_len = x.size(1)
        x = Variable(x.float()).to(device) #Move x to the GPU

        # Reshape from 3D to 2D multiplying out (0 = proteins, 1 = residues)
        x = torch.reshape(x, (x.size(0)*x.size(1), x.size(2)))

        ## Feed the features into the pre-emptive NN and work it through
        out = self.bnn1(x)
        out = self.relu(out)
        out = self.bnn2(out)
        out = self.relu(out)
        out = self.bnn3(out)
        out = self.relu(out)
        out = self.bnn4(out)
        out = self.relu(out)

        ## Now, reshape the array again to prepare for packing
        out = torch.reshape(out, (-1, inital_seq_len, out.size(1)))

        # Now, we will pack the padded sequences!
        # What this means is that our sequences are padded, but we do not want unneeded calculations, so we pack the array
        # into a jagged array which LSTMs and other RNN models can deal with (We will re-pad after the LSTM layer)
        pack = nn.utils.rnn.pack_padded_sequence(out, array_lengths, batch_first=True, enforce_sorted=False)
        # We want to store the initial hidden and cell (h and c) states of the bidirectional LSTM to look at later
        h0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size1).to(device))
        c0 = Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size1).to(device))
        
        # Forward propagate RNN
        out, _ = self.lstm1(pack, (h0,c0))
        del(h0)
        del(c0)
        gc.collect()
        # Now, we want to unpack the packed array and reshape it to what is needed before feeding back into the linear NN layers
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        this_batch_len = unpacked.size(1)
        out = unpacked
        out = torch.reshape(out, (out.size(0)*out.size(1), out.size(2)))

        # Our final round of linear NN layers leading to the final regression output
        out = self.nn1(out)
        out = self.relu(out)
        out = self.nn2(out)
        out = self.relu(out)
        out = self.nn3(out)
        out = self.relu(out)
        out = self.nn4(out)
        out = self.relu(out)
        out = self.nn5(out)
        out = self.relu(out)
        out = self.nn6(out)
        out = self.relu(out)
        out = self.nn7(out)
        out = self.relu(out)
        out = self.nn8(out)
        
        # Now, reshape it into the original 3D from 2D
        out = torch.reshape(out, (-1, this_batch_len, 1))
        

        return out


#Actual running of the model requires input values
input_size = 1308 # Number of feature variables [1308] is the number of residues in protein without padding
hidden_size1 = 512
hidden_size2 = 64
num_layers = 1
init_lr = 0.001
num_epochs = 400
seq_len = 500


model = RNN(input_size, hidden_size1, hidden_size2,  num_layers, seq_len).to(device)
print(model)
print('Number of trainable parameters:', sum(p.numel() for p in model.parameters()))

# The loss and the optimizer. For our model we are using MSE for the loss and the Adam optimizer with lr above.
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

# Feed in the training data and the y data for the model
all_x = np.load('/home/tjdean2/research_data/CHBE594/class_project/dataset/x_full_3000_fixed' + '.npy', allow_pickle=True)
all_y = np.load('/home/tjdean2/research_data/CHBE594/class_project/dataset/y_3000' + '.npy', allow_pickle=True)

# Now we begin the iterations for each step of the model with training
for epoch in range(num_epochs):
  loop = 0
  avg_loss = 0
  for i, (parameters, no_req) in enumerate(train_loader): # Initiate the data loader to develop the necessary train, test, valid split
    input_x = torch.from_numpy(all_x[parameters,:,:]).to(device)
    output_y = torch.from_numpy(all_y[parameters,:,:]).to(device)
    input_x = torch.reshape(input_x, (input_x.size(0), input_x.size(2), input_x.size(3)))
    output_y = torch.reshape(output_y, (output_y.size(0), output_y.size(2), output_y.size(3)))
    
    # Now initiate the forward pass of the model  
    array_lengths = input_x[:,0,1308]
    array_lengths = array_lengths.int()
    array_lengths = array_lengths.tolist()
    outputs = model(input_x[:,:,0:1308], array_lengths)
    outputs = torch.reshape(outputs, (-1,int(max(array_lengths))) )
    output_y = torch.reshape(output_y[:,0:int(max(array_lengths)), 0], (-1,int(max(array_lengths))))
    weight_matrix = torch.ones(output_y.size()).to(device) # Write out the weights matrix to the GPU

    # Next, iterate to create the loss function for the model
    loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True).to(device)
    for k in range(input_x.size()[0]):
      prot_len = int(input_x[k,0,1308].item())
      vector_mul = torch.ones((1, int(max(array_lengths)))).to(device)
      zero_vector = torch.zeros((1, int(max(array_lengths)))).to(device)
      vector_mul[prot_len:int(max(array_lengths))] = 0.00
      weighted_loss = (torch.mul(torch.reshape(outputs[k,:], (1,-1)), vector_mul)-  torch.reshape(output_y[k,:], (1,-1))).to(device)
      weighted_loss = torch.mul( weighted_loss, weight_matrix[k,:]).float()
      loss = loss + criterion( weighted_loss, zero_vector) # Calculate the loss using the weighted loss, the zero vector, and ensuring
      # that we are only taking into account the protein residues and not the packed 0s at the end for each protein in the dataset.

    # Now, we can take the average loss over the training in the epoch
    loss = loss/input_x.size()[0]
    avg_loss = (avg_loss*i + loss.item())/(i+1)

    # Now, to ensure GPU does not run out of memory, remove temp variables from epoch and run garbage collector
    del(input_x)
    del(output_y)
    del(vector_mul)
    del(outputs)
    gc.collect()

    # Now, we execute the backwards pass to update the parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  # After every epoch, save the weights as a .pth file that can be initiated in testing with the class
  if (epoch%1 == 0):
    torch.save(model, f'/home/tjdean2/research_data/CHBE594/class_project/models/LSTM/epoch_{epoch+1}.pth')
    writer.add_scalar("Loss per epoch/train", avg_loss, epoch)

    # Also between epochs, calculate the validation loss and return it to the TensorBoard for evaluation
    # We use torch.no_grad() to conserve memory as we are preventing autograd from tracking operations
    with torch.no_grad():
        valid_loss = 0.0
        for j, (param_valid, no_req) in enumerate(valid_loader):
            input_x = torch.from_numpy(all_x[param_valid,:,:]).to(device)
            output_y = torch.from_numpy(all_y[param_valid,:,:]).to(device)
            input_x = torch.reshape(input_x, (input_x.size(0), input_x.size(2), input_x.size(3)))
            output_y = torch.reshape(output_y, (output_y.size(0), output_y.size(2), output_y.size(3)))
            
            # The forward pass  
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

