import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import Variable
from torch.nn import init

class MultiLSTM(nn.Module):
    def __init__(self, vocab_size, num_props):
        super(MultiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.rnn_1 = nn.LSTMCell(512, 512)
        self.rnn_2 = nn.LSTMCell(512, 512)
        self.rnn_3 = nn.LSTMCell(512, 512)
        self.linear = nn.Linear(512, vocab_size)
        

        self.ff_output = nn.Sequential(
            nn.Linear(num_props, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

    def forward(self, x, hc, properties):
        device = next(self.parameters()).device
        x = self.embedding(x).to(device)
        props = self.ff_output(properties)
        x = torch.cat((x, props), 1)
        h, c = hc
        h_out = Variable(torch.zeros(h.size())).to(device)
        c_out = Variable(torch.zeros(c.size())).to(device)
        x, _ = h_out[0], c_out[0] = self.rnn_1(x, (h[0].to(device), c[0].to(device)))
        x, _ = h_out[1], c_out[1] = self.rnn_2(x, (h[1].to(device), c[1].to(device)))
        x, _ = h_out[2], c_out[2] = self.rnn_3(x, (h[2].to(device), c[2].to(device)))
        x = self.linear(x)
        hc_out = h_out, c_out
        return x, hc_out


    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h = Variable(torch.zeros(3, batch_size, 512)).to(device)
        c = Variable(torch.zeros(3, batch_size, 512)).to(device)
        hc = h, c
        # initialize the hidden state and the cell state to zeros
        return hc



class RNN():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc, num_props):
        self.rnn = MultiLSTM(voc.vocab_size, num_props=num_props)
        if torch.cuda.is_available():
            # Wrap the model with DataParallel to run on multiple GPUs
            self.rnn = self.rnn.cuda()
        self.voc = voc
        self.nlllike = nn.NLLLoss(reduction='none') # Computes loss per sample, not summing over batch


    def likelihood(self, target, properties):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        batch_size, seq_length = target.size()
        
        # Ensure start_token is on the correct device
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # Specify device here
        start_token[:] = self.voc.vocab['START']
        
        # Ensure target is also moved to the correct device
        target = target.to(device)
        
        # Concatenate and move to device
        x = torch.cat((start_token, target[:, :-1]), 1).to(device)  # Ensure x is on the correct device
        hc = self.rnn.init_hidden(batch_size)  # Ensure properties are on the correct device

        
        total_loss = torch.zeros(batch_size, device=device)

        for step in range(seq_length):
            logits, hc = self.rnn(x[:, step], hc, properties.to(device))
            log_prob = F.log_softmax(logits, dim=1)
            total_loss += self.nlllike(log_prob, target[:, step])

        
        return total_loss

    def sample(self, batch_size, properties, temperature, max_length=150):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_token = torch.zeros(batch_size, device=device).long()
        start_token[:] = self.voc.vocab['START']
        hc = self.rnn.init_hidden(batch_size)
        x = start_token.to(device)

        sequences = []
        finished = torch.zeros(batch_size, device=device).byte()
        total_loss = torch.zeros(batch_size, device=device)



        for step in range(max_length):
            logits, hc = self.rnn(x, hc, properties.to(device)) 
            temp = temperature
            prob = F.softmax(logits / temp, dim=1)
            x = torch.multinomial(prob,1).view(-1)
            sequences.append(x.view(-1, 1))


            log_prob = F.log_softmax(logits, dim=1)
            step_loss = self.nlllike(log_prob, x)
            total_loss += step_loss

            x = Variable(x.data)
            EOS_sampled = (x == self.voc.vocab['END']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, total_loss
