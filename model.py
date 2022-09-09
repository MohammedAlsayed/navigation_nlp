# IMPLEMENT YOUR MODEL CLASS HERE
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
torch.device("cpu")

class NavigationLSTM(nn.Module):

    def __init__(self, params):
        super(NavigationLSTM, self).__init__()
        if params["glove"]:
            print("using glove embeddings")
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(params['pre_embeddings']).float(), freeze=False)
        else:
            print("using regular embeddings")
            self.embedding = nn.Embedding(params['vocab_size'], params['embedding_dim'], padding_idx=0)

        self.lstm = nn.LSTM(params['seq_len']*params['embedding_dim'],
                            params['lstm_hidden_dim'],
                            num_layers = params['lstm_layers'],
                            dropout=params['dropout'],
                            batch_first=True)

        # the fully connected layer transforms the output to give the final output layer
        self.fc = nn.Linear(params['lstm_hidden_dim'], params['linear_output_dim'])

        self.linear = nn.Linear(params['linear_output_dim'], params['action_outputs'])
        self.linear = nn.Linear(params['linear_output_dim'], params['target_outputs'])


    def forward(self, s):
        embeds = self.embedding(s) # out dim: vocab_size x seq_len x embedding_dim
        lstm_out, _ = self.lstm(embeds.view(len(s), 1, -1))  # in dim: seq_len*embedding_dim x lstm_hidden_dim
                                                             # out dim: batch_size x 1 x lstm_hidden_dim
                                                             # rehape: batch_size x lstm_hidden_dim
                                                             # one output for every sentence

        lstm_out = lstm_out.contiguous()
        fc = self.fc(lstm_out.view(len(s), -1))

        action_space = self.linear(fc)
        target_space = self.linear(fc)

        return action_space, target_space
        