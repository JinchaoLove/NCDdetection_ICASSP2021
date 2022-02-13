import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer

class MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_size=32,
            dropout=0.5,
            n_classes=2,
    ):
        super().__init__()
        self.dropout = dropout
        self.fc = nn.Linear(input_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dp = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, n_classes)

    def forward(self, X):
        out = F.relu(self.fc(X))
        out = self.dense(self.dp(out))
        return out


class BERT(nn.Module):
    def __init__(self, n_classes=2, name_path='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(name_path)
        self.freeze()
        self.fc = nn.Linear(self.bert.config.to_dict()['hidden_size'], n_classes)

    def freeze(self, training=False):
        for param in self.bert.parameters():
            param.requires_grad = training

    def forward(self, X):
        context, bert_masks = X[0].to(torch.long), X[1].to(torch.long)
        with torch.no_grad():
            last_hidden_states = self.bert(context, attention_mask=bert_masks)
        last_hidden_states = last_hidden_states[0][:, 0, :]  # use the <cls> token
        out = self.fc(last_hidden_states)
        return out

class LSTMAttention(nn.Module):
    def __init__(
            self,
            input_dim1,
            input_dim2,
            hidden_size=128,
            dropout=0,
            n_classes=2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_dim1, hidden_size // 2, batch_first=True,
                             dropout=dropout, bidirectional=False)
        self.lstm2 = nn.LSTM(input_dim2, hidden_size // 2, batch_first=True,
                             dropout=dropout, bidirectional=False)
        self.dense = nn.Linear(hidden_size // 2, n_classes)

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size // 2))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size // 2))
        return (h0, c0)

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    def forward(self, X):
        X0 = X[0].unsqueeze(1)
        X1 = X[1].unsqueeze(1)
        hidden0 = self.init_hidden(X0.shape[0])
        rnn_out0, (h_n0, _) = self.lstm1(X0, hidden0)
        hidden1 = self.init_hidden(X1.shape[0])
        rnn_out1, (h_n1, _) = self.lstm2(X1, hidden1)
        # rnn_out, h_n = torch.cat([rnn_out0, rnn_out1], -1), torch.cat([h_n0, h_n1], -1)
        attn_out0 = self.attention(rnn_out0, h_n0)
        attn_out1 = self.attention(rnn_out1, h_n1)
        # attn_out = torch.cat([attn_out0, attn_out1], -1)
        attn_out = torch.mul(attn_out0, attn_out1)
        out = self.dense(attn_out)
        return out
