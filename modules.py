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
            dropout=0,
            n_classes=2,
            verbose=0,
    ):
        super().__init__()
        self.dropout = dropout
        self.verbose = verbose
        self.fc = nn.Linear(input_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dp = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, n_classes)

    def forward(self, X):
        if self.verbose: print(X.shape)
        out = F.relu(self.fc(X))
        if self.verbose: print(out.shape)
        # out = self.bn(out)
        # if self.verbose: print(out.shape)
        # out = self.dense(F.dropout(out, p=self.dropout))
        out = self.dense(self.dp(out))
        if self.verbose: print(out.shape)
        return out


class BERT(nn.Module):
    def __init__(
            self,
            n_classes=2,
            name_path='bert-base-uncased'
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(name_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.bert.config.to_dict()['hidden_size'], n_classes)

    def forward(self, X):
        context, bert_masks = X[0].to(torch.long), X[1].to(torch.long)
        with torch.no_grad():
            last_hidden_states = self.bert(context, attention_mask=bert_masks)
        last_hidden_states = last_hidden_states[0][:, 0, :]  # use cls token
        out = self.fc(last_hidden_states)
        return out


class CRNN1D(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_size=128,
            dropout=0,
            n_classes=2,
    ):
        super().__init__()
        # (batch, channels, seq_len)
        k, s = 2, 1
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, stride=s),
            # nn.BatchNorm1d(1),
            nn.ReLU(),
            # nn.MaxPool1d(k),
        )
        dim1 = ((input_dim - k)//s+1)  # //k
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, stride=s),
            # nn.BatchNorm1d(1),
            nn.ReLU(),
            # nn.MaxPool1d(k),
        )
        dim2 = ((dim1 - k)//s+1)  # //k
        # (batch, seq_len, features)
        self.rnn = nn.GRU(1, 1, num_layers=1, batch_first=True)
        self.dp = nn.Dropout(dropout)
        self.dense = nn.Linear(dim2, n_classes)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = x.permute(0, 2, 1)
        # _, x = self.rnn(x) # get hidden (GRU)
        # x = x[-1]  # take output of last RNN layer
        x, _ = self.rnn(x)  # get output
        x = x.mean(-1)
        out = self.dense(self.dp(x))
        return out


class CRNN2D(nn.Module):
    def __init__(
            self,
            seq_len,
            nfilt=26,
            dropout=0,
            n_classes=2,
    ):
        super().__init__()
        # (batch, channels, seq_len) # (N, 26, 3000) # (108, 128, 938)
        k, p = 2, 3
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=nfilt, out_channels=32, kernel_size=k+1),
            nn.BatchNorm1d(32),  # , affine=False # to unlearn
            nn.ReLU(),
            nn.MaxPool1d(p),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=k+1),
            nn.BatchNorm1d(16),  # , affine=False # to unlearn
            nn.ReLU(),
            nn.MaxPool1d(p),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=k+1),
            nn.BatchNorm1d(8),  # , affine=False # to unlearn
            nn.ReLU(),
            nn.MaxPool1d(p),
        )
        # (batch, seq_len, features)
        self.rnn = nn.GRU(8, 8, num_layers=1, batch_first=True)
        self.dp = nn.Dropout(dropout)
        self.dense = nn.Linear((((seq_len-k)//p-k)//p-k)//p * 8, n_classes)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # N, C, L = x.size()
        # x = x.reshape(N, C*10, L//10)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.dp(x)
        x = x.permute(0, 2, 1)
        # _, x = self.rnn(x) # gru
        # x = x[-1]  # take output of last RNN layer
        x, _ = self.rnn(x)  # LSTM
        x = x.reshape(x.shape[0], -1)
        out = self.dense(x)
        return out


class RNN(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_size=128,
            dropout=0,
            n_classes=2,
    ):
        super().__init__()
        self.dropout = dropout
        # (batch, seq_len, features)
        self.rnn = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True)
        self.dense = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.mean(1)  # take output of last RNN layer
        drop = F.dropout(x, p=self.dropout)
        out = self.dense(drop)
        return out


class LSTMAttention(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_size=128,
            dropout=0,
            n_classes=2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # self.fc = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.dense = nn.Linear(hidden_size, n_classes)

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return (h0, c0)

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1).squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state).squeeze(2)
        weights = F.softmax(weights)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights.unsqueeze(2)).squeeze(2)

    def forward(self, X):
        # X = F.relu(self.fc(X))
        hidden = self.init_hidden(X.shape[0])
        rnn_out, (h_n, _) = self.lstm(X, hidden)
        # attn_out = F.relu(self.attention(rnn_out, h_n))
        attn_out = self.attention(rnn_out, h_n)
        out = self.dense(attn_out)
        return out


class LSTMAttention1(nn.Module):
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
        self.fc1 = nn.Linear(input_dim1, hidden_size//2)
        self.fc2 = nn.Linear(input_dim2, hidden_size//2)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True,
                            dropout=dropout, bidirectional=False)
        self.dense = nn.Linear(hidden_size, n_classes)

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
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
        X1 = F.relu(self.fc1(X[0]))
        X2 = F.relu(self.fc2(X[1]))
        X = torch.cat([X1, X2], -1)
        X = X.unsqueeze(1)
        hidden = self.init_hidden(X.shape[0])
        rnn_out, (h_n, _) = self.lstm(X, hidden)
        # attn_out = F.relu(self.attention(rnn_out, h_n))
        attn_out = self.attention(rnn_out, h_n)
        out = self.dense(attn_out)
        return out


class LSTMAttention2(nn.Module):
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
        self.lstm1 = nn.LSTM(input_dim1, hidden_size//2, batch_first=True,
                             dropout=dropout, bidirectional=False)
        self.lstm2 = nn.LSTM(input_dim2, hidden_size//2, batch_first=True,
                             dropout=dropout, bidirectional=False)
        self.dense = nn.Linear(hidden_size//2, n_classes)

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size//2))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size//2))
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


class BiLSTMAttention(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_size=128,
            dropout=0,
            n_classes=2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(input_dim, hidden_size//2, batch_first=True,
                              dropout=dropout, bidirectional=True)
        self.dense = nn.Linear(hidden_size, n_classes)

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2, batch_size, self.hidden_size//2))
        c0 = Variable(torch.zeros(2, batch_size, self.hidden_size//2))
        return (h0, c0)

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    def forward(self, X):
        hidden = self.init_hidden(X.shape[0])
        rnn_out, (h_n, _) = self.bilstm(X, hidden)
        attn_out = self.attention(rnn_out, h_n)
        out = self.dense(attn_out)
        return out
