import torch
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict

class MyRNNCell(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.h = torch.zeros(out_dim)
        self.ih = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.hh = torch.nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)
        self.norm = torch.nn.LayerNorm(normalized_shape=out_dim)
        self.relu = torch.nn.ReLU()

    def detach_hidden(self):
        self.h.detach_()

    def forward(self, x):
        # recurrence here
        a = self.hh(self.h) + self.ih(x)
        a = self.norm(a + x)
        self.h = self.relu(a)
        return self.h

class MyDeepRNN(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=in_dim, embedding_dim=hidden_dim, max_norm=True)
        self.rnn = torch.nn.Sequential(OrderedDict([
                        ('rnn1', MyRNNCell(in_dim=hidden_dim, out_dim=hidden_dim)),
                        ('rnn2', MyRNNCell(in_dim=hidden_dim, out_dim=hidden_dim)),
                        ('rnn3', MyRNNCell(in_dim=hidden_dim, out_dim=hidden_dim)),
                        ('rnn4', MyRNNCell(in_dim=hidden_dim, out_dim=hidden_dim))
                    ]))
        self.ho = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=True)
        self.norm = torch.nn.LayerNorm(normalized_shape=out_dim)

    def detach_hidden(self):
        for unit in self.rnn:
            unit.detach_hidden()

    def display_params(self):
        for name, param in self.named_parameters():
            print(f'{name}={param.shape}')

    def forward(self, x):
        h = self.embedding(x)
        h = self.rnn(h) + h #residual connection
        o = self.ho(h)
        y = self.norm(o)
        return y

if __name__ == '__main__':
    inputs = ['hello, world!\n', 'this is my first rnn demo.\n', 'excited? nope!\n'] # \n=EOS
    x_seq = [torch.LongTensor(list(input.encode('utf-8'))) for input in inputs]
    x_padded = pad_sequence(x_seq, batch_first=False, padding_value=0) #.permute(1,0)

    # since we're only predicting with the last, and we have sequences of different lengths
    # we create multiple minimatches where we pass first K-1 tokens in the sequence and ask it to predict K
    # and we vary K to be a sequence of size 2, to all the way N
    batches = [x_padded[:n+2,:] for n in range(x_padded.shape[0]-1)] # create N(N-1) batches from size N

    vocab_size = 256
    hidden_dim = 8
    num_epochs = 100

    model = MyDeepRNN(in_dim=vocab_size, hidden_dim=hidden_dim, out_dim=vocab_size)
    optim = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(num_epochs):
        for batch in batches:
            model.detach_hidden()

            for input, target in zip(batch[:-1], batch[1:]):
                pred = model(input)

            loss = criterion(pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
