import torch
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict

class MyLSTMCell(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()        
        self.c = torch.zeros(out_dim) # long term memory
        self.h = torch.zeros(out_dim) # short term memory

        # process input gates
        self._ii = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self._hi = torch.nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)
        self._inorm = torch.nn.LayerNorm(normalized_shape=out_dim)
        self._ai = torch.nn.Sigmoid()

        # process forget gates
        self._if = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self._hf = torch.nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)
        self._fnorm = torch.nn.LayerNorm(normalized_shape=out_dim)
        self._af = torch.nn.Sigmoid()

        # process gating gates
        self._ig = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self._hg = torch.nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)
        self._gnorm = torch.nn.LayerNorm(normalized_shape=out_dim)
        self._ag = torch.nn.Tanh()

        # process output gates
        self._io = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self._ho = torch.nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)
        self._onorm = torch.nn.LayerNorm(normalized_shape=out_dim)
        self._ao = torch.nn.Sigmoid()

        self._ch = torch.nn.Tanh()

    def detach_hidden(self):
        self.h.detach_()
        self.c.detach_()

    def forward(self, x):
        f = self._hf(self.h) + self._if(x)
        f = self._fnorm(f)
        f = self._af(f)

        i = self._hi(self.h) + self._ii(x)
        i = self._inorm(f)
        i = self._ai(i)

        g = self._hg(self.h) + self._ig(x)
        g = self._gnorm(g)
        g = self._ag(g)

        o = self._ho(self.h) + self._io(x)
        o = self._onorm(o)
        o = self._ao(o)

        # long term memory: forget old stuff a little bit, add new stuff a little bit
        # note that c is just multiplied by the forget gate and added by the input - no weights being multiplied here
        self.c = torch.mul(self.c, f) + torch.mul(i, g)

        # short term memory: remember the output primarily, also add a little bit from the new long-term memory that formed
        self.h = torch.mul(o, self._ch(self.c))

        return self.h

class MyDeepLSTM(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=in_dim, embedding_dim=hidden_dim, max_norm=True)
        self.rnn = torch.nn.Sequential(OrderedDict([
                        ('rnn1', MyLSTMCell(in_dim=hidden_dim, out_dim=hidden_dim)),
                        ('rnn2', MyLSTMCell(in_dim=hidden_dim, out_dim=hidden_dim)),
                        ('rnn3', MyLSTMCell(in_dim=hidden_dim, out_dim=hidden_dim)),
                        ('rnn4', MyLSTMCell(in_dim=hidden_dim, out_dim=hidden_dim))
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
        h = self.rnn(h)
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

    model = MyDeepLSTM(in_dim=vocab_size, hidden_dim=hidden_dim, out_dim=vocab_size)
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
