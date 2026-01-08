def mha_batched(q,K,V):
    """
    args:
        q: [b,h,k]
        K: [b,h,m,k]
        V: [b,h,m,v]
    returns:
        O: [b,h,v]
    """
    logits = torch.einsum('bhk,bhmk->bhm',q,K)
    weights = F.softmax(logits, dim=-1)
    O = torch.einsum('bhm,bhmv->bhv', weights, V)
    return O

class MultiHeadAttentionSequentialBatched(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiHeadAttentionSequentialBatched, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    '''
    args:
        x: [b,d]
        prev_K: [b,h,m,k]
        prev_V: [b,h,m,v]
    returns:
        y: [b,d]
        K: [b,h,m+1,k]
        V: [b,h,m+1,v]
    '''
    def forward(self, x, prev_K, prev_V):
        Q = torch.einsum('bd,hdk->bhk', x, self.Wq)
        K = torch.cat((prev_K, torch.einsum('bd,hdk->bhk', x, self.Wk).unsqueeze(-2)), dim=-2)
        V = torch.cat((prev_V, torch.einsum('bd,hdv->bhv', x, self.Wv).unsqueeze(-2)), dim=-2)
        O = mha_batched(Q, K, V)
        Y = torch.einsum('bhv,hvd->bd', O, self.Wo)
        return Y, K, V
