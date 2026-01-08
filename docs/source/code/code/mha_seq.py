def mha(q,K,V):
    """
    args:
        q: [h,k]
        K: [h,m,k]
        V: [h,m,v]
    returns:
        o: [h,v]
    """
    logits = torch.einsum('hk,hmk->hm',q,K)
    weights = F.softmax(logits, dim=1)
    o = torch.einsum('hm,hmv->hv', weights, V)
    return o

class MultiHeadAttentionSequential(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiHeadAttentionSequential, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    '''
    args:
        x: [d]
        prev_K: [h,m,k]
        prev_V: [h,m,v]
    returns:
        y: [d]
        K: [h,m+1,k]
        V: [h,m+1,v]
    '''
    def forward(self, x, prev_K, prev_V):
        q = torch.einsum('d,hdk->hk', x, self.Wq)
        K = torch.cat((prev_K, torch.einsum('d,hdk->hk', x, self.Wk).unsqueeze(-2)), dim=-2)
        V = torch.cat((prev_V, torch.einsum('d,hdv->hv', x, self.Wv).unsqueeze(-2)), dim=-2)
        o = mha(q, K, V)
        y = torch.einsum('hv,hvd->d', o, self.Wo)
        return y, K, V
