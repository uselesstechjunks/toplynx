def mha_par_batched(Q,K,V,mask):
    """
    args:
        Q: [b,h,n,k]
        K: [b,h,m,k]
        V: [b,h,m,v]
        mask: [n,m]
    returns:
        O: [b,h,n,v]
    """
    logits = torch.einsum('bhnk,bhmk->bhnm', Q, K)
    attn_mask = mask == 0
    logits.masked_fill_(attn_mask, float('-inf'))
    weights = F.softmax(logits, dim=-1)
    O = torch.einsum('bhnm,bhmv->bhnv', weights, V)
    return O

class MaskedMultiHeadAttentionParallelBatched(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MaskedMultiHeadAttentionParallelBatched, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, X, M, mask):
        Q = torch.einsum('bnd,hdk->bhnk', X, self.Wq)
        K = torch.einsum('bmd,hdk->bhmk', M, self.Wk)
        V = torch.einsum('bmd,hdv->bhmv', M, self.Wv)
        O = mha_par_batched(Q,K,V,mask)
        Y = torch.einsum('bhnv,hvd->bnd', O, self.Wo)
        return Y
