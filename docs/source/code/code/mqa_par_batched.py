def mqa_par_batched(Q,K,V,mask):
    """
    Args:
        Q: [b,h,n,k]
        K: [b,m,k]
        V: [b,m,v]
        mask: [n,m]
    Returns:
        O: [b,h,n,v]
    """
    logits = torch.einsum('bhnk,bmk->bhnm', Q, K)
    attn_mask = mask == 0
    logits.masked_fill_(attn_mask, float('-inf'))
    weights = F.softmax(logits, dim=-1)
    O = torch.einsum('bhnm,bmv->bhnv', weights, V)
    return O

class MaskedMultiQueryAttentionParallelBatched(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MaskedMultiQueryAttentionParallelBatched, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(d,k))
        self.Wv = nn.Parameter(torch.randn(d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, X, M, mask):
        """
        Args:
            X: [b,n,d]
            M: [b,m,d]
            mask: [n,m]
        Returns:
            Y: [b,n,d]
        """
        Q = torch.einsum('bnd,hdk->bhnk', X, self.Wq)
        K = torch.einsum('bmd,dk->bmk', M, self.Wk)
        V = torch.einsum('bmd,dv->bmv', M, self.Wv)
        O = mqa_par_batched(Q,K,V,mask)
        Y = torch.einsum('bhnv,hvd->bnd', O, self.Wo)
        return Y
