def mqa_batched(q,K,V):
    """
    Args:
        q: [b,h,k]
        k: [b,m,k]
        v: [b,m,v]
    Returns:
        o: [b,h,n,v]
    """
    logits = torch.einsum('bhk,bmk->bhm', q, K)
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('bhm,bmv->bhv', weights, V)
    return o

class MultiQueryAttentionSequentialBatched(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiQueryAttentionSequentialBatched, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(d,k))
        self.Wv = nn.Parameter(torch.randn(d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, x, prev_K, prev_V):
        """
        Args:
            x: [b,d]
            prev_K: [b,m,k]
            prev_V: [b,m,v]
        Returns:
            y: [b,d]
            K: [b,m+1,k]
            V: [b,m+1,v]
        """
        q = torch.einsum('bd,hdk->bhk', x, self.Wq)
        K = torch.cat((prev_K, torch.einsum('bd,dk->bk', x, self.Wk).unsqueeze(-2)), dim=-2)
        V = torch.cat((prev_V, torch.einsum('bd,dv->bv', x, self.Wv).unsqueeze(-2)), dim=-2)
        o = mqa_batched(q, K, V)
        y = torch.einsum('bhv,hvd->bd', o, self.Wo)
        return y, K, V
