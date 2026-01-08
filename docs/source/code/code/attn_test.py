"""
Sample output:
--------------------------------------
Attention
tensor([ -3.0437,  -5.3354,  -2.7996,   4.7690,   6.1953,  -3.1872,   2.4339,
         -4.9126,  -0.5149,  -3.6056,   1.6128, -14.4580,  -2.2639,  -2.7896,
         -0.7055,   6.9216], grad_fn=<ViewBackward0>)
MultiHeadAttention
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], grad_fn=<ViewBackward0>)
MultiHeadAttentionSequential
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], grad_fn=<ViewBackward0>)
MaskedMultiHeadAttentionParallel
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], requires_grad=True)
MultiHeadAttentionSequentialBatched
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], requires_grad=True)
MaskedMultiHeadAttentionParallelBatched
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], requires_grad=True)
MultiQueryAttentionSequentialBatched
tensor([ -6.9532,  16.7457, -11.2753,  -6.5196,  22.5019,  13.0422,  -6.6303,
         -6.5617,   2.0252,   2.2950,   1.2324,  15.0855, -20.8858,  17.2391,
         -8.0939,  -1.4088], requires_grad=True)
MaskedMultiQueryAttentionParallelBatched
tensor([ -6.9532,  16.7457, -11.2753,  -6.5196,  22.5018,  13.0422,  -6.6303,
         -6.5617,   2.0252,   2.2950,   1.2324,  15.0855, -20.8858,  17.2391,
         -8.0938,  -1.4088], requires_grad=True)
GroupedQueryAttentionSequentialBatched(g=1)
tensor([ -6.9532,  16.7457, -11.2753,  -6.5196,  22.5019,  13.0422,  -6.6303,
         -6.5617,   2.0252,   2.2950,   1.2324,  15.0855, -20.8858,  17.2391,
         -8.0939,  -1.4088], requires_grad=True)
GroupedQueryAttentionSequentialBatched(g=2)
tensor([-18.4005,  11.4970,   5.9840,   9.8845, -15.3181,   1.3615,  -2.5959,
         17.3029, -11.3590,  25.8750, -14.3187,  -3.3374,   2.2135, -13.3058,
         -1.9368, -15.8990], requires_grad=True)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def attn(q,K,V):
    """
    Args:
        q: [k]
        K: [m,k]
        V: [m,v]
    Returns:
        y: [v]
    """
    logits = torch.einsum('k,mk->m',q,K)
    weights = F.softmax(logits, dim=0)
    y = torch.einsum('m,mv->v', weights, V)
    return y

def mha(q,K,V):
    """
    Args:
        q: [h,k]
        K: [h,m,k]
        V: [h,m,v]
    Returns:
        o: [h,v]
    """
    logits = torch.einsum('hk,hmk->hm',q,K)
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('hm,hmv->hv', weights, V)
    return o

def mha_par(Q,K,V,mask):
    """
    Args:
        Q: [h,n,k]
        K: [h,m,k]
        V: [h,m,v]
        mask: [n,m]
    Returns:
        O: [h,n,v]
    """
    logits = torch.einsum('hnk,hmk->hnm',Q,K)
    attn_mask = mask == 0
    logits.masked_fill_(attn_mask, float('-inf'))
    weights = F.softmax(logits, dim=-1)
    O = torch.einsum('hnm,hmv->hnv', weights, V)
    return O

def mha_batched(q,K,V):
    """
    Args:
        q: [b,h,k]
        K: [b,h,m,k]
        V: [b,h,m,v]
    Returns:
        O: [b,h,v]
    """
    logits = torch.einsum('bhk,bhmk->bhm',q,K)
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('bhm,bhmv->bhv', weights, V)
    return o

def gqa_batched(q,K,V):
    """
    Args:
        q: [b,h,k]
        K: [b,g,m,k]
        V: [b,g,m,v]
    Returns:
        O: [b,h,v]
    """
    # TODO: fixit
    h = q.shape[-2]
    g = K.shape[-3]
    r = int(h/g)
    K = torch.cat([K]*r,dim=-3) # back to dim
    V = torch.cat([V]*r,dim=-3) # back to dim
    o = mha_batched(q,K,V)
    return o

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

def mha_par_batched(Q,K,V,mask):
    """
    Args:
        Q: [b,h,n,k]
        K: [b,h,m,k]
        V: [b,h,m,v]
        mask: [n,m]
    Returns:
        O: [b,h,n,v]
    """
    logits = torch.einsum('bhnk,bhmk->bhnm', Q, K)
    attn_mask = mask == 0
    logits.masked_fill_(attn_mask, float('-inf'))
    weights = F.softmax(logits, dim=-1)
    O = torch.einsum('bhnm,bhmv->bhnv', weights, V)
    return O

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

class Attention(torch.nn.Module):
    def __init__(self, d, k, v):
        super(Attention, self).__init__()
        self.Wq = nn.Parameter(torch.randn(d,k))
        self.Wk = nn.Parameter(torch.randn(d,k))
        self.Wv = nn.Parameter(torch.randn(d,v))

    def forward(self, x, M):
        q = torch.einsum('d,dk->k', x, self.Wq)
        K = torch.einsum('md,dk->mk', M, self.Wk)
        V = torch.einsum('md,dv->mv', M, self.Wv)
        y = attn(q,K,V)
        return y

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiHeadAttention, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, x, M):
        q = torch.einsum('d,hdk->hk', x, self.Wq)
        K = torch.einsum('md,hdk->hmk', M, self.Wk)
        V = torch.einsum('md,hdv->hmv', M, self.Wv)
        o = mha(q, K, V)
        y = torch.einsum('hv,hvd->d', o, self.Wo)
        return y

class MaskedMultiHeadAttentionParallel(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MaskedMultiHeadAttentionParallel, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, X, M, mask):
        Q = torch.einsum('nd,hdk->hnk', X, self.Wq)
        K = torch.einsum('md,hdk->hmk', M, self.Wk)
        V = torch.einsum('md,hdv->hmv', M, self.Wv)
        O = mha_par(Q, K, V, mask)
        Y = torch.einsum('hnv,hvd->nd', O, self.Wo)
        return Y

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

class MultiHeadAttentionSequential(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiHeadAttentionSequential, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, x, prev_K, prev_V):
        """
        Args:
            x: [d]
            prev_K: [h,m,k]
            prev_V: [h,m,v]
        Returns:
            y: [d]
            K: [h,m+1,k]
            V: [h,m+1,v]
        """
        q = torch.einsum('d,hdk->hk', x, self.Wq)
        K = torch.cat((prev_K, torch.einsum('d,hdk->hk', x, self.Wk).unsqueeze(-2)), dim=-2)
        V = torch.cat((prev_V, torch.einsum('d,hdv->hv', x, self.Wv).unsqueeze(-2)), dim=-2)
        o = mha(q, K, V)
        y = torch.einsum('hv,hvd->d', o, self.Wo)
        return y, K, V

class MultiHeadAttentionSequentialBatched(torch.nn.Module):
    def __init__(self, h, d, k, v):
        super(MultiHeadAttentionSequentialBatched, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(h,d,k))
        self.Wv = nn.Parameter(torch.randn(h,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, x, prev_K, prev_V):
        """
        Args:
            x: [b,d]
            prev_K: [b,h,m,k]
            prev_V: [b,h,m,v]
        Returns:
            y: [b,d]
            K: [b,h,m+1,k]
            V: [b,h,m+1,v]
        """
        q = torch.einsum('bd,hdk->bhk', x, self.Wq)
        K = torch.cat((prev_K, torch.einsum('bd,hdk->bhk', x, self.Wk).unsqueeze(-2)), dim=-2)
        V = torch.cat((prev_V, torch.einsum('bd,hdv->bhv', x, self.Wv).unsqueeze(-2)), dim=-2)
        o = mha_batched(q, K, V)
        y = torch.einsum('bhv,hvd->bd', o, self.Wo)
        return y, K, V

class GroupedQueryAttentionSequentialBatched(torch.nn.Module):
    def __init__(self, h, g, d, k, v):
        super(GroupedQueryAttentionSequentialBatched, self).__init__()
        self.Wq = nn.Parameter(torch.randn(h,d,k))
        self.Wk = nn.Parameter(torch.randn(g,d,k))
        self.Wv = nn.Parameter(torch.randn(g,d,v))
        self.Wo = nn.Parameter(torch.randn(h,v,d))

    def forward(self, x, prev_K, prev_V):
        """
        Args:
            x: [b,d]
            prev_K: [b,g,m,k]
            prev_V: [b,g,m,v]
        Returns:
            y: [b,d]
            K: [b,m+1,k]
            V: [b,m+1,v]
        """
        q = torch.einsum('bd,hdk->bhk', x, self.Wq)
        K = torch.cat((prev_K, torch.einsum('bd,gdk->bgk', x, self.Wk).unsqueeze(-2)), dim=-2)
        V = torch.cat((prev_V, torch.einsum('bd,gdv->bgv', x, self.Wv).unsqueeze(-2)), dim=-2)
        o = gqa_batched(q, K, V)
        y = torch.einsum('bhv,hvd->bd', o, self.Wo)
        return y, K, V

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

def test_attn(Attention, d, k, M):
    model = Attention(d,k,d)
    y = model(M[0],M)
    print(f'Attention\n{y}')

def test_mha(MultiHeadAttention, d, k, v, h, M):
    torch.manual_seed(42)
    model = MultiHeadAttention(h,d,k,v)
    y = model(M[-1],M)
    with torch.no_grad():
        print(f'MultiHeadAttention\n{y}')

def test_mha_seq(MultiHeadAttentionSequential, d, k, v, h, M):
    torch.manual_seed(42)
    model = MultiHeadAttentionSequential(h,d,k,v)
    prev_K = torch.FloatTensor(h,0,k)
    prev_V = torch.FloatTensor(h,0,v)
    y = None

    for x in M:
        y, prev_K, prev_V = model(x, prev_K, prev_V)
    
    with torch.no_grad():
        print(f'MultiHeadAttentionSequential\n{y}')

def test_mha_par(MaskedMultiHeadAttentionParallel, d, k, v, h, M, mask):
    torch.manual_seed(42)
    model = MaskedMultiHeadAttentionParallel(h,d,k,v)
    Y = model(M,M,mask)
    with torch.no_grad():
        print(f'MaskedMultiHeadAttentionParallel\n{Y[-1]}')

def test_mha_par_batched(MaskedMultiHeadAttentionParallelBatched, d, k, v, h, M, mask):
    torch.manual_seed(42)
    model = MaskedMultiHeadAttentionParallelBatched(h,d,k,v)
    X = M.unsqueeze(0)
    Y = model(X,X,mask)
    with torch.no_grad():
        Y = Y.squeeze(0)
        print(f'MaskedMultiHeadAttentionParallelBatched\n{Y[-1]}')

def test_mha_seq_batched(MultiHeadAttentionSequentialBatched, d, k, v, h, M):
    torch.manual_seed(42)
    model = MultiHeadAttentionSequentialBatched(h,d,k,v)
    prev_K = torch.FloatTensor(1,h,0,k)
    prev_V = torch.FloatTensor(1,h,0,v)
    y = None

    for x in M:
        y, prev_K, prev_V = model(x.unsqueeze(0), prev_K, prev_V)

    with torch.no_grad():
        y = y.squeeze(0)
        print(f'MultiHeadAttentionSequentialBatched\n{y}')

def test_mqa_par_batched(MaskedMultiQueryAttentionParallelBatched, d, k, v, h, M, mask):
    torch.manual_seed(42)
    model = MaskedMultiQueryAttentionParallelBatched(h,d,k,v)
    X = M.unsqueeze(0)
    Y = model(X,X,mask)
    with torch.no_grad():
        Y = Y.squeeze(0)
        print(f'MaskedMultiQueryAttentionParallelBatched\n{Y[-1]}')

def test_mqa_seq_batched(MultiQueryAttentionSequentialBatched, d, k, v, h, M):
    torch.manual_seed(42)
    model = MultiQueryAttentionSequentialBatched(h,d,k,v)
    prev_K = torch.FloatTensor(1,0,k)
    prev_V = torch.FloatTensor(1,0,v)
    y = None

    for x in M:
        y, prev_K, prev_V = model(x.unsqueeze(0), prev_K, prev_V)

    with torch.no_grad():
        y = y.squeeze(0)
        print(f'MultiQueryAttentionSequentialBatched\n{y}')

def test_gqa1_seq_batched(GroupedQueryAttentionSequentialBatched, d, k, v, h, M):
    torch.manual_seed(42)
    g = 1
    model = GroupedQueryAttentionSequentialBatched(h,g,d,k,v)
    prev_K = torch.FloatTensor(1,g,0,k)
    prev_V = torch.FloatTensor(1,g,0,v)
    y = None

    for x in M:
        y, prev_K, prev_V = model(x.unsqueeze(0), prev_K, prev_V)

    with torch.no_grad():
        y = y.squeeze(0)
        print(f'GroupedQueryAttentionSequentialBatched(g={g})\n{y}')

def test_gqah_seq_batched(GroupedQueryAttentionSequentialBatched, d, k, v, h, M):
    torch.manual_seed(42)
    g = h
    model = GroupedQueryAttentionSequentialBatched(h,g,d,k,v)
    prev_K = torch.FloatTensor(1,g,0,k)
    prev_V = torch.FloatTensor(1,g,0,v)
    y = None

    for x in M:
        y, prev_K, prev_V = model(x.unsqueeze(0), prev_K, prev_V)

    with torch.no_grad():
        y = y.squeeze(0)
        print(f'GroupedQueryAttentionSequentialBatched(g={g})\n{y}')

if __name__ == '__main__':    
    m = 10
    d = 16
    k = 8
    v = 8
    h = 2

    M = torch.randn((m,d))
    # triangular mask mimicing the decoder
    # same code can be reused for encoder with all bits on
    mask = torch.tril(torch.ones(M.shape[0],M.shape[0]))

    #########################################
    # SHA
    #########################################
    test_attn(Attention, d, k, M)

    #########################################
    # MHA
    #########################################
    test_mha(MultiHeadAttention, d, k, v, h, M)
    test_mha_seq(MultiHeadAttentionSequential, d, k, v, h, M)
    test_mha_par(MaskedMultiHeadAttentionParallel, d, k, v, h, M, mask)    
    test_mha_seq_batched(MultiHeadAttentionSequentialBatched, d, k, v, h, M)
    test_mha_par_batched(MaskedMultiHeadAttentionParallelBatched, d, k, v, h, M, mask)

    #########################################
    # MQA
    #########################################    
    test_mqa_seq_batched(MultiQueryAttentionSequentialBatched, d, k, v, h, M)
    test_mqa_par_batched(MaskedMultiQueryAttentionParallelBatched, d, k, v, h, M, mask)

    #########################################
    # GQA
    #########################################    
    test_gqa1_seq_batched(GroupedQueryAttentionSequentialBatched, d, k, v, h, M)
    test_gqah_seq_batched(GroupedQueryAttentionSequentialBatched, d, k, v, h, M)
