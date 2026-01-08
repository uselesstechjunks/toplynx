import torch
import torch.nn.functional as F

"""
Rule of thumb:
------------------------------------------------------------------------
(a) dimensions that appear in the output would appear in the outer-loop.
    we'll fill in for these dimensions element-wise.
(b) dimensions that appear in both the inputs (common dimensions) 
    are multiplied element-wise.
(c) dimensions that appear in the inputs but not on the output are summed over.
    this means for dimensions that satisfy both (b) and (c) are first multiplied
    and then summer over.
"""

def test_tensorprod():
    torch.manual_seed(42)
    X = torch.randn((4,5))
    Y = torch.randn((5,3))
    
    Expected = torch.einsum('ij,jk->ijk', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # output dimension
    for i in torch.arange(X.shape[-2]):
        # output dimension
        for j in torch.arange(X.shape[-1]):
        # output dimension
            for k in torch.arange(Y.shape[-1]):
                # loop through common dimension j for element-wise product
                Actual[i,j,k] = X[i,j] * Y[j,k]

    assert(torch.all(torch.isclose(Expected, Actual)))

def test_matmul():
    torch.manual_seed(42)
    X = torch.randn((4,5))
    Y = torch.randn((5,3))
    
    Expected = torch.einsum('ij,jk->ik', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # output dimension
    for i in torch.arange(X.shape[-2]):
        # output dimension
        for k in torch.arange(Y.shape[-1]):
            # loop through common dimension j for element-wise product
            Actual[i,k] = torch.dot(X[i,:], Y[:,k])

    assert(torch.all(torch.isclose(Expected, Actual)))

def test_matmul_reduce_k():
    torch.manual_seed(42)
    X = torch.randn((4,5))
    Y = torch.randn((5,3))
    
    Expected = torch.einsum('ij,jk->i', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # since k is free for Y, we sum over k and cache it
    sum_k_Y = torch.sum(Y,dim=-1)

    # output dimension
    for i in torch.arange(X.shape[-2]):        
        # loop through common dimension j for element-wise product
        Actual[i] = torch.dot(X[i,:], sum_k_Y)

    assert(torch.all(torch.isclose(Expected, Actual)))

def test_matmul_reduce_i():
    torch.manual_seed(42)
    X = torch.randn((4,5))
    Y = torch.randn((5,3))
    
    Expected = torch.einsum('ij,jk->k', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # since i is free for X, we sum over i and cache it
    sum_i_X = torch.sum(X,dim=-2)

    # output dimension
    for k in torch.arange(Y.shape[-1]):        
        # loop through common dimension j for element-wise product
        Actual[k] = torch.dot(sum_i_X, Y[:,k])

    assert(torch.all(torch.isclose(Expected, Actual)))

def test_matmul_reduce_j():
    torch.manual_seed(42)
    X = torch.randn((4,5))
    Y = torch.randn((5,3))
    
    Expected = torch.einsum('ij,jk->j', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # since i is free for X, we sum over i and cache it
    sum_i_X = torch.sum(X,dim=-2)
    # since k is free for Y, we sum over k and cache it
    sum_k_Y = torch.sum(Y,dim=-1)

    # output dimension
    for j in torch.arange(X.shape[-1]):        
        # loop through common dimension j for element-wise product
        Actual[j] = sum_i_X[j] * sum_k_Y[j]

    assert(torch.all(torch.isclose(Expected, Actual)))

def test_matmul_reduce_all():
    torch.manual_seed(42)
    X = torch.randn((4,5))
    Y = torch.randn((5,3))
    
    Expected = torch.einsum('ij,jk->', X, Y)

    # einsum impl
    Actual = torch.zeros_like(Expected)

    # since i is free for X, we sum over i and cache it
    sum_i_X = torch.sum(X,dim=-2)
    # since k is free for Y, we sum over k and cache it
    sum_k_Y = torch.sum(Y,dim=-1)

    # output dimension
    Actual = torch.dot(sum_i_X, sum_k_Y)

    assert(torch.all(torch.isclose(Expected, Actual)))

###################################################
# Test with multi-head attention code
###################################################
def mha_par_batched(Q,K,V):
    """
    args:
        Q: [b,h,n,k]
        K: [b,h,m,k]
        V: [b,h,m,v]
    returns:
        O: [b,h,n,v]
    """
    logits = torch.einsum('bhnk,bhmk->bhnm', Q, K)
    weights = F.softmax(logits, dim=-1)
    O = torch.einsum('bhnm,bhmv->bhnv', weights, V)
    return logits, O

def mha_par_batched_impl(Q,K,V):
    _b = Q.shape[0]
    _h = Q.shape[1]
    _n = Q.shape[2]
    _m = K.shape[2]
    _v = V.shape[-1]
    logits = torch.zeros(_b,_h,_n,_m)
    for b in torch.arange(_b):
        for h in torch.arange(_h):
            for n in torch.arange(_n):
                for m in torch.arange(_m):
                    logits[b,h,n,m] = torch.dot(Q[b,h,n,:], K[b,h,m,:])
    weights = F.softmax(logits, dim=-1)
    O = torch.zeros(_b,_h,_n,_v)
    for b in torch.arange(_b):
        for h in torch.arange(_h):
            for n in torch.arange(_n):
                for v in torch.arange(_v):
                    O[b,h,n,v] = torch.dot(weights[b,h,n,:], V[b,h,:,v])
    return logits, O

def test_mha_par_batched():
    torch.manual_seed(42)
    b = 5
    h = 2
    n = m = 4
    k = v = 4
    Q = torch.randn((b,h,n,k))
    K = torch.randn((b,h,m,k))
    V = torch.randn((b,h,m,v))
    logits_expected, O_expected = mha_par_batched(Q,K,V)
    logits_actual, O_actual = mha_par_batched_impl(Q,K,V)

    assert(torch.all(torch.isclose(logits_expected, logits_actual)))
    assert(torch.all(torch.isclose(O_expected, O_actual)))

if __name__ == '__main__':
    test_tensorprod()
    test_matmul()
    test_matmul_reduce_k()
    test_matmul_reduce_i()
    test_matmul_reduce_j()
    test_matmul_reduce_all()
    test_mha_par_batched()
