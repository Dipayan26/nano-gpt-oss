
import torch

'''
“Take every element from x (index i) and multiply it with every element from y (index j), and store the result in position (i, j) of the output tensor.”
'''
a = torch.tensor([[1,2],[3,4]])
a.shape
# trace
aa= torch.einsum('ii', a)

# diagonal
torch.einsum('ii->i', torch.randn(4, 4))

# outer product
x = torch.tensor([1, 2, 3,4,5])
y = torch.tensor([10, 20, 30])
torch.einsum('i,j->ij', x, y)



'''>>> x
tensor([1, 2, 3, 4, 5])
>>>
>>> y
tensor([10, 20, 30])
>>>
>>> torch.einsum('i,j->ij', x, y)
tensor([[ 10,  20,  30],
        [ 20,  40,  60],
        [ 30,  60,  90],
        [ 40,  80, 120],
        [ 50, 100, 150]])

'''
# batch matrix multiplication
As = torch.randn(3, 2, 5)
Bs = torch.randn(3, 5, 4)
torch.einsum('bij,bjk->bik', As, Bs)
torch.einsum('abc,acg->abg', As, Bs)
torch.einsum('abc,def->dbf', As, Bs).shape
torch.einsum('abc,def->', As, Bs)

# with sublist format and ellipsis
torch.einsum(As, [..., 0, 1], Bs, [..., 1, 2], [..., 0, 2])

# batch permute
A = torch.randn(2, 3, 4, 5)
torch.einsum('...ij->...ji', A).shape

# equivalent to torch.nn.functional.bilinear
A = torch.randn(3, 5, 4)
l = torch.randn(2, 5)
r = torch.randn(2, 4)
torch.einsum('bn,anm,bm->ba', l, A, r)





