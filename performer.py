

# import torch
# from performer_pytorch import CrossAttention

# attn = CrossAttention(
#     dim = 512,
#     heads = 8,
#     causal = True,
# ).cuda()

# x = torch.randn(1, 1024, 512).cuda()
# context = torch.randn(1, 512, 512).cuda()

# attn(x, context = context) # (1, 1024, 512)



import torch
from performer_pytorch import FastAttention

# queries / keys / values with heads already split and transposed to first dimension
# 8 heads, dimension of head is 64, sequence length of 512
q = torch.randn(1, 8, 512, 64)
k = torch.randn(1, 8, 512, 64)
v = torch.randn(1, 8, 512, 64)

attn_fn = FastAttention(
    dim_heads = 64,
    nb_features = 256,
    causal = True
).cuda()

out = attn_fn(q, k, v) # (1, 8, 512, 64)
# now merge heads and combine outputs with Wo