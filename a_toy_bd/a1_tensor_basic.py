import sys
import torch

def attention_forward(x,w_q,w_k,w_v,prefix_len = 4):
    # x's shape is 256*16
    # w_q,w_k,w_v's shape is 16*16
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    score = q @ k.T
    # score is 256*256



    # 这里设置mask 将右上角全部设置为—无穷大 这样softmax就是0
    allowed = torch.ones_like(score,dtype=torch.bool).tril()
    #这里进行block difussion 改造
    # 1 0 0 0 0 0
    # 1 1 0 0 0 0
    # 1 1 1 0 0 0
    # 1 1 1 1 0 0
    # 1 1 1 1 1 1
    # 1 1 1 1 1 1
    allowed[prefix_len:,:] = True
    print(allowed)
    score_masked = score.masked_fill(~allowed,float('-inf'))


    attention = (score_masked / (k.shape[-1]**0.5)).softmax(dim = -1)
    # attention is 256*256

    context = attention @ v
    # context = 256*16
    print(context)
    print(context.shape)
    return context


if __name__ == '__main__':

    X = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    W_Q = torch.tensor([
        [1.0, 0.0],
        [0.0, 2.0]
    ])

    W_K = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0]
    ],dtype=torch.float32)

    W_V = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ],dtype=torch.float32)

    torch.manual_seed(0)

    X = torch.randn(6, 16)
    W_Q = torch.randn(16, 16)
    W_K = torch.randn(16, 16)
    W_V = torch.randn(16, 16)

    context = attention_forward(X,W_Q,W_K,W_V)

    
