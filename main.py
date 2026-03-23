import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Linear(d_model, d_model)

        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)

        self.kv_cache = None

    def attention(self, x: torch.Tensor, mask: torch.Tensor = None, streaming: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        assert mask is None or mask.shape == (B, L, L), f"Wrong size {mask.shape}!"

        # calculate Q, K, V
        # Q, K, V: [B, L, D] -> [B, L, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if streaming:
            if self.kv_cache is None:
                self.kv_cache = [k, v]
            else:
                self.kv_cache[0] = torch.cat([self.kv_cache[0], k], dim=1)
                self.kv_cache[1] = torch.cat([self.kv_cache[1], v], dim=1)
            k, v = self.kv_cache

        # calculate attention score SDPA
        # [B, L, D] @ [B, D, L] -> [B, L, L] (score)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D) # softmax(Q*K^T / scale)
        if mask is not None:
            score = torch.masked_fill(score, mask == False, float('-inf'))
        score = F.softmax(score, dim=-1)

        # [B, L, L] @ [B, L, D] -> [B, L, D]
        res = torch.matmul(score, v)

        # W_O
        res = self.o_proj(res)

        return res
    
    # def attention_w_kv(self, x: torch.Tensor, streaming: bool = False) -> torch.Tensor:
    #     # step L
    #     # self.kv: [B, L - 1, D]
    #     # x: [B, 1, D]
    #     q = self.q_proj(x)
    #     k = self.k_proj(x)
    #     v = self.v_proj(x)
    #     if streaming:
    #         k = torch.cat([self.kv_cache[0], k], dim=1)
    #         v = torch.cat([self.kv_cache[1], v], dim=1)

    #     # calculate attention score SDPA
    #     # [B, L, D] @ [B, D, L] -> [B, L, L] (score)
    #     score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D) # softmax(Q*K^T / scale)
    #     if mask is not None:
    #         score = torch.masked_fill(score, mask == False, float('-inf'))
    #     score = F.softmax(score, dim=-1)

    #     # [B, L, L] @ [B, L, D] -> [B, L, D]
    #     res = torch.matmul(score, v)

    #     # W_O
    #     res = self.o_proj(res)


    #     res = None
    #     # res: [B, 1, D]
    #     return res

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, streaming: bool = False) -> torch.Tensor:
        # x: [B, L, D]
        
        # attention
        hidden = self.pre_norm(x)
        x = x + self.attention(hidden, mask=mask, streaming=streaming)

        # ffn
        hidden = self.post_norm(x)
        x = x + self.ffn(hidden)

        return x

def softmax(x: torch.Tensor) -> torch.Tensor:
    # x: [..., L] -> [..., L]
    # exp(x[..., i] - max()) / sum_i exp(x[..., i] - max())
    
    # softmax [1...L - 1]: + [L]
    # any i: exp(x[i] - max(x[1...L-1])); sum_i exp(x[i...L-1] - max(x[i...L-1]))
    # any i: exp(x[i] - max(x[1...L])); sum_i exp(x[i...L] - max(x[i...L]))

    cum_mx = 
    for i in range(L):
        fenzi *= exp(-(max(x[i], cum_mx) - cum_mx))

    x -= torch.max(x)
    return torch.exp(x) / torch.sum(torch.exp(x), dim=-1)

def main():
    BZ = 4
    D_MODEL = 64
    SEQ_LEN = 10
    SA = SelfAttention(D_MODEL)
    x = torch.randn(BZ, SEQ_LEN, D_MODEL)
    mask = torch.tril(torch.ones(BZ, SEQ_LEN, SEQ_LEN))

    gt = SA(x.clone(), mask=mask)
    print(gt.shape)

    # x = torch.randn(BZ, SEQ_LEN, D_MODEL)
    # print(mask)
    # print(SA(x, mask=mask))

    attn_res = torch.empty(BZ, 0, D_MODEL)
    for i in range(SEQ_LEN):
        # tmp_mask = torch.tril(torch.ones(BZ, i, i))
        tmp_res = SA(x[:, i : i + 1, :].clone(), streaming=True) # [BZ, 1, D]
        attn_res = torch.cat([attn_res, tmp_res], dim=1) # [BZ, (i) + 1, D]
        # [BZ, i, D_MODEL]
        print(attn_res.shape)
    
    print(gt[:, 1, :].mean(), attn_res[:, 1, :].mean())



    print(F.mse_loss(gt, attn_res))


if __name__ == '__main__':
    main()