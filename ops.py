import torch
from torch.nn import functional as F
from torch import nn
import math

def vqvae_encode(x):
    f = torch.randn([x.shape[0], 32, 16, 16])
    return f


def vqvae_decode(x):
    img = torch.randn([x.shape[0], 3, 256, 256])
    return img


def resize(x, size, mode='bicubic'):
    x = F.interpolate(x, size=(size, size), mode=mode)
    return x


def phi_conv(x):
    return x


def mse_loss(x, y):
    loss = F.mse_loss(x, y)
    return loss


# for vqvae
def get_fhat(f, codebook, patch_nums):
    batch_size = f.shape[0]
    codebook_dim = codebook.weight.shape[-1]
    origin_size = patch_nums[-1]

    # init
    f_hat = 0

    for pk in patch_nums:
        fk = resize(f, pk, mode='area').permute(0, 2, 3, 1).view(-1, codebook_dim)  # [batch_size * pk * pk, codebook_dim]

        rk = get_idx_with_codebook(fk, codebook).view([batch_size, pk, pk])  # R.append(rk)

        zk = codebook(rk).permute(0, 3, 1, 2)  # [bs, codebook_dim, pk, pk]
        zk = resize(zk, origin_size, mode='bicubic')
        zk = phi_conv(zk)  # To correct a resize error

        f_hat += zk  # for recon
        f -= zk  # for next token

    return f_hat


def get_idx_with_codebook(rk, codebook):
    distances = F.pairwise_distance(rk.unsqueeze(1), codebook.weight.data.unsqueeze(0))  # [N, 1, C], [1, V, C] -> [N, V]
    idx = torch.argmin(distances, dim=1)
    return idx


# for var
def get_token_maps(R, codebook, patch_nums):
    batch_size = R[0].shape[0]
    codebook_dim = codebook.weight.shape[-1]
    patch_count = len(patch_nums)

    # init
    patch_size = patch_nums[0]
    origin_size = patch_nums[-1]
    f_hat = 0
    token_maps = []

    # 2. Embed each resolution of the image.
    for i in range(patch_count - 1):
        # Generate a feature map of the current scale through embedding.
        rk = R[i]
        zk = codebook(rk).transpose(1, 2).view([batch_size, codebook_dim, patch_size, patch_size])
        zk = resize(zk, size=origin_size, mode='bicubic')
        zk = phi_conv(zk)

        f_hat += zk

        # next patch
        next_patch_size = patch_nums[i + 1]
        patch_size = next_patch_size

        token = resize(f_hat, size=next_patch_size, mode='area').view([batch_size, codebook_dim, -1]).transpose(1, 2) # [bs, p_i+1 * p_i+1, codebook_dim]
        token_maps.append(token)

    token_maps = torch.cat(token_maps, dim=1)  # no start token

    return token_maps


def get_idx_GT(img, patch_nums):
    # input
    batch_size = img.shape[0]

    # hyper-parameter
    codebook_num = 4096
    codebook_dim = 32
    codebook = nn.Embedding(codebook_num, codebook_dim)

    # encode
    latent = vqvae_encode(img)  # [1, 32, 16, 16]
    origin_size = latent.shape[-1]

    # init
    f = latent
    R = []

    for pk in patch_nums:
        fk = resize(f, pk, mode='area').permute(0, 2, 3, 1).view(-1, codebook_dim)  # [batch_size * pk * pk, codebook_dim]

        rk = get_idx_with_codebook(fk, codebook)
        rk_hw = rk.view([batch_size, pk, pk])

        zk = codebook(rk_hw).permute(0, 3, 1, 2)  # [bs, codebook_dim, pk, pk]
        zk = resize(zk, origin_size, mode='bicubic')
        zk = phi_conv(zk)  # To correct a resize error

        f -= zk  # for next token

        idx_N = rk.view([batch_size, pk * pk])
        R.append(idx_N)

    return R, codebook


def token_linear(x, embed_dim):
    return torch.randn([x.shape[0], x.shape[1], embed_dim])


def token_block(x, class_emb):
    return x


def get_logit(x, class_emb):
    return torch.randn([x.shape[0], x.shape[1], 4096])


def get_next_autoregressive_input(stage_index, patch_nums, latent, token):
    # latent = f_hat
    # token = zk

    origin_size = patch_nums[-1]

    if stage_index != len(patch_nums) - 1:
        token = resize(token, size=origin_size, mode='bicubic')
        h = phi_conv(token)  # conv after upsample

        latent = latent + h

        next_token = resize(latent, size=patch_nums[stage_index + 1], mode='area')  # 이건 확대인데 area네...

        return latent, next_token
    else:
        h = phi_conv(token)
        latent = latent + h

        return latent, latent


def sampling_top_k_p(logits, top_k, top_p, num_samples):
    B, L, V = logits.shape

    if top_k > 0:  # Selecting only the top k values among the V values.
        values, _ = logits.topk(top_k, dim=-1)
        min_values = values[:, :, -1].unsqueeze(-1)  # 최소 값이 모든 k 중 가장 작은 값
        logits = torch.where(logits < min_values, torch.full_like(logits, -torch.inf), logits)

    if top_p > 0:  # Remove cumulative product ≥ p, sample significant logits.
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, :, 1:] = sorted_indices_to_remove[:, :, :-1].clone()  # Since the last value is always 1...
        sorted_indices_to_remove[:, :, 0] = False  # Keep the first index.
        indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill_(indices_to_remove, -torch.inf)

    probs = F.softmax(logits, dim=-1)
    samples_idx = torch.multinomial(probs.view(-1, V), num_samples=num_samples, replacement=True).view(B, L, num_samples)
    return samples_idx


def cross_entropy(x, y):
    loss = nn.CrossEntropyLoss(reduction='none')(x.view(-1, x.shape[-1]), y.view(-1)).view(x.shape[0], -1)
    return loss


def init_embed(init_embed, embed_dim):
    init_std = math.sqrt(1 / embed_dim / 3)

    # 클래스 임베딩 or level embedding
    embed = nn.Embedding(init_embed, embed_dim)
    nn.init.trunc_normal_(embed.weight.data, mean=0, std=init_std)

    return embed


def init_position_embed(patch_nums, embed_dim):
    init_std = math.sqrt(1 / embed_dim / 3)

    # start pos embed (for class emb)
    first_patch_size = patch_nums[0] ** 2
    pos_start_embed = torch.empty(1, first_patch_size, embed_dim)  # [1, 1, 1024]
    nn.init.trunc_normal_(pos_start_embed, mean=0, std=init_std)
    pos_start_embed = nn.Parameter(pos_start_embed)

    # all pos embed
    pos_embed = []
    for pn in patch_nums:
        patch_size = pn ** 2
        pe = torch.empty(1, patch_size, embed_dim)
        nn.init.trunc_normal_(pe, mean=0, std=init_std)
        pos_embed.append(pe)
    pos_embed = nn.Parameter(torch.cat(pos_embed, dim=1))  # [1, total_patches, embed_dim]

    return pos_start_embed, pos_embed