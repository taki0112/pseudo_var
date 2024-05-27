from ops import *

img = torch.rand([1, 3, 256, 256]) * 2 - 1 # [-1, 1]
class_label = torch.tensor([7])
class_num = 1000  # ImageNet
patch_nums = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]

def vqvae_training(img, patch_nums):
    # hyper-parameter
    codebook_num = 4096
    codebook_dim = 32
    beta = 0.25
    codebook = nn.Embedding(codebook_num, codebook_dim)

    # encode
    latent = vqvae_encode(img) # [bs, 32, 16, 16]
    f_hat = get_fhat(f=latent, codebook=codebook, patch_nums=patch_nums)

    # recon
    recon_img = vqvae_decode(f_hat)

    # loss
    perceptual_loss = 0
    discriminative_loss = 0
    loss = beta * mse_loss(latent, f_hat.detach()) + mse_loss(latent.detach(), f_hat) + mse_loss(img, recon_img) + perceptual_loss + discriminative_loss

    return loss

def var_training(img, patch_nums, class_label):
    depth = 16
    embed_dim = 1024 # depth * 64
    L = sum(pn * pn for pn in patch_nums)

    """ GT Index w/ pretrained vqvae """
    R, codebook = get_idx_GT(img, patch_nums)
    gt_idx = torch.cat(R, dim=1) # [bs, L]

    """ define embedding """
    # codebook
    class_codebook = init_embed(init_embed=class_num+1, embed_dim=embed_dim)
    level_codebook = init_embed(init_embed=(len(patch_nums)), embed_dim=embed_dim)

    # position
    pos_start_emb, pos_emb = init_position_embed(patch_nums=patch_nums, embed_dim=embed_dim)

    # level
    level_seq = [torch.full((pn * pn,), i) for i, pn in enumerate(patch_nums)]
    level_seq = torch.cat(level_seq).unsqueeze(0)

    # embedding
    level_emb = level_codebook(level_seq) + pos_emb # [bs, L, 1024]
    class_emb = class_codebook(class_label) + pos_start_emb # [bs, 1, 1024]

    """ Input """
    token_maps = get_token_maps(R, codebook, patch_nums) # [bs, L-1, codebook_dim]
    token_maps_emb = token_linear(token_maps, embed_dim) # [bs, L-1, embed_dim]
    token_maps_emb = torch.cat([class_emb, token_maps_emb], dim=1) + level_emb # [bs, L, embed_dim]

    """ Insert condition """
    for i in range(depth):
        token_maps_emb = token_block(token_maps_emb, class_emb) # SA + FFN

    """ Output: loss """
    # logit
    logit = get_logit(token_maps_emb, class_emb) # [bs, L, 4096]

    # loss
    loss = cross_entropy(logit, gt_idx) # [bs, L]
    loss = loss.sum(dim=1).mean()

    return loss

def var_inference(class_label, patch_nums):
    bs = class_label.shape[0]
    L = sum(pn * pn for pn in patch_nums)
    embed_dim = 1024
    depth = 16

    # pretrained-vqvae
    codebook_num = 4096
    codebook_dim = 32
    vqvae_codebook = nn.Embedding(codebook_num, codebook_dim)

    # Load pretrained embedding at var_training
    class_emb = torch.randn([bs * 2, 1, embed_dim])
    """
    class_codebook(class_label) + pos_start
    *2 for classfier free guidance -> e.g., [label0, label1, label3, class_num, class_num, class_num]
    """
    pos_emb = torch.randn([bs, L, embed_dim])
    level_emb = torch.randn([bs, L, embed_dim])

    """ Input """
    token_emb = class_emb + pos_emb[:, :1] + level_emb[:, :1] # [bs * 2, 1, embed_dim] -> start_token
    latent = torch.zeros([bs, 32, 16, 16])

    """ Insert condition """
    last_stage_index = len(patch_nums) - 1
    current_position = 0
    for stage_index, pn in enumerate(patch_nums):
        stage_ratio = stage_index / last_stage_index
        current_position += pn * pn

        for i in range(depth):
            token_emb = token_block(token_emb, class_emb)

        logit = get_logit(token_emb, class_emb)

        # classfier-free guidance
        cfg_scale = 5
        t = cfg_scale * stage_ratio
        logit = (1 + t) * logit[:bs] - t * logit[bs:]

        # sampling
        sampled_logit = sampling_top_k_p(logit, top_k=900, top_p=0.96, num_samples=1)[:, :, 0]

        # get token embedding
        token_emb = vqvae_codebook(sampled_logit)
        token_emb = token_emb.transpose(1, 2).view(bs, codebook_dim, pn, pn)

        # get next token embedding
        latent, next_token_emb = get_next_autoregressive_input(stage_index, patch_nums, latent, token_emb)

        if stage_index != last_stage_index:
            next_token_emb = next_token_emb.view(bs, codebook_dim, -1).transpose(1, 2)

            next_position = current_position + patch_nums[stage_index + 1] ** 2

            # [bs, next_patch*next_patch, embed_dim]
            next_token_map = token_linear(next_token_emb, embed_dim) + level_emb[:, current_position: next_position]

            token_emb = next_token_map.repeat(2, 1, 1)

    """ Output: Generated image """
    img = vqvae_decode(latent)

    return img


# First. VQVAE Training
vqvae_loss = vqvae_training(img, patch_nums)
print('vqvae training finish')

# Second. VAR Training
var_loss = var_training(img, patch_nums, class_label)
print('var training finish')

# Final. VAR Inference
generated_image = var_inference(class_label, patch_nums)
print('Generate image: ', generated_image.shape)
