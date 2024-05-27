# pseudo_var
Pseudo code for [VAR: Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)

---
### Details
#### Token block
<img width="159" alt="image" src="https://github.com/taki0112/pseudo_var/assets/26713992/f40a6835-07e3-4c6d-843d-c10eaa1a45dc">
---

### Training Process
#### 1. VQVAE Training
```
vqvae_loss = vqvae_training(img, patch_nums)
print('vqvae training finish')
```

#### 2. VAR Training
```
var_loss = var_training(img, patch_nums, class_label)
print('var training finish')
```

---

### Inference
#### VAR Inference
```
generated_image = var_inference(class_label, patch_nums)
print('Generate image: ', generated_image.shape)
```
