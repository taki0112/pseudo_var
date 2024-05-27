# pseudo_var
Pseudo code for [VAR: Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)

---
### Details
#### Token block (not implemented here)
<img width="159" alt="image" src="https://github.com/taki0112/pseudo_var/assets/26713992/f40a6835-07e3-4c6d-843d-c10eaa1a45dc">

---

### Training Process
#### 1. VQVAE Training
<img width="612" alt="image" src="https://github.com/taki0112/pseudo_var/assets/26713992/83b60be1-a60a-4020-b453-aa90d7d1c0f2">

<img width="1291" alt="image" src="https://github.com/taki0112/pseudo_var/assets/26713992/19235126-053a-460f-b2ea-ed57e360ae7d">


```
vqvae_loss = vqvae_training(img, patch_nums)
print('vqvae training finish')
```

#### 2. VAR Training
<img width="872" alt="image" src="https://github.com/taki0112/pseudo_var/assets/26713992/d92f3289-7fcb-4ed7-bfa5-e426ed621e2c">

```
var_loss = var_training(img, patch_nums, class_label)
print('var training finish')
```

---

### Inference
#### VAR Inference
<img width="873" alt="image" src="https://github.com/taki0112/pseudo_var/assets/26713992/4f07af30-c234-4d4b-87e5-68eb43569a16">

```
generated_image = var_inference(class_label, patch_nums)
print('Generate image: ', generated_image.shape)
```
