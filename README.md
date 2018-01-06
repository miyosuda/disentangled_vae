# Disentangled VAE

[![CircleCI](https://circleci.com/gh/miyosuda/disentangled_vae.svg?style=svg)](https://circleci.com/gh/miyosuda/disentangled_vae)

Replicating DeepMind's papers ["β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"](https://openreview.net/forum?id=Sy2fzU9gl) and ["Understanding disentangling in β-VAE"](https://drive.google.com/file/d/0Bwy4Nlx78QCCNktVTFFMTUs4N2oxY295VU9qV25MWTBQS2Uw/view)


## 2D shape disentaglement

Result by changing latent Z from -3.0 to 3.0 with γ=100.0 and C=20.0

Latent variables with small variances seem extracting "x", "y", "rotation" and "scale" parameters.

(This experiment is using DeepMind's [dsprite data set](https://github.com/deepmind/dsprites-dataset).)


Z  | Image                             | Parameter | Variance
---| ----------------------------------|---------- |-------
z0 | ![](disentangle_anim/anim_z0.gif) |           | 0.9216
z1 | ![](disentangle_anim/anim_z1.gif) |           | 0.9216
z2 | ![](disentangle_anim/anim_z2.gif) | Rotation  | 0.0011
z3 | ![](disentangle_anim/anim_z3.gif) | Rotation? | 0.0038
z4 | ![](disentangle_anim/anim_z4.gif) | Pos X     | 0.0002
z5 | ![](disentangle_anim/anim_z5.gif) |           | 0.9384
z6 | ![](disentangle_anim/anim_z6.gif) | Scale?    | 0.0004
z7 | ![](disentangle_anim/anim_z7.gif) |           | 0.8991
z8 | ![](disentangle_anim/anim_z8.gif) |           | 0.9483
z9 | ![](disentangle_anim/anim_z9.gif) | Pos Y     | 0.0004


## Reconstruction result

Left: original Right: reconstructed image

![](reconstr_img/org_0.png)
![](reconstr_img/reconstr_0.png)

![](reconstr_img/org_1.png)
![](reconstr_img/reconstr_1.png)

![](reconstr_img/org_2.png)
![](reconstr_img/reconstr_2.png)

![](reconstr_img/org_3.png)
![](reconstr_img/reconstr_3.png)

![](reconstr_img/org_4.png)
![](reconstr_img/reconstr_4.png)

![](reconstr_img/org_5.png)
![](reconstr_img/reconstr_5.png)

![](reconstr_img/org_7.png)
![](reconstr_img/reconstr_7.png)

![](reconstr_img/org_8.png)
![](reconstr_img/reconstr_8.png)

![](reconstr_img/org_9.png)
![](reconstr_img/reconstr_9.png)
