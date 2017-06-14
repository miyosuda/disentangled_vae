# Disentangled VAE

[![CircleCI](https://circleci.com/gh/miyosuda/disentangled_vae.svg?style=svg)](https://circleci.com/gh/miyosuda/disentangled_vae)

Replicating DeepMind's paper "Early Visual Concept Learning with Unsupervised Deep Learning"
https://arxiv.org/pdf/1606.05579.pdf

## 2D shape disentaglement

Result by changing latent Z from -3.0 to 3.0 with β=4.0

Disentaglment is obscure but latent variables with small variance "Z2", "Z3, "Z5", "Z7", "Z8" seem extracting "x", "y", "rotation" and "scale" parameters.

Other latent variables are all drawin into N(0,I), so variances are round 1.0.

(This experiment is using DeepMind's [dsprite data set](https://github.com/deepmind/dsprites-dataset).)


z0:
![](disentangle_img/check_z0_0.png)
![](disentangle_img/check_z0_1.png)
![](disentangle_img/check_z0_2.png)
![](disentangle_img/check_z0_3.png)
![](disentangle_img/check_z0_4.png)
![](disentangle_img/check_z0_5.png)
![](disentangle_img/check_z0_6.png)
![](disentangle_img/check_z0_7.png)
![](disentangle_img/check_z0_8.png)
![](disentangle_img/check_z0_9.png)
1.00

z1:
![](disentangle_img/check_z1_0.png)
![](disentangle_img/check_z1_1.png)
![](disentangle_img/check_z1_2.png)
![](disentangle_img/check_z1_3.png)
![](disentangle_img/check_z1_4.png)
![](disentangle_img/check_z1_5.png)
![](disentangle_img/check_z1_6.png)
![](disentangle_img/check_z1_7.png)
![](disentangle_img/check_z1_8.png)
![](disentangle_img/check_z1_9.png)
1.00

**z2**:
![](disentangle_img/check_z2_0.png)
![](disentangle_img/check_z2_1.png)
![](disentangle_img/check_z2_2.png)
![](disentangle_img/check_z2_3.png)
![](disentangle_img/check_z2_4.png)
![](disentangle_img/check_z2_5.png)
![](disentangle_img/check_z2_6.png)
![](disentangle_img/check_z2_7.png)
![](disentangle_img/check_z2_8.png)
![](disentangle_img/check_z2_9.png)
**0.02** -> rotation

**z3**:
![](disentangle_img/check_z3_0.png)
![](disentangle_img/check_z3_1.png)
![](disentangle_img/check_z3_2.png)
![](disentangle_img/check_z3_3.png)
![](disentangle_img/check_z3_4.png)
![](disentangle_img/check_z3_5.png)
![](disentangle_img/check_z3_6.png)
![](disentangle_img/check_z3_7.png)
![](disentangle_img/check_z3_8.png)
![](disentangle_img/check_z3_9.png)
**0.05** -> rotation

z4:
![](disentangle_img/check_z4_0.png)
![](disentangle_img/check_z4_1.png)
![](disentangle_img/check_z4_2.png)
![](disentangle_img/check_z4_3.png)
![](disentangle_img/check_z4_4.png)
![](disentangle_img/check_z4_5.png)
![](disentangle_img/check_z4_6.png)
![](disentangle_img/check_z4_7.png)
![](disentangle_img/check_z4_8.png)
![](disentangle_img/check_z4_9.png)
1.00

**z5**:
![](disentangle_img/check_z5_0.png)
![](disentangle_img/check_z5_1.png)
![](disentangle_img/check_z5_2.png)
![](disentangle_img/check_z5_3.png)
![](disentangle_img/check_z5_4.png)
![](disentangle_img/check_z5_5.png)
![](disentangle_img/check_z5_6.png)
![](disentangle_img/check_z5_7.png)
![](disentangle_img/check_z5_8.png)
![](disentangle_img/check_z5_9.png)
**0.03** -> X

z6:
![](disentangle_img/check_z6_0.png)
![](disentangle_img/check_z6_1.png)
![](disentangle_img/check_z6_2.png)
![](disentangle_img/check_z6_3.png)
![](disentangle_img/check_z6_4.png)
![](disentangle_img/check_z6_5.png)
![](disentangle_img/check_z6_6.png)
![](disentangle_img/check_z6_7.png)
![](disentangle_img/check_z6_8.png)
![](disentangle_img/check_z6_9.png)
1.00

**z7**:
![](disentangle_img/check_z7_0.png)
![](disentangle_img/check_z7_1.png)
![](disentangle_img/check_z7_2.png)
![](disentangle_img/check_z7_3.png)
![](disentangle_img/check_z7_4.png)
![](disentangle_img/check_z7_5.png)
![](disentangle_img/check_z7_6.png)
![](disentangle_img/check_z7_7.png)
![](disentangle_img/check_z7_8.png)
![](disentangle_img/check_z7_9.png)
**0.01** -> scale

**z8**:
![](disentangle_img/check_z8_0.png)
![](disentangle_img/check_z8_1.png)
![](disentangle_img/check_z8_2.png)
![](disentangle_img/check_z8_3.png)
![](disentangle_img/check_z8_4.png)
![](disentangle_img/check_z8_5.png)
![](disentangle_img/check_z8_6.png)
![](disentangle_img/check_z8_7.png)
![](disentangle_img/check_z8_8.png)
![](disentangle_img/check_z8_9.png)
**0.02** -> Y

z9:
![](disentangle_img/check_z9_0.png)
![](disentangle_img/check_z9_1.png)
![](disentangle_img/check_z9_2.png)
![](disentangle_img/check_z9_3.png)
![](disentangle_img/check_z9_4.png)
![](disentangle_img/check_z9_5.png)
![](disentangle_img/check_z9_6.png)
![](disentangle_img/check_z9_7.png)
![](disentangle_img/check_z9_8.png)
![](disentangle_img/check_z9_9.png)
0.99


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
