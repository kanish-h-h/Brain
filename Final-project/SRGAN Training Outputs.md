
<hr>

# Attempt 1

## Initial
- `generator_optimizer`: `5e-6`
- `discriminator_optimizer`: `1e-5`

- Total loss
- `total_gen_loss = 0.01 * adv_loss_value + content_loss_value + 0.1 * perceptual_loss_value`

## Train
 ![[Pasted image 20250323220433.png]]
 
 - PSNR: 7.29 dB 
 - SSIM: 0.3482

---

## Valid
![[Pasted image 20250323220519.png]]

- PSNR: 9.88 dB 
- SSIM: 0.3010

## Losses

| Epoch        | Gen Loss | Disc Loss | Time (sec) |
|-------------|---------|----------|------------|
| 1980/2000  | 0.9101  | 1.0004   | 0.37       |
| 1981/2000  | 0.9096  | 1.0004   | 0.37       |
| 1982/2000  | 0.9091  | 1.0004   | 0.37       |
| 1983/2000  | 0.9086  | 1.0004   | 0.37       |
| 1984/2000  | 0.9081  | 1.0003   | 0.37       |
| 1985/2000  | 0.9076  | 1.0003   | 0.37       |
| 1986/2000  | 0.9071  | 1.0003   | 0.37       |
| 1987/2000  | 0.9066  | 1.0003   | 0.37       |
| 1988/2000  | 0.9060  | 1.0003   | 0.37       |
| 1989/2000  | 0.9055  | 1.0003   | 0.37       |
| 1990/2000  | 0.9051  | 1.0003   | 0.37       |
| 1991/2000  | 0.9046  | 1.0003   | 0.37       |
| 1992/2000  | 0.9041  | 1.0003   | 0.37       |
| 1993/2000  | 0.9036  | 1.0003   | 0.37       |
| 1994/2000  | 0.9031  | 1.0003   | 0.37       |
| 1995/2000  | 0.9026  | 1.0003   | 0.37       |
| 1996/2000  | 0.9021  | 1.0003   | 0.37       |
| 1997/2000  | 0.9016  | 1.0003   | 0.37       |
| 1998/2000  | 0.9011  | 1.0003   | 0.37       |
| 1999/2000  | 0.9006  | 1.0003   | 0.37       |
| 2000/2000  | 0.9001  | 1.0003   | 0.77       |

`csv`: {link to github}


<hr>

# Attempt 2

## Changes
- `generator_optimizer`: `1e-5`
- `discriminator_optimizer`: `5e-6`

- Total loss
- `total_gen_loss = 0.01 * adv_loss_value + content_loss_value + 0.2 * perceptual_loss_value`

## Expected Results
- High resolution
- capture more detail (SSIM)
- more details : sharper edges, curves, colors
- LR image -> SR image which should be ~ HR image

## Train
 ![[Pasted image 20250323223746.png]]
 
 - PSNR: 7.26 dB 
 - SSIM: 0.3643

---

## Valid
![[Pasted image 20250323223817.png]]

- PSNR: 9.69 dB 
- SSIM: 0.2865

## Losses

| Epoch        | Gen Loss | Disc Loss | Time (sec) |
|-------------|---------|----------|------------|
| 1980/2000  | 0.9170  | 1.0011   | 0.36       |
| 1981/2000  | 0.9164  | 1.0011   | 0.37       |
| 1982/2000  | 0.9159  | 1.0011   | 0.36       |
| 1983/2000  | 0.9153  | 1.0011   | 0.36       |
| 1984/2000  | 0.9148  | 1.0011   | 0.37       |
| 1985/2000  | 0.9143  | 1.0011   | 0.36       |
| 1986/2000  | 0.9137  | 1.0011   | 0.37       |
| 1987/2000  | 0.9132  | 1.0011   | 0.37       |
| 1988/2000  | 0.9127  | 1.0011   | 0.36       |
| 1989/2000  | 0.9121  | 1.0011   | 0.37       |
| 1990/2000  | 0.9116  | 1.0011   | 0.37       |
| 1991/2000  | 0.9111  | 1.0011   | 0.36       |
| 1992/2000  | 0.9105  | 1.0011   | 0.36       |
| 1993/2000  | 0.9100  | 1.0011   | 0.36       |
| 1994/2000  | 0.9095  | 1.0011   | 0.37       |
| 1995/2000  | 0.9089  | 1.0010   | 0.36       |
| 1996/2000  | 0.9084  | 1.0010   | 0.37       |
| 1997/2000  | 0.9079  | 1.0010   | 0.36       |
| 1998/2000  | 0.9073  | 1.0010   | 0.36       |
| 1999/2000  | 0.9068  | 1.0010   | 0.36       |
| 2000/2000  | 0.9063  | 1.0010   | 0.76       |

`csv`: {link to github}

<hr>

# Attempt 3

## Changes
- `generator_optimizer`: `2e-5`
- `discriminator_optimizer`: `5e-6`

- Total loss
- `total_gen_loss = 0.01 * adv_loss_value + content_loss_value + 0.3 * perceptual_loss_value`

- Epochs 2000 -> 3000

## Expected Results
- Improved Texture & Perceptual Quality
- Slightly Lower PSNR, But Higher SSIM
	- **PSNR may slightly drop** because textures are added, increasing pixel-wise differences  
	- **SSIM should increase** because it measures structural similarity, which is the real goal
- more details : sharper edges, curves, colors
- LR image -> SR image which should be ~ HR image

## Train
 ![[Pasted image 20250323231341.png]]
 
 - PSNR: 7.29 dB 
 - SSIM: 0.4506

---

## Valid
![[Pasted image 20250323231413.png]]

- PSNR: 9.92 dB 
- SSIM: 0.3523

## Losses

| Epoch  | Gen Loss | Disc Loss | Time (sec) |
|--------|---------|----------|------------|
| 2980   | 0.4347  | 1.0005   | 0.38       |
| 2981   | 0.4344  | 1.0005   | 0.38       |
| 2982   | 0.4342  | 1.0005   | 0.38       |
| 2983   | 0.4340  | 1.0005   | 0.38       |
| 2984   | 0.4337  | 1.0005   | 0.38       |
| 2985   | 0.4335  | 1.0005   | 0.38       |
| 2986   | 0.4332  | 1.0005   | 0.38       |
| 2987   | 0.4330  | 1.0005   | 0.38       |
| 2988   | 0.4328  | 1.0005   | 0.38       |
| 2989   | 0.4326  | 1.0005   | 0.38       |
| 2990   | 0.4324  | 1.0005   | 0.38       |
| 2991   | 0.4322  | 1.0005   | 0.38       |
| 2992   | 0.4319  | 1.0005   | 0.38       |
| 2993   | 0.4317  | 1.0005   | 0.38       |
| 2994   | 0.4315  | 1.0005   | 0.38       |
| 2995   | 0.4313  | 1.0005   | 0.38       |
| 2996   | 0.4311  | 1.0005   | 0.38       |
| 2997   | 0.4309  | 1.0005   | 0.38       |
| 2998   | 0.4306  | 1.0005   | 0.38       |
| 2999   | 0.4304  | 1.0005   | 0.38       |
| 3000   | 0.4302  | 1.0005   | 0.95       |

`csv`: {link to github}

# Attempt 4

## Changes
- `generator_optimizer`: `5e-6`
- `discriminator_optimizer`: `5e-6`

- Total loss
- `total_gen_loss = 0.01 * adv_loss_value + content_loss_value + 0.05 * perceptual_loss_value`

- Epochs 2000 -> 3000

## Expected Results
|Change|Expected Improvement|Warning Signs|
|---|---|---|
|**Generator LR ↑** (5e-6 → 5e-5)|Faster learning, sharper details|Unstable training, artifacts appear|
|**Discriminator LR ↓** (1e-5 → 5e-6)|Generator catches up, better textures|Discriminator gets too weak (check loss <0.5)|
|**Adv. Loss Weight ↑** (0.001 → 0.01)|More realistic textures|Artifacts in generated images|
|**Perceptual Loss ↓** (0.3 → 0.05)|Better pixel accuracy (PSNR ↑)|Slight drop in SSIM (less feature-matching)|
|**2000 → 3000 Epochs**|More refinement in textures|If loss plateaus, might not help|

## Train
 ![[Pasted image 20250323231341.png]]
 
 - PSNR: 7.29 dB 
 - SSIM: 0.4506

---

## Valid
![[Pasted image 20250323231413.png]]

- PSNR: 9.92 dB 
- SSIM: 0.3523

## Losses

| Epoch  | Gen Loss | Disc Loss | Time (sec) |
|--------|---------|----------|------------|
| 2980   | 0.4347  | 1.0005   | 0.38       |
| 2981   | 0.4344  | 1.0005   | 0.38       |
| 2982   | 0.4342  | 1.0005   | 0.38       |
| 2983   | 0.4340  | 1.0005   | 0.38       |
| 2984   | 0.4337  | 1.0005   | 0.38       |
| 2985   | 0.4335  | 1.0005   | 0.38       |
| 2986   | 0.4332  | 1.0005   | 0.38       |
| 2987   | 0.4330  | 1.0005   | 0.38       |
| 2988   | 0.4328  | 1.0005   | 0.38       |
| 2989   | 0.4326  | 1.0005   | 0.38       |
| 2990   | 0.4324  | 1.0005   | 0.38       |
| 2991   | 0.4322  | 1.0005   | 0.38       |
| 2992   | 0.4319  | 1.0005   | 0.38       |
| 2993   | 0.4317  | 1.0005   | 0.38       |
| 2994   | 0.4315  | 1.0005   | 0.38       |
| 2995   | 0.4313  | 1.0005   | 0.38       |
| 2996   | 0.4311  | 1.0005   | 0.38       |
| 2997   | 0.4309  | 1.0005   | 0.38       |
| 2998   | 0.4306  | 1.0005   | 0.38       |
| 2999   | 0.4304  | 1.0005   | 0.38       |
| 3000   | 0.4302  | 1.0005   | 0.95       |

`csv`: {link to github}


# Attempt 5

## Changes
   
|**Change**|**Current Value**|**New Value**|**Expected Improvement**|
|---|---|---|---|
|Generator LR|`5e-5`|`1e-4`|Faster convergence, better feature learning|
|Discriminator LR|`5e-6`|`2e-6`|Stabilizes training, prevents mode collapse|
|Adversarial Loss Weight|`0.05`|`0.01`|Prevents over-reliance on fooling discriminator|
|Perceptual Loss Weight|`0.05`|`0.1`|Improves textures, edges, and fine details|
|Training Epochs|`3000`|`4000`|Better convergence, higher PSNR|

## Train

![[Pasted image 20250323235655.png]]

- PSNR: 7.31 dB
- SSIM: 0.3831

---

## Valid

![[Pasted image 20250323235708.png]]

- PSNR: 10.2 dB
- SSIM: 0.3602

## Losses


| Epoch  | Gen Loss | Disc Loss | Time (sec) |
|--------|---------|----------|------------|
| 2980   | 0.3131  | 1.0005   | 0.38       |
| 2981   | 0.3130  | 1.0005   | 0.38       |
| 2982   | 0.3129  | 1.0005   | 0.38       |
| 2983   | 0.3128  | 1.0005   | 0.38       |
| 2984   | 0.3126  | 1.0005   | 0.38       |
| 2985   | 0.3125  | 1.0005   | 0.38       |
| 2986   | 0.3124  | 1.0005   | 0.38       |
| 2987   | 0.3123  | 1.0005   | 0.38       |
| 2988   | 0.3122  | 1.0005   | 0.38       |
| 2989   | 0.3121  | 1.0005   | 0.38       |
| 2990   | 0.3120  | 1.0005   | 0.38       |
| 2991   | 0.3119  | 1.0005   | 0.38       |
| 2992   | 0.3117  | 1.0005   | 0.38       |
| 2993   | 0.3116  | 1.0005   | 0.38       |
| 2994   | 0.3115  | 1.0005   | 0.38       |
| 2995   | 0.3114  | 1.0005   | 0.38       |
| 2996   | 0.3113  | 1.0005   | 0.38       |
| 2997   | 0.3112  | 1.0005   | 0.38       |
| 2998   | 0.3111  | 1.0005   | 0.38       |
| 2999   | 0.3110  | 1.0005   | 0.38       |
| 3000   | 0.3109  | 1.0005   | 0.93       |

`csv`: {link to github}

# Attempt 6

## Changes
   
|**Change**|**Expected Effect**|
|---|---|
|**Higher Gen LR (2e-4)**|Faster feature learning, higher PSNR|
|**Lower Disc LR (2e-6)**|Better training balance, avoids overfitting|
|**Higher Perceptual Loss (0.15)**|Retains details, boosts SSIM|
|**Extra 1000 Epochs**|Gradual improvement, watch for stagnation|

## Train

![[Pasted image 20250324002747.png]]

- PSNR: 7.33 dB
- SSIM: 0.4787

---

## Valid

![[Pasted image 20250324002811.png]]

- PSNR: 10.3 dB
- SSIM: 0.3816

## Losses


| Epoch  | Gen Loss | Disc Loss | Time (sec) |
|--------|---------|----------|------------|
| 2980   | 0.0584  | 1.0051   | 0.37       |
| 2981   | 0.0585  | 1.0051   | 0.37       |
| 2982   | 0.0572  | 1.0051   | 0.37       |
| 2983   | 0.0557  | 1.0051   | 0.37       |
| 2984   | 0.0544  | 1.0051   | 0.37       |
| 2985   | 0.0541  | 1.0051   | 0.37       |
| 2986   | 0.0548  | 1.0051   | 0.37       |
| 2987   | 0.0555  | 1.0051   | 0.37       |
| 2988   | 0.0555  | 1.0051   | 0.37       |
| 2989   | 0.0545  | 1.0050   | 0.37       |
| 2990   | 0.0536  | 1.0050   | 0.37       |
| 2991   | 0.0535  | 1.0050   | 0.37       |
| 2992   | 0.0539  | 1.0050   | 0.37       |
| 2993   | 0.0541  | 1.0050   | 0.37       |
| 2994   | 0.0539  | 1.0050   | 0.37       |
| 2995   | 0.0537  | 1.0050   | 0.37       |
| 2996   | 0.0537  | 1.0050   | 0.37       |
| 2997   | 0.0537  | 1.0050   | 0.37       |
| 2998   | 0.0535  | 1.0050   | 0.37       |
| 2999   | 0.0531  | 1.0050   | 0.37       |
| 3000   | 0.0529  | 1.0050   | 0.77       |


`csv`: {link to github}

<hr>

# Attempt 7

## Changes
   

## Train

![[Pasted image 20250324010753.png]]

- PSNR: 7.34 dB
- SSIM: 0.4935

---

## Valid

![[Pasted image 20250324010720.png]]

- PSNR: 10.7 dB
- SSIM: 0.3915

## Losses


| Epoch  | Gen Loss | Disc Loss | Time (sec) |
|--------|---------|----------|------------|
| 3980   | 0.0261  | 1.0047   | 0.39       |
| 3981   | 0.0260  | 1.0052   | 0.38       |
| 3982   | 0.0260  | 1.0047   | 0.39       |
| 3983   | 0.0261  | 1.0051   | 0.39       |
| 3984   | 0.0259  | 1.0047   | 0.38       |
| 3985   | 0.0261  | 1.0050   | 0.38       |
| 3986   | 0.0260  | 1.0047   | 0.38       |
| 3987   | 0.0260  | 1.0049   | 0.39       |
| 3988   | 0.0259  | 1.0047   | 0.38       |
| 3989   | 0.0258  | 1.0050   | 0.39       |
| 3990   | 0.0256  | 1.0047   | 0.39       |
| 3991   | 0.0256  | 1.0050   | 0.39       |
| 3992   | 0.0256  | 1.0047   | 0.39       |
| 3993   | 0.0257  | 1.0050   | 0.38       |
| 3994   | 0.0258  | 1.0047   | 0.39       |
| 3995   | 0.0258  | 1.0051   | 0.38       |
| 3996   | 0.0257  | 1.0047   | 0.39       |
| 3997   | 0.0256  | 1.0050   | 0.39       |
| 3998   | 0.0254  | 1.0047   | 0.38       |
| 3999   | 0.0254  | 1.0049   | 0.38       |
| 4000   | 0.0254  | 1.0047   | 0.85       |



`csv`: {link to github}