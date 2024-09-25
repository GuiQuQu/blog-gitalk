---
title: "VAE"
date: 2023-04-20T17:24:55+08:00
draft: true
url: /deep-learning/vae
categories:
    - deep-learning
---

# VAE
{{< math >}}
$$
\begin{align}
 \log p(x) & = \int_z q(z|x) \log p(x) dz \\
 & = \int_z q(z|x) \log (\frac{p(z,x)}{p(z|x)}) \\
 & = \int_z q(z|x) \log (\frac{p(z,x) q(z|x)}{q(z,x)p(z|x)}) \\
 & = \int_z q(z|x) \log (\frac{p(z,x)}{q(z|x)}) + \int_z q(z|x) \log (\frac{q(z|x)}{p(z|x)}) \\
 & = ELBO + KL(q(z|x) \parallel p(z|x)) \\
& \ge ELBO = \mathbb{E}_{q(z|x)}[\log (\frac{p(z,x)}{q(z|x)})] \\
\end{align} \tag 1
$$
{{< /math >}}

{{< math >}}
$$
 a = y b + c \\
 da \\
 q_a \\
$$
{{< /math >}}


$a+b=c$ 

