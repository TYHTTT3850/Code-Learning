传统的机器学习模型大多只能对输入数据进行分析、分类或预测，不具有创造性。生成式网络能够无中生有地创造出全新的、逼真且多样的内容(如图像、文本、音频和视频)。

GAN(生成对抗网络，Generative Adversarial Network)就是一个经典的生成式网络。

# 基本概念

GAN 的核心思想是“对抗”，它由两个主要的神经网络组成：

- 生成器(Generator, 简称 G)：它的任务是“无中生有”。接收一段随机噪声向量 $z$ (通常从正态分布中采样)，然后输出一张图片 $x$ 。它的目标是尽可能生成逼真的图片来骗过判别器。
- 判别器(Discriminator, 简称 D)：它的任务是“分辨真假”。接收一张图片 $x$ (可能是真实的，也可能是 G 生成的)，输出一个标量，代表这张图片是真实的概率。它的目标是准确分辨出真图和假图。

训练过程是交替进行的(以生成图片为例)：

- 固定 G，训练 D： 把真实图片打上标签 1，把 G 生成的图片打上标签 0，目标是让 D 学习如何区分它们。

- 固定 D，训练 G： G 生成图片，丢给已经训练好的 D 去打分，目标是让由 G 生成的图片，D 给出的概率越接近 1 越好。 

在这个互相对抗的过程中，G 生成的图片越来越逼真，D 的鉴别能力也越来越强，最终达到一个平衡。

# GAN 的理论

从数学理论上讲，真实图片服从一个分布 $P_{\text{data}}$，生成器生成的图片服从一个分布 $P_G$。

G 的目标就是让 $P_G$ 无限接近于 $P_{\text{data}}$ ，也就是：
$$
G^* = \arg\min_G \text{Div}(P_G, P_{\text{data}})
$$

$\text{Div}(P_G, P_{\text{data}})$ 是计算 $P_G, P_{\text{data}}$ 的散度。**注意这个散度和数学分析中的散度不是一个概念！**

GAN 里的散度是概率论与信息论中的散度，用来衡量两个概率分布之间“差异程度”的指标，如 KL 散度(连续和离散形式)：

$$
\text{Div}(p,q) = \int_{-\infty}^{+\infty} p(x) \log \left( \frac{p(x)}{q(x)} \right) dx,\\
\text{Div}(P,Q) = \sum_{i=1}^{n} P(x_i) \log \left( \frac{P(x_i)}{Q(x_i)} \right).
$$
数学分析中的散度定义：三维空间中有一可微矢量场 $\vec{F}(x, y, z) = P\cdot\vec{i} + Q\cdot\vec{j} + R\cdot\vec{k}$，则该矢量场在某一点的散度定义为各分量偏导数之和：

$$
\text{div} \mathbf{F} = \nabla \cdot \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}
$$

