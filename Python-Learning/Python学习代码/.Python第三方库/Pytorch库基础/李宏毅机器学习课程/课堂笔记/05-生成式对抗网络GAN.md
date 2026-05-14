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
\text{KL}(p,q) = \int_{-\infty}^{+\infty} p(x) \log \left( \frac{p(x)}{q(x)} \right) dx,\\
\text{KL}(P,Q) = \sum_{i=1}^{n} P(x_i) \log \left( \frac{P(x_i)}{Q(x_i)} \right).
$$
数学分析中的散度定义：三维空间中有一可微矢量场 $\vec{F}(x, y, z) = P\cdot\vec{i} + Q\cdot\vec{j} + R\cdot\vec{k}$，则该矢量场在某一点的散度定义为各分量偏导数之和：

$$
\text{div} \mathbf{F} = \nabla \cdot \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}.
$$

由于只有一堆真实照片的样本，根本写不出分布 $P_{\text{data}}$ 的具体公式，所以上面那个散度 $\text{Div}$ 实际上是无法直接计算的。

但是 D 的目标函数可以实际写出：

$$
V(G, D) = \mathbb{E}_{y \sim P_{\text{data}}}[\log D(y)] + \mathbb{E}_{y \sim P_G}[\log\left(1 - D(y)\right)],\\

D^* = \arg\max_D V(G,D).
$$
$\displaystyle\max_D V(G,D)$ 与 $\text{Div}(P_G, P_{\text{data}})$ 有关(JS 散度)，所以就用 $\displaystyle\max_D V(G,D)$ 取代替散度：

$$
G^* = \arg\min_G\left(\max_D V(G,D)\right).
$$
这个式子的意思就是：

- 首先在给定 $G$ 下，找到一个 $D$ 使得 $V(G,D)$ 达到最大。

- 然后在找到的 $D$ 下，寻找一个 $G$ 使得 $\displaystyle\max_D V(G,D)$ 达到最小。

## JS 散度的问题

JS 散度并不直接对比两个分布 $P$ 和 $Q$，而是先取两者的平均分布 $M = \frac{1}{2}(P + Q)$ ，然后计算 $P$ 与 $M$ 的 KL 散度，以及 $Q$ 与 $M$ 的 KL 散度的平均值。

其数学表达式为：

$$
\text{JS}(P,Q) = \frac{1}{2} \text{KL}(P,M) + \frac{1}{2} \text{KL}(Q,M).
$$

图像在高维空间中其实是低维的流形(Manifold)。这意味着 $P_{\text{data}}$ 和 $P_G$ 在高维空间中极大概率是没有重叠的，即在 $P$ 有值的地方，$Q$ 的值全是 $0$ ，在 $Q$ 有值的地方，$P$ 的值全是 $0$。所以无论它们离得多远还是多近，计算出的 JS 散度永远是一个常数 $\log 2$ 。

## WGAN

WGAN 放弃了 JS 散度，引入了 Wasserstein 距离。其核心思想是：假设有一堆土($P_G$)，要把它挖去填满另一个坑($P_{\text{data}}$)，有很多可能的移动方案，每种方案都有各自的平均移动距离，取最小的平均移动距离就是 Wasserstein 距离。

为了算 Wasserstein 距离，判别器 $D$ 必须满足 $1-\text{Lipschitz}$ 连续性，即函数必须要平滑，不能变化太剧烈，Wasserstein 距离就是：
$$
W(P_{\text{data}}, P_G) = \max_{D \in 1-\text{Lipschitz}} \left\{ \mathbb{E}_{y \sim P_{\text{data}}}[D(y)] - \mathbb{E}_{y \sim P_G}[D(y)] \right\}.
$$

生成器 $G$ 就是：

$$
G^* = \arg\min_G W(P_{\text{data}}, P_G).
$$
上述两个式子的意思就是：

- 首先在给定 $G$ 下，找到一个符合$1-\text{Lipschitz}$ 连续性的 $D$ 使得 $\mathbb{E}_{y \sim P_{\text{data}}}[D(y)] - \mathbb{E}_{y \sim P_G}[D(y)]$ 达到最大，这个最大的值就是 $P_{\text{data}}, P_G$的 Wasserstein 距离 $W(P_{\text{data}}, P_G)$ 。

- 然后在找到的 $D$ 下，寻找一个 $G$ 使得 $P_{\text{data}}, P_G$的 Wasserstein 距离 $W(P_{\text{data}}, P_G)$  达到最小。