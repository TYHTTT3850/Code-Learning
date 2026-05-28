要把模型用在真正应用，只有正确率高是不够的，还需要能够应付来自人类的恶意攻击，在有人试图想要欺骗它的情况下，也得到高的正确率。

攻击是指通过设计特定的输入数据、操纵训练过程或利用模型的底层漏洞，来破坏、欺骗、误导或刺探机器学习系统的行为。例如：在一张照片的每个像素点上都加入一个小噪声，通常都小到肉眼看不出来，将这张照片输入到网络当中，模型输出的类别不是正确的答案。

# 如何进行攻击

## Non-Targeted & Targeted

**Non-targeted**：任何非预期的输出都可以。如：带噪声的图片 $\boldsymbol{x}$ 输入给模型，希望产生的预测 $\hat{\boldsymbol{y}}$ 要跟实际答案 $\boldsymbol{y}$ 差距越大越好。

**Targeted**：要求有一个特定输出 $\boldsymbol{y}_\text{traget}$，期望预测的 $\hat{\boldsymbol{y}}$ 不只跟真实的 $\boldsymbol{y}$ 差距越大越好，还要跟特定输出 $\boldsymbol{y}_\text{traget}$ 越接近越好。

此外，期望加入噪声后的输入 $\boldsymbol{x}$ 要与原始输入 $\boldsymbol{x}_0$ 越接近越好，所以会加入 $d(\boldsymbol{x}_0,\boldsymbol{x}) \leqslant \varepsilon$ 的限制，反映到图像上就是让两张图片的差距小于等于人类感知的极限。最后我们想要的带噪声的输入就是：

$$
\boldsymbol{x}^* = \arg\min_{d(\boldsymbol{x}_0,\boldsymbol{x}) \leqslant \varepsilon} \mathcal{L}(x),\\
\mathcal{L}(x)=
\begin{cases}
-\text{error}(\hat{\boldsymbol{y}},\boldsymbol{y}),\\
-\text{error}(\hat{\boldsymbol{y}},\boldsymbol{y})+\text{error}(\hat{\boldsymbol{y}},\boldsymbol{y}_\text{traget}).
\end{cases}
$$

计算 $d(\boldsymbol{x}_0,\boldsymbol{x})$ 的方式：

- L2-norm：$\left\| \boldsymbol{x}-\boldsymbol{x}_0 \right\|_2 = (\Delta x_1)^2+\cdots+(\Delta x_n)^2$ 。
- L-infinity：$\left\| \boldsymbol{x}-\boldsymbol{x}_0 \right\|_\infty = \max\{|\Delta x_1|,\cdots,|\Delta x_n|\}$ 。
- $\cdots$

具体选用哪一种方式需要根据特定领域内的专业知识来决定。

## 白盒攻击(White Box Attack)

白盒攻击：模型参数已知，本质上就是解一个优化问题：$\boldsymbol{x}^* = \arg\displaystyle\min_{d(\boldsymbol{x}_0,\boldsymbol{x}) \leqslant \varepsilon} \mathcal{L}(x)$ 。

先不考虑 $d(\boldsymbol{x}_0,\boldsymbol{x}) \leqslant \varepsilon$ ，可以使用梯度下降方法实现，只不过这里需要优化的是输入而不是模型的参数，初始化直接从 $x_0$ 开始，因为希望找到的带噪声的输入 $\boldsymbol{x}^*$ 要跟原始输入 $\boldsymbol{x}_0$ 越接近越好，随后进行迭代。每次迭代完，再将 $d(\boldsymbol{x}_0,\boldsymbol{x}) \leqslant \varepsilon$ 约束进来，设第 $t$ 次迭代找到的带噪声的输入为 $\boldsymbol{x}_t$ ，若 $d(\boldsymbol{x}_0,\boldsymbol{x}) > \varepsilon$ ，则将 $\boldsymbol{x}_t$ 修正一下以符合限制。

典型的方法有 FGSM(Fast Gradient Sign Method) 及其迭代版本 Iterative FGSM 。

## 黑盒攻击

黑盒攻击：模型参数未知。一般来说线上服务的模型都不知道参数。

可以训练一个 proxy network 来模仿被攻击的对象。 如果 proxy network 跟要被攻击的对象比较相似，并且用加入噪声后的输入 $\boldsymbol{x}$ 对 proxy network 进行攻击能够产生效果，那么将 $\boldsymbol{x}$ 输入到不知道参数的模型中进行攻击一般来说也会成功。

黑盒攻击两种状况 ：

- 有办法取得模型的训练数据：那么以此数据训练 proxy network，如此它们就有一定程度的相似度。
- 没办法取得模型的训练数据：将数据输入到未知参数的模型得到输出，如此就有输入输出的成对数据，再将其作为训练数据拿去训练出一个 proxy network ，再进行攻击。

实际上黑箱攻击是在 non-targeted attack 的时候比较容易成功，targeted attack 不太容易成功。

## 为什么攻击如此简单

原因可能是出现在资料上而不是模型上。 在有限的资料上，机器学到的就是这样子的结论，当我们有足够的资料，也许就有机会避免 adversarial attack

# 攻击的案例

one-pixel attack：只动了图片中的一个 pixel，图像识别系统的判断就产生错误。

Universal Adversarial Attack：只用一种 attacked signal 就成功攻击所有的图片，无需对每一张图片进行定制化处理。

Speech Processing：一段显然的合成声音信号加上一个微小的噪声以后，同一个侦测声音是否合成的模型会觉得刚才那段声音是真实的声音，而不是合成的声音。

 Natural Language Processing：控制 Q&A 模型结果。例如：在所有文章末尾加上 “Why How Because To Kill American People” 这段文字，接下来不管问机器什么问题，它的答案都是 “To Kill American People” 。

人脸识别：戴上一副特殊眼镜后导致模型辨识错误，即使转换多个角度都会产生错误的结果。

标志辨识：在交通信号上贴贴纸导致模型识别错误。

Adversarial Reprogramming：让一个模型完成训练任务之外的工作，例如：用图像识别的系统进行数方格的任务，将方格图片嵌入到噪声中。

“Backdoor” in Model：在模型训练的阶段就进行攻击。例如：训练数据中加入一些人看起来没有问题，但对模型会有问题的数据。 训练完成后，模型就如同开了一个后门，在测试阶段只对某一张图片识别错误，而对其他图片表现正常。

# 防御的方式

## 被动防御

1. 在不改变模型的情况下，将数据输入进模型之前，先加 filter 以削减 attack signal 。例如：把图片稍微做一点模糊化，就可以达到的防御效果。这是因为attack signal 其实只有在某一个方向上的某一种攻击才能够成功，并不是随便一个噪声都可以攻击成功。但是会有副作用：模型输出的置信度会下降。
2. 压缩图像，因为图像保存为 JPEG 文件以后会失真，可以降低 attack signal 的影响。
3. 把输入的图片用 generator 重新产生。图片上加的微小噪声对 generator 而言从来没有看过，也就无法产生这些非常小的噪声，利用 generator 重新产生图片可以达到防御的效果。

被动防御的问题：

- 一旦被知道采用哪种防御方法，可能就没有防御的效果了。
- 即使随机选择不同的防御方法，如果知道随机分布，还是有可能攻破这种防御的方式。

## 主动防御

Adversarial Training(Data Augmentation) 会在训练阶段，主动利用算法生成带有干扰的攻击样本，然后将这些攻击样本混合到正常的训练数据中，让模型在恶意攻击中学习并纠正错误。

主动防御的问题：

- 不一定能挡得住所有攻击方式，在 Adversarial Training 时没有见过的攻击方式可能没有办法防御。

- 需要大量的计算资源，特别是对图像模型的 Adversarial Training 。
