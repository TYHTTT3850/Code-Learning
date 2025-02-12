# Matplotlib理论部分

Matplotlib 提出了Object Container(对象容器)的概念，主要有Figure，Axes，Axis，Tick。Figure 和 Axes 对象是图形绘制的核心组成部分。

Figure负责图像大小，位置等操作；Axes负责坐标轴位置，绘图等操作；Axis负责坐标轴的设置等操作；Tick负责格式化刻度的样式等操作。

这四种对象间是层层包含关系。Tick ⊂ Axis ⊂ Axes ⊂ Figure。
## Figure对象

Figure 对象是整个绘图的顶层容器，是 Matplotlib 中所有绘图操作的基础，所有的绘图元素和操作都是围绕Figure对象进行的。

Figure 对象是整个图形的顶层容器，代表了一幅完整的图像。它可以包含一个或多个 Axes 对象、标题、图例、注释等。可以把 Figure对象想象成一张画布，所有的绘图元素（如线条、文本、图例等）都在这张画布上进行绘制。

## Axes对象

Axes 对象代表图形中的一个坐标系统，它是图形中数据可视化的主要区域。每个 `Axes` 对象可以包含多个绘图元素，如线条、点、文本等。

一个Figure对象可以包含多个Axes对象，而一个Axes对象仅存在于一个Figure对象中。

