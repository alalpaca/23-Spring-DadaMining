# DataMining 考点汇总

10*判断

20*单选

30*多选

40*大题

- 2*5' 简答
- 2*15' 应用

## Before Start.

THU DataMining MOOC:  https://www.bilibili.com/video/BV154411Q7mG

相当赞。



## 第一章

#### 定义、范畴

- 从数据中提取隐式、以前未知的和可能有用的信息
- 通过自动或半自动方式对大量数据进行探索和分析，以便发现有价值的模式
- 即是探究一个东西，而不是对一个既有知识的搜索

#### 判断依据

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527104910805.png" alt="image-20230527104910805" style="zoom: 67%;" />

#### 任务类型

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527105005258.png" alt="image-20230527105005258" style="zoom: 67%;" />

#### 一个什么应用：在ppt上

#### 分类任务：

> ​                                                              Test Set 
>
> ​                                                                    ↓
>
> Trainning Set → **Learn Classifier**   →   Model

**分类任务举例：**

<img src="C:\Users\Kevin Wang\AppData\Roaming\Typora\typora-user-images\image-20230603144611456.png" alt="image-20230603144611456" style="zoom: 50%;" />

**分类应用：**

- 信用卡欺诈检测
- 电话客户的客户流失预测
- 宇宙探测

#### 回归任务：

> 回归任务最终得到的是一个值！

> 假设依赖项的线性或非线性模型，基于其他变量的值预测给定连续变量的值。
>
> 广泛用于统计研究和神经网络领域

**回归应用：**

- 基于广告支出预测新产品的销售额。
- 预测风速，作为温度、湿度、气压等的函数。
- 股票市场指数的时间序列预测。

#### 聚类任务：

> 查找对象组，使组中的对象彼此相似（或相关），并且与其他组中的对象不同（或与对象无关）

<img src="C:\Users\Kevin Wang\AppData\Roaming\Typora\typora-user-images\image-20230603145843152.png" alt="image-20230603145843152" style="zoom:50%;" />

**聚类应用：**

- 数据理解 -> 统计相似事物来理解其数据
- 数据总结 -> 减小大型数据集的大小
- 市场细分 -> 寻找类似客户的集群
- 文档聚类
  - 目标：根据文档中显示的重要术语查找相似的文档组
  - 方法：确定每个文档中频繁出现的术语
  - 根据不同术语的频率形成相似度量，使用它进行群集



#### 数据挖掘挑战可能会问一下

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527105447083.png" alt="image-20230527105447083" style="zoom: 67%;" />

## 第二章 数据 考挺多 很重要（小题）

#### 分析属性类别

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527105636202.png" alt="image-20230527105636202" style="zoom: 67%;" />

#### 离散和连续属性的特点

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527105834899.png" alt="image-20230527105834899" style="zoom: 67%;" />

#### 对称和非对称

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527105853478.png" alt="image-20230527105853478" style="zoom: 67%;" />

#### 各种各样数据类型的特点

看ppt 2.1-2.4 p20-30

#### 数据的质量 影响原因*5

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110115625.png" alt="image-20230527110115625" style="zoom:67%;" />

#### 相似度和距离（简单的出一出）马氏距离不考

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110218040.png" alt="image-20230527110218040" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110229117.png" alt="image-20230527110229117" style="zoom:67%;" />

#### 【计算】简单匹配系数

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110252391.png" alt="image-20230527110252391" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110341263.png" alt="image-20230527110341263" style="zoom:67%;" />



#### 【计算】相关性 《稍微看一看》

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110507542.png" alt="image-20230527110507542" style="zoom: 67%;" />

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110451604.png" style="zoom: 67%;" />

#### 熵

$$
H(X)=-\sum_{i=1}^{n} p_{i} \log _{2} p_{i}
$$

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527111913314.png" alt="image-20230527111913314" style="zoom:67%;" />

### 数据预处理

（好好看一下）

#### 聚集

将两个或多个对象合并成单个对象

**聚集的目的**：

- 减少数据量，从而减小数据存储空间及处理时间；
- 范围或标度转换，例如：
  将不同城市按照行政区、州、国家等进行聚集；
  将日期按照星期、月份、年份等进行聚集。
- 提供更加“稳定的” 数据：
  聚集的数据往往具有更小的变异性

#### 抽样

**抽样的意义**：

- 对统计学家来说，获取感兴趣的整个数据集代价太高，且太费时间；
- 对数据挖掘人员来说，处理所有数据所需的内存太大、时间太长、计算成本太高。

**抽样方法**：

- 简单随机抽样（有放回抽样较简单、无放回抽样理论误差较低）
- 分层抽样
- 渐进抽取

#### 降维

**降维的意义**：

- 减少数据量，从而减小数据存储空间及处理时间
- 避免维灾难
- 消除无关变量
- 有利于模型的理解及数据可视化

**降维方法**：

- 奇异值分解(SVD)
- 主成分分析(PCA)
- 因子分析(FA)
- 独立成分分析(ICA)。

#### 特征子集选择

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527115712428.png" alt="image-20230527115712428" style="zoom:67%;" />

#### 特征创建

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527115820872.png" alt="image-20230527115820872" style="zoom:67%;" />

#### 离散化与二值化

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527115921215.png" alt="image-20230527115921215" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527115944443.png" alt="image-20230527115944443" style="zoom:67%;" />

#### 属性变换

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527120007251.png" alt="image-20230527120007251" style="zoom:67%;" />



## 第三章 分类的基本概念与技术

### 基本分类器 

- 规则方法（Rule-based methods）
- 决策树（Decision Tree based methods）
- 贝叶斯及贝叶斯信念网络（Naïve Bayes and Bayesian Belief Networks）
- 最近邻（Nearest-neighbor）
- **支持向量机**（Support Vector Machines）
- 神经网络（Neural Networks）

*集成分类器

- Boosting, Bagging, Random Forest（随机森林）

### 指标要知道怎么算

#### 【计算】基尼系数

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527134842221.png" alt="image-20230527134842221" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527135002306.png" alt="image-20230527135002306" style="zoom:67%;" />

#### 分类误差 好像没有

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527135229108.png" alt="image-20230527135229108" style="zoom:67%;" />

### 决策树 优缺点

优点

- 算法简单，分类效率高
- 对规模较小的决策树，具有较强的可解释性
- 受不相关属性（如噪声）影响较低，特别是如果构建决策树时采用了相关策略去避免过拟合
- 可有效的处理冗余属性

缺点

- 可能的决策树模型数量随属性增加成指数增长；采用贪心策略可能导致局部最优；
- 不利于处理属性之间的相互作用
- 决策边界仅由单个属性所决定

前剪枝后剪枝？

### 过拟合

原因

- 训练数据集过小
- 过高的模型计算复杂度

解决方案

- 增加训练数据
- 多重比较过程
- 注重有效模型的选择

### 交叉验证

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527140829748.png" alt="image-20230527140829748" style="zoom:67%;" />



## 第四章 最近邻分类器 肯定会考（大小）

分类 或者聚类

#### knn肯定会考

具体可自己寻找材料，==**统计学习方法：第三章**==



<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527143438943.png" alt="image-20230527143438943" style="zoom:67%;" />

#### 人工神经网络肯定会考

具体也可以自己找材料，这里推荐一个我看过的速成资料：

- 北京大学 TensorFlow2.0 其实也就是讲神经网络。https://www.bilibili.com/video/BV1B7411L7Qt/

  

感知器、XORdata

激活函数了解一下

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527143844139.png" alt="image-20230527143844139" style="zoom:67%;" />

==**梯度下降法** 必须理解==

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527143654968.png" alt="image-20230527143654968" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527143716120.png" alt="image-20230527143716120" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527143746812.png" alt="image-20230527143746812" style="zoom: 67%;" />

#### 支持向量机 重点

==边界、支持向量==

==**统计学习方法：第七章**==

#### 贝叶斯 好像不考

==**统计学习方法：第四章**==

### 4.5集成方法 会考

原理：**就是投票**

==**统计学习方法：第八章**==

### 【大题预定！计算】4.6类不平衡 肯定会考

roc不考

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527144730343.png" alt="image-20230527144730343" style="zoom:67%;" />

F1度量：
$$
F 1=\frac{2 \times p \times r}{p+r}
$$
<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527144751062.png" alt="image-20230527144751062" style="zoom:67%;" />



## 第五章 关联分析 肯定会考（大题）

ppt 5.1 

#### 直接去看 一定会考

DataMining, THU. https://www.bilibili.com/video/BV154411Q7mG/?p=40

《不需要任何一个数学公式》



### 5.4不用看 可能有一点的



## 第六章 聚类

kmeans重点

层次更关键

==**统计学习方法：第十四章 聚类方法**==



## 第七章 推荐系统

重点看前两部分

基本MF模型 、矩阵分解 两页

==**统计学习方法：第十七章 潜在语义分析**==



## 第八章 知识图谱（小）

你愿意看英文就去PPT过一遍。

不愿意看就当常识题做，前提是你做过最后一个实验。



## 总结 重点

数据、数据预处理、特征处理

关联规则

#### 统计学习方法需要看的部分：

- 第一章：统计学习及监督学习概论
- **第三章：k近邻法**
- 第四章：朴素贝叶斯法
- **第五章：决策树**
- **第七章：支持向量机**
- **第八章：提升方法**
- **第十四章：聚类方法** 
- **第十七章：潜在语义分析**
