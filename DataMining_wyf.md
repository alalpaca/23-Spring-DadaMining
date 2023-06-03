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



#### 分类任务：

> ​                                                              Test Set 
>
> ​                                                                    ↓
>
> Trainning Set → **Learn Classifier**   →   Model

##### 分类任务举例

<img src="C:\Users\Kevin Wang\AppData\Roaming\Typora\typora-user-images\image-20230603144611456.png" alt="image-20230603144611456" style="zoom: 50%;" />

##### 分类应用

- 信用卡欺诈检测
- 电话客户的客户流失预测
- 宇宙探测

#### 回归任务：

> 回归任务最终得到的是一个值！

> 假设依赖项的线性或非线性模型，基于其他变量的值预测给定连续变量的值。
>
> 广泛用于统计研究和神经网络领域

##### 回归应用

- 基于广告支出预测新产品的销售额。
- 预测风速，作为温度、湿度、气压等的函数。
- 股票市场指数的时间序列预测。

#### 聚类任务：

> 查找对象组，使组中的对象彼此相似（或相关），并且与其他组中的对象不同（或与对象无关）

<img src="C:\Users\Kevin Wang\AppData\Roaming\Typora\typora-user-images\image-20230603145843152.png" alt="image-20230603145843152" style="zoom:50%;" />

##### 聚类应用

- 数据理解 -> 统计相似事物来理解其数据
- 数据总结 -> 减小大型数据集的大小
- 市场细分 -> 寻找类似客户的集群
- 文档聚类
  - 目标：根据文档中显示的重要术语查找相似的文档组
  - 方法：确定每个文档中频繁出现的术语，根据不同术语的频率形成相似度量，使用它进行群集

#### 关联规则

> 给定一组记录，每个记录都包括给定集合中的一些项数
>
> 生成依赖项规则，基于其他项的发生率预测项的发生

##### 关联分析应用

- 市场分析
- 电信报警诊断
- 医学信息学

#### 偏差/异常/变化检测

> 检测与正常行为的重大偏差

##### 应用

- 信用卡欺诈检测
- 网络入侵检测
- 识别传感器网络的异常行为，用于监控
- 检测全球森林覆盖的变化

#### 数据挖掘挑战

（可能会问）

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527105447083.png" alt="image-20230527105447083" style="zoom: 67%;" />

## 第二章 数据 

#### 2.1 数据属性与对象

##### 数据的定义

- 属性Attribute：对象的特性或特征
- 对象Object：属性的集合

##### 属性值与属性

- 相同的属性可以映射到不同的属性值
  - eg: 高度可以用英尺或米来表示
- 不同的属性可以映射到同一组值
  - eg: ID和age的属性值都是整数

##### 属性类别与属性值特性

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527105636202.png" alt="image-20230527105636202" style="zoom: 50%;" />



<img src="C:\Users\Kevin Wang\AppData\Roaming\Typora\typora-user-images\image-20230603192015538.png" alt="image-20230603192015538" style="zoom: 67%;" />



#### 离散和连续属性

- 离散属性

  > 只有一组有限的或可数无限的值

  - 示例：邮政编码、计数或文档中的一组单词（整数变量）
  - 二进制属性是离散属性的特例

- 连续属性

  > 将实数作为属性值

  - 示例：温度、身高或体重（浮点变量）
  - 实际值只能用有限的数字来测量和表示

#### 非对称属性

> 只有存在（非零属性值）才被视为重要
>
> **存在即正义**

- 示例：文档中出现的词语、客户交流记录中的项目

- 有时需要用两个不对称的二进制属性来表示一个普通的二进制属性
  - 关联分析使用非对称属性
  - 非对称属性通常产生于集合中的对象

#### 2.2 数据类型

##### 数据类型

- Record
  - 数据矩阵
  - 文件数据
  - 交易数据
- Graph
  - 万维网
  - 分子结构
- Ordered
  - 空间数据
  - 事态数据
  - 顺序数据
  - 遗传序列数据

##### 数据的重要特点

- 维度-Dimensionality
- 稀疏性-Sparsity
- 分辨率-Resolution
- 大小-Size

#### 2.3 数据的质量

影响数据质量的原因

- 噪声和离群点
  - 噪声
    - 对数据对象：与被测量对象没有直接联系的谬误对象
    - 对属性：测量误差的随机部分
    - 示例：通话质量较低产生的声音失真、电视机“雪花点”
  - 离群点
    - 对数据对象：不同于数据集中大部分数据对象特征的数据对象
    - 对数据属性：相比于该属性典型值来说不寻常的属性值
    - 离群点可能被当作噪声进行滤除或将其忽视
    - 离群点也可能作为被重点关注的对象或值 -> 信用卡欺诈、网络入侵
- 错误数据
- 虚假数据
- 遗漏值
  - 数据遗漏的原因
    - 信息收集不全
    - 部分属性不适合描述部分对象
  - 处理方法
    - 删除数据对象或属性
    - 估计遗漏值
    - 在分析时忽略遗漏值
- 重复数据
  - 并非所有重复数据都应被移除，多个对象的属性度量相同也可能是合法的

#### 2.4 相似度与距离



<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110218040.png" alt="image-20230527110218040" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110229117.png" alt="image-20230527110229117" style="zoom:67%;" />

##### 【计算】简单匹配系数

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110252391.png" alt="image-20230527110252391" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110341263.png" alt="image-20230527110341263" style="zoom:67%;" />



##### 【计算】相关性 《稍微看一看》

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110507542.png" alt="image-20230527110507542" style="zoom: 67%;" />

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527110451604.png" style="zoom: 67%;" />

##### 熵

> 实际表示的是记录时间结果所需要的bits
>
> 大小在0~log2n之间

$$
H(X)=-\sum_{i=1}^{n} p_{i} \log _{2} p_{i}
$$

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527111913314.png" alt="image-20230527111913314" style="zoom:67%;" />

##### 互信息

> 一组值对另一组提供多少信息的度量
$$
H(X)=H(X)+H(Y)-H(X,Y)
$$
其中,H(X,Y)为X和Y的联合熵：
$$
H(X,Y)=-\sum_{i}\sum_{j} p_{ij} log_{2} p_{ij}
$$

##### 邻近度差异适用范围分析：

- 简单匹配系数SMC：只含有二元属性的两个对象的相似度
- （广义）Jaccard系数：用于文档数据
- 余弦相似度：统计文档数据中的词频相似
- 相关性：测量两组被观测值之间的线性关系，可以是不同变量或不同对象
- 欧几里得距离：注重数值（大小）

### 2.5 数据预处理

（好好看一下）

##### 聚集

> 将两个或多个对象合并成单个对象

**聚集的目的：**

- 减少数据量，从而减小数据存储空间及处理时间；
- 范围或标度转换
  - 将不同城市按照行政区、州、国家等进行聚集
  - 将日期按照星期、月份、年份等进行聚集
  
- 提供更加“稳定的” 数据
  - 聚集的数据往往具有更小的变异性


##### 抽样

> 减少数据量的常用方法
>
> 在统计学中，常用于实现调查和最终数据的分析

**抽样的意义**：

- 对统计学家来说，获取感兴趣的整个数据集代价太高，且太费时间
- 对数据挖掘人员来说，处理所有数据所需的内存太大、时间太长、计算成本太高

**抽样方法**：

- 简单随机抽样
  - 等概率抽样
  - 有放回抽样较简单、无放回抽样理论误差较低

- 分层抽样
  - 从不同类型的对象中随机抽取样本

- 渐进抽取

##### 降维

**降维的意义**：

- 减少数据量，从而减小数据存储空间及处理时间
- 避免维灾难
  - 维灾难：随着维度增加，数据在其所在空间中越来越稀疏
  - 而维度太小也会导致难以对不同点的类进行划分

- 消除无关变量
- 有利于模型的理解及数据可视化

**降维方法**：

- 奇异值分解(SVD)
- 主成分分析(PCA)
  - 原数据映射到具有最大方差的方向上

- 因子分析(FA)
- 独立成分分析(ICA)。

##### 特征子集选择

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527115712428.png" alt="image-20230527115712428" style="zoom:67%;" />

##### 特征创建

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527115820872.png" alt="image-20230527115820872" style="zoom:67%;" />

##### 离散化与二值化

1. 离散化

- > 将连续属性（区间、比率）变换为序数属性的过程

  - 理论上无限个数的值可以映射到有限的几个类别中
  - 常用于分类任务，合并某些值减少类别数量

- 以下为鸢尾花的离散化样例

<img src="C:\Users\Kevin Wang\AppData\Roaming\Typora\typora-user-images\image-20230604004900567.png" alt="image-20230604004900567" style="zoom:50%;" />

<img src="C:\Users\Kevin Wang\AppData\Roaming\Typora\typora-user-images\image-20230604004833711.png" alt="image-20230604004833711" style="zoom:50%;" />

- 离散化分类
  - 无监督离散化：根据数据分布确定分割点（如上例）
  - 有监督离散化：使用类信息（标签）确定分割点

2. 二值化

- > 将连续或分类属性映射到一个或多个二进制变量中

  - 通常连续转分类，分类再转二进制
  - 常用于关联分析，需要分析是否存在非对称的二元属性
  - eg：眼睛颜色深度和高度通常表示为{低、中、高}，可表示为{1 0 0、0 1 0 、0 0 1}

##### 属性变换

<img src="https://raw.githubusercontent.com/SHUzhekiNg/SHUzhekiNg.github.io/main/assets/typoraimages/image-20230527120007251.png" alt="image-20230527120007251" style="zoom:67%;" />



## 第三章 分类：基本概念和技术

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
