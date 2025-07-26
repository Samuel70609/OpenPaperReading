K这篇论文提出了一个基于 **专家乘积（Product of Experts, PoE）** 的框架，用于高效解决LLM比较评估中的计算瓶颈问题（传统方法计算量随候选文本数呈平方级增长）。

---

### **1. 问题定义**
- **输入**：`N`个候选文本 $x_{1:N}$，`K`个成对比较结果 $\mathcal{C}_{1:K}$。
- **每个比较**：形式为 $(i, j, p_{ij})$，其中 $p_{ij} = P_{\text{LM}}(x_i \succ x_j)$ 是LLM判断$x_i$优于$x_j$的概率。
- **目标**：预测文本的**隐式质量分数** $\hat{s}_{1:N}$，使其接近真实分数 $s^*_{1:N}$。

---

### **2. 核心公式推导与含义**
#### **(1) PoE框架基础 (公式3)**
$
\mathrm{p}(s_{1:N}|\mathcal{C}_{1:K}) = \frac{1}{Z} \prod_{(i,j) \in \mathcal{C}} \mathrm{p}(s_i - s_j | p_{ij})
$
- **推导**：将每个比较 $(i,j)$ 视为一个**独立专家**，提供关于分数差 $s_i - s_j$ 的信息（并不是概率，而是一个概率分布）。
- **物理意义**：最大化该概率等价于找到最符合所有比较结果的分数分布。
- **关于分数差的理解**：概率信息是 LLM 对成对文本质量相对优劣的直接输出，而分数差是模型为了整合这些概率信息、推断文本绝对质量分数而引入的潜在变量
---

#### **(2) 正态专家模型 (公式4-8)**
假设分数差服从正态分布：
$
\mathrm{p}(s_i - s_j | p_{ij}) = \mathcal{N}\big( s_i - s_j; f_\mu(p_{ij}), f_\sigma(p_{ij}) \big)
$
- **矩阵形式**（公式4）：
  $
  \mathrm{p}(\mathbf{W}\mathbf{s} | \mathcal{C}) = \mathcal{N}\big( \mathbf{W}\mathbf{s}; \boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2) \big)
  $
  - $\mathbf{W} \in \mathbb{R}^{K \times N}$：**比较矩阵**。每行对应一次比较，例如比较$(i,j)$时，$W_{ki}=1, W_{kj}=-1$，其余为0（附录A.1）。
  - $\boldsymbol{\mu}, \boldsymbol{\sigma}^2$：由 $f_\mu(p_{ij})$ 和 $f_\sigma(p_{ij})$ 构成的向量。
  -  $\text{diag}(\boldsymbol{\sigma}^2)$ 表示由方差向量 $\boldsymbol{\sigma}^2$ 构成的对角矩阵。

- **分数分布的闭式解**（公式7-8）：
  $
  \boldsymbol{\mu}^*_{\mathbf{s}} = (\tilde{\mathbf{W}}^\top \tilde{\boldsymbol{\Sigma}}^{-1} \tilde{\mathbf{W}})^{-1} \tilde{\mathbf{W}}^\top \tilde{\boldsymbol{\Sigma}}^{-1} \tilde{\boldsymbol{\mu}}
  $
  $
  \boldsymbol{\Sigma}^*_{\mathbf{s}} = (\tilde{\mathbf{W}}^\top \tilde{\boldsymbol{\Sigma}}^{-1} \tilde{\mathbf{W}})^{-1}
  $
  - **推导**：通过贝叶斯定理将正态分布转化为关于 $\mathbf{s}$ 的后验分布（附录A.4）。
  - **物理意义**：分数的最优估计 $\hat{\mathbf{s}} = \boldsymbol{\mu}^*_{\mathbf{s}}$ 是**加权最小二乘解**，权重由方差 $\boldsymbol{\sigma}^2$ 决定（方差越小，比较结果越可信）。

  - **$\tilde{\mathbf{W}}$（扩展比较矩阵）**：  
    维度为“（比较数+1）×文本数”，每行对应一个比较（或额外的约束行）。例如，若第$k$行对应文本$i$与$j$的比较，则$W_{ki}=1$、$W_{kj}=-1$，其余元素为0（表示仅关注$i$与$j$的分数差）；额外的约束行用于解决分数估计的偏移问题（如第一行可能为$[1, 0, 0, ..., 0]$，固定首个文本的分数基准）。  

  - **$\tilde{\boldsymbol{\Sigma}}^{-1}$（精度矩阵）**：  
    是方差矩阵$\tilde{\boldsymbol{\Sigma}}$的逆矩阵，$\tilde{\boldsymbol{\Sigma}}$为对角矩阵（$\text{diag}(\tilde{\sigma}^2)$），对角线元素为每个比较的方差（反映该比较的可信度，方差越小可信度越高）。因此$\tilde{\boldsymbol{\Sigma}}^{-1}$的对角线元素为方差的倒数，本质是“权重”——方差越小，权重越大。  
   - $\tilde{\mathbf{W}}$的原始维度是“比较数×文本数”，每行代表一个比较对文本分数差的约束（如$W_{k} \cdot \mathbf{s} = s_i - s_j$，即第$k$个比较的分数差）。  
   - 转置后$\tilde{\mathbf{W}}^\top$的维度为“文本数×比较数”，其作用是将“比较对分数差的约束”转换为“文本分数对比较的贡献”，以便与权重矩阵$\tilde{\boldsymbol{\Sigma}}^{-1}$（比较数×比较数）和均值向量$\tilde{\boldsymbol{\mu}}$（比较数×1）相乘，最终整合所有比较的信息。  


        1. $\tilde{\mathbf{W}}^\top \tilde{\boldsymbol{\Sigma}}^{-1}$：将比较的权重（$\tilde{\boldsymbol{\Sigma}}^{-1}$）与文本贡献（$\tilde{\mathbf{W}}^\top$）结合，得到“文本数×比较数”的矩阵，表示每个文本在所有加权比较中的影响力。  
        2. 再乘以$\tilde{\mathbf{W}}$：得到“文本数×文本数”的矩阵，刻画文本之间通过比较形成的关联（如共同参与比较的文本会产生关联）。  
        3. 求逆$(\cdot)^{-1}$：解出加权方程组，消除文本间的关联干扰。  
        4. 最后乘以$\tilde{\mathbf{W}}^\top \tilde{\boldsymbol{\Sigma}}^{-1} \tilde{\boldsymbol{\mu}}$：将所有比较的加权均值（$\tilde{\boldsymbol{\mu}}$）映射为每个文本的全局分数估计。  

---

#### **(3) 实用简化：线性正态专家 (公式11)**
为免人工标注，假设：
1. 方差恒定：$f_\sigma(p) = \sigma^2$
2. 均值线性：$f_\mu(p) = \alpha (p - \beta)$
解简化为：
$
\hat{\mathbf{s}} = \alpha (\tilde{\mathbf{W}}^\top \tilde{\mathbf{W}})^{-1} \tilde{\mathbf{W}}^\top \tilde{\boldsymbol{\mu}}
$
- **参数选择**：
  - $\beta = 0.5$：假设LLM无偏时，等质量文本的 $p_{ij}=0.5$。
  - $\alpha = 1$：仅缩放分数范围，不影响排序。
- **物理意义**：解等价于**最小化比较结果的平方误差**（附录A.3证明全比较集下与平均概率法等价）。

---

#### **(4) 位置偏差修正 (公式12)**
LLM可能存在位置偏好（如倾向选择第一个文本）：
$
\mathbb{E}[s_i - s_j] = \alpha (\mathbb{E}[p_{ij}] - \beta) = 0 \quad \Rightarrow \quad \beta = \mathbb{E}[p_{ij}]
$
- **修正方法**：用所有比较的平均概率 $\bar{p}$ 估计 $\beta$，调整均值函数为 $f_\mu(p) = \alpha (p - \bar{p})$。
- **效果**：消除系统偏差，提升零样本性能（图7验证）。

---

#### **(5) 主动比较选择 (公式16-17)**
为最大化信息量，选择能最小化分数方差的比较对：
$
(i^*, j^*) = \arg\max_{i,j} \left( \mathbf{A}_{ii}^{(k)} + \mathbf{A}_{jj}^{(k)} - 2\mathbf{A}_{ij}^{(k)} \right)
$
- **推导**：基于正态分布的熵（公式13），正态分布分数分布的概率与$\det(\tilde{W}^\top \tilde{W})$成正比。最大化 $\det(\tilde{\mathbf{W}}^\top \tilde{\mathbf{W}})$。
- 其中$\mathbf{A}^{(k)} = (\tilde{W}^{(k)*\top} \tilde{W}^{(k)*})^{-1}$是当前比较矩阵逆矩阵，反映了已有比较下分数估计的不确定性
- **物理意义**：文本i与j的分数差的方差，选择**分数不确定且差异大**的文本对（$\mathbf{A}_{ii}$ 大表示 $s_i$ 方差大，$\mathbf{A}_{ij}$ 负值表示 $s_i, s_j$ 强相关）。
- **高效更新**：使用Sherman-Morrison公式迭代更新逆矩阵 $\mathbf{A}^{(k)}$（公式40）。

---

### **3. 关键结论与优势**
1. **计算效率**：
   - 正态PoE提供**闭式解**，避免迭代优化（如Bradley-Terry需Zermelo算法）。
   - 仅需 $O(N)$ 次比较（而非 $O(N^2)$）即可接近全比较集性能（图1, 图3）。

2. **灵活性**：
   - 支持任意专家模型（正态/Bradley-Terry）。
   - 兼容软概率 ($p_{ij}$) 和硬决策（Win-Ratio）。

3. **性能提升**：
   - 在多个NLG任务（SummEval, HANNA等）上，PoE在低比较次数下显著优于基线（表1, 表3）。
   - 主动选择策略进一步提升效果（表2）。

---
