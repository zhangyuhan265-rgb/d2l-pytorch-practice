# PyTorch / d2l 速查表

> 用法：**`Ctrl+Shift+O`**（转到符号）→ 直接打函数名 → 回车。
> 这是"输入即过滤"的函数名搜索，不用切窗口。
> 搜正文（例子、坑）用 **`Ctrl+F`**。
>
> 全部条目按你 d2l 练习文件里**实际用过的** API 整理，行为都已实机验证。

---

## 目录

1. [张量创建](#1-张量创建)
2. [形状与变换](#2-形状与变换)
3. [运算](#3-运算)
4. [索引与取值](#4-索引与取值)
5. [自动微分](#5-自动微分)
6. [神经网络层与损失](#6-神经网络层与损失)
7. [数据加载](#7-数据加载)
8. [优化器](#8-优化器)
9. [设备（CPU/GPU）](#9-设备cpugpu)
10. [Python 内置函数](#10-python-内置函数)
11. [自己写过的函数索引](#11-自己写过的函数索引)
12. [形状速查表](#12-形状速查表)
13. [踩过的坑（浓缩版）](#13-踩过的坑浓缩版)
14. [自己加条目](#14-自己加条目)

---

# 1 张量创建

## `torch.tensor(data)`

> 从 Python 列表/标量造张量。

```python
torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # → (2,2) float32
torch.tensor([1, 2, 3])                  # → int64（全是整数就是整型！）
```

⚠️ 元素里只要有一个浮点，整张表就升级成 `float32`；全是整数则是 `int64`。
⚠️ `softmax` / `exp` / `log` 都不吃 `int64`，会报 `not implemented for 'Long'`。

---

## `torch.zeros(*size)` / `torch.ones(*size)`

> 全 0 / 全 1 张量。

```python
torch.zeros(2, 3)      # → (2,3) 全 0，float32
torch.ones(1)          # → (1,) 全 1
```

⚠️ 生成的张量默认 `requires_grad=False`，拿来当参数要再包 `nn.Parameter`。

---

## `torch.zeros_like(t)` / `torch.ones_like(t)`

> 照 `t` 的形状和 dtype 造全 0 / 全 1。

```python
t = torch.randn(2, 3)
torch.zeros_like(t)    # → (2,3)，和 t 同 dtype
```

⭐ 比 `torch.zeros(*t.shape)` 更稳：dtype 和设备都跟着 `t` 走。

---

## `torch.randn(*size)`

> 标准正态分布 `N(0,1)`。

```python
torch.randn(2, 3)      # → (2,3)，每个数 ~N(0,1)
```

⚠️ **不是** 0~1 均匀分布（那是 `torch.rand`）。约一半是负数。
⚠️ 每次调用都不一样，要复现先 `torch.manual_seed()`。

---

## `torch.normal(mean, std, size)`

> 指定均值与标准差的正态分布。

```python
torch.normal(0, 0.01, (2, 1))    # → 标准差 0.01，比 randn 小 100 倍
```

⭐ d2l 3.2 手写初始化就用它：`torch.normal(0, 0.01, size=(2,1), requires_grad=True)`。

---

## `torch.arange(start, end, step)`

> 等差序列（不含 `end`）。

```python
torch.arange(12)                  # → [0..11]
torch.arange(0, 1, 0.2)           # → [0, .2, .4, .6, .8]
```

⚠️ 默认 dtype 是 `int64` 当参数全为整数；要和浮点运算得 `.float()`。

---

## `torch.manual_seed(n)`

> 固定随机数种子，让结果可复现。

```python
torch.manual_seed(42)
torch.randn(2)    # 每次跑都一样
```

⚠️ 是**全局状态**：设一次后，之后所有随机操作都按这个序列走。

---

## `torch.randperm(n)`

> `0..n-1` 的随机排列。

```python
torch.randperm(5)      # → 例：[3, 0, 4, 1, 2]
```

⭐ `DataLoader(shuffle=True)` 底层就是它。

---

# 2 形状与变换

## `.shape`

> 张量各维大小。

```python
x = torch.randn(4, 3)
x.shape          # → torch.Size([4, 3])
tuple(x.shape)   # → (4, 3)   ← 打印时更干净
```

---

## `.numel()`

> 元素总数（`numel` = number of elements）。

```python
torch.zeros(2, 3, 4).numel()    # → 24
```

⭐ 数参数量：`sum(p.numel() for p in model.parameters())`。

---

## `.reshape(*shape)`

> 改形状，元素总数必须一致；**必要时自动拷贝**。

```python
x = torch.arange(12)
x.reshape(3, 4)      # → (3,4)
x.reshape(-1, 4)     # 老写 -1 让它自己算：→ (3,4)
```

⚠️ 只检查元素总数，**不检查语义**。`(12,)` 能 reshape 成 `(3,4)` 也能成 `(4,3)`，但含义完全不同。

---

## `.view(*shape)`

> 改形状，但**要求内存连续**，不拷贝。

```python
x = torch.arange(12).reshape(3, 4)
x.view(-1)         # ✅ 连续，能过
x.T.view(-1)       # ❌ RuntimeError: view size is not compatible ...
x.T.reshape(-1)    # ✅ 成功（reshape 会拷贝）
```

⭐ 实践：**不确定就一律用 `reshape`**，不会因为不连续而炸。

---

## `.flatten(start_dim=0)`

> 把一段维度拉平成一维。

```python
x = torch.randn(2, 3, 4)
x.flatten()            # → (24,)
x.flatten(1)           # → (2, 12)   只拉平第 1 维往后
```

⚠️ 和 `squeeze` 不同：`flatten` 是"合并维度"，`squeeze` 是"删掉大小为 1 的维度"。

---

## `.squeeze(dim)` / `.unsqueeze(dim)`

> 删掉 / 插入大小为 1 的维度。

```python
x = torch.randn(1, 3, 1)
x.squeeze()        # → (3,)      删掉所有 1 维
x.squeeze(0)       # → (3, 1)    只删第 0 维
y = torch.randn(3)
y.unsqueeze(0)     # → (1, 3)
y.unsqueeze(1)     # → (3, 1)
```

⚠️ `(n,)` 和 `(n,1)` 相减会**广播成 `(n,n)`**，不报错但全错。这是最常见的静默 bug。

---

## `.T`

> 转置（≤2 维是真正的转置）。

```python
x = torch.randn(2, 3)
x.T.shape          # → (3, 2)
```

⚠️ 3 维以上 `.T` 会把**所有**维度倒序，通常不是你想要的；多维也转置用 `.transpose(i, j)`。

---

## `torch.cat(tensors, dim)`

> 在**已有维度**上拼接。

```python
a = torch.zeros(2, 3)
b = torch.ones(2, 3)
torch.cat([a, b], dim=0)    # → (4, 3)   行数相加
torch.cat([a, b], dim=1)    # → (2, 6)   列数相加
```

---

## `torch.stack(tensors, dim)`

**新建一个维度**再堆叠。

```python
a = torch.zeros(2, 3)
b = torch.ones(2, 3)
torch.stack([a, b], dim=0)  # → (2, 2, 3)   多了最外层的"第几个"
```

⭐ 记法：`cat` = 接长，`stack` = 叠高。

---

# 3 运算

## `@` / `torch.matmul(a, b)`

> 矩阵乘法。

```python
A = torch.randn(5, 2)
B = torch.randn(2, 1)
A @ B               # → (5, 1)
torch.matmul(A, B)  # 同上
```

⭐ 规则：`(m,k) @ (k,n) = (m,n)`；前面的维度按广播走。
⭐ `A @ B` 就是 `F.linear(A, W, b)` 的 `A @ W.T + b` 那部分。

---

## `torch.dot(a, b)`

> 两个**一维**向量的点积，返回标量。

```python
torch.dot(torch.tensor([1., 2.]), torch.tensor([3., 4.]))   # → 11.0
```

⚠️ 只接受一维；二维要用 `@` 或 `(a*b).sum()`。

---

## `.sum(dim=None, keepdim=False)` / `.mean(...)`

> 求和 / 求平均。

```python
x = torch.arange(6, dtype=torch.float32).reshape(2, 3)

x.sum()                        # → 15.0          标量，shape ()
x.sum(dim=1)                   # → [3., 12.]     shape (2,)
x.sum(dim=1, keepdim=True)     # → shape (2,1)   ← 保住那一维
x.mean()                       # → 2.5
```

⚠️ `.sum()` 不给 `dim` 是把**所有元素**加起来。二维里想按行求和必须 `sum(dim=1, keepdim=True)`，否则形状不对、广播出错。

---

## `torch.exp(x)` / `torch.log(x)`

> 逐元素 `e^x` / 自然对数 `ln(x)`。

```python
torch.exp(torch.tensor(2.0))    # → 7.389056   = e²
torch.log(torch.tensor(math.e)) # → 1.0        ← 底数是 e
```

⚠️ `torch.exp(89)` 在 float32 下直接 `inf`（上限约 3.4e38）。
⚠️ `torch.log(0)` → `-inf`。
⭐ 这就是 softmax 必须"减最大值"、交叉熵要用 `log_softmax` 的原因。

---

## `.norm(p=2, dim=None)`

> 范数。

```python
torch.tensor([3., 4.]).norm()      # → 5.0     L2
torch.tensor([3., 4.]).norm(1)     # → 7.0     L1
```

⭐ 看梯度大小常用 `p.grad.norm().item()`。

---

## `.clamp(min, max)`

> 把值夹在区间内。

```python
torch.tensor([-2., 0., 5.]).clamp(0, 3)    # → [0., 0., 3.]
```

---

## `x.argmax(dim)`

> 沿某一维找**最大值的索引（位置）**，不是最大值本身。

```python
y_hat = torch.tensor([[0.1, 0.3, 0.6],
                      [0.3, 0.2, 0.5]])

y_hat.argmax(dim=1)          # → tensor([2, 2])    ← 第几类（索引）
y_hat.max(dim=1).values      # → tensor([0.6, 0.5]) ← 多大的概率（数值）
y_hat.max(dim=1).indices     # → tensor([2, 2])    ← 和 argmax 等价
```

⚠️ 混淆 `argmax` 和 `max` 是常见错误：**准确率要索引，看置信度要数值**。
⚠️ `dim` 指向类别那一维：`(n, C)` 用 `dim=1`。
⚠️ 有并列时取**第一个**最大值的位置。
⚠️ torch 接受 `axis=` 当别名，但官方签名是 `dim`。

---

## `torch.allclose(a, b, atol=1e-8)`

> 浮点张量是否"几乎相等"。

```python
torch.allclose(torch.tensor([1.0]), torch.tensor([1.0 + 1e-9]))   # → True
```

⭐ **比较浮点数一律用它**，不要用 `a == b`。逐位相等才用 `torch.equal`。

---

## `torch.equal(a, b)`

> 形状和每个元素**完全**相同。

```python
torch.equal(torch.tensor([1., 2.]), torch.tensor([1., 2.]))   # → True
```

---

## 广播（broadcasting）规则

> 从**最右边**的维度开始对齐，两条兼容条件：
- 两个维度相等，或
- 其中一个是 1，或不存在

```python
(5, 2) + (2,)      # → (5,2)      (2,) 当作 (1,2) 再扩展
(5, 1) - (5,)      # → (5,5)      ⚠️ 不报错但几乎总是错的
(2, 3) * (2, 1)    # → (2,3)      ✅ 正常的按行缩放
```

⚠️ 看到形状和预期不符，第一件事是 `print(x.shape)`。

---

# 4 索引与取值

## 基础索引 `x[i]` / `x[i, j]` / `x[:, j]`

```python
x = torch.randn(3, 4)
x[0]        # → (4,)    第 0 行
x[0, 1]     # → 标量    第 0 行第 1 列
x[:, 1]     # → (3,)    所有行的第 1 列
x[0:2]      # → (2,4)   切片
```

⚠️ 下标从 **0** 开始，第 n 个元素的下标是 n-1。

---

## 高级索引 `x[row_idx, col_idx]`

> 两个索引都是序列时，**按下标位置配对成坐标**：取 `(a[0],b[0])`、`(a[1],b[1])`……

```python
y_hat = torch.tensor([[0.1, 0.3, 0.6],
                      [0.3, 0.2, 0.5]])
y = torch.tensor([0, 2])

y_hat[[0, 1], y]        # → tensor([0.1, 0.5])
#   配对过程：(0, y[0]=0) → y_hat[0,0] = 0.1
#             (1, y[1]=2) → y_hat[1,2] = 0.5
```

**等价于手写：**

```python
torch.tensor([y_hat[0, 0], y_hat[1, 2]])    # → [0.1, 0.5]
```

**三种写法差别很大：**

| 写法 | 结果 | 形状 |
|---|---|---|
| `y_hat[[0,1], y]` | `[0.1, 0.5]` | `(2,)` ✅ 要的 |
| `y_hat[[0,1]]` | 两个整行 | `(2,3)` 子矩阵 |
| `y_hat[:, y]` | 每行取第 0、2 列 | `(2,2)` ⚠️ 常见错误 |

⚠️ 既不是"子矩阵"，也不是"笛卡尔积"，是**按位置配对**。
⚠️ 高级索引返回**拷贝**，切片返回**视图**（`data_ptr()` 不同）。
⭐ 两个索引形状不同时会广播：`x[[[0],[1]], [0,2]]` → `(2,2)`。
⭐ 3.6 手写交叉熵就用它：`-torch.log(y_hat[range(len(y_hat)), y])`，`range(n)` 正好和样本下标对齐。

---

## 布尔掩码

> 拿一个布尔张量当索引，取出为 `True` 的那些元素（同样返回拷贝，形状会变一维）。

```python
x = torch.tensor([-1., 2., -3.])
x[x > 0]              # → tensor([2.])
(x > 0).sum()         # → 1        True 的个数
```

---

# 5 自动微分

## `requires_grad=True`

> 标记这个张量需要求导。

```python
w = torch.tensor([1.0], requires_grad=True)
```

⚠️ 只有**叶子张量**能直接设；对运算结果设要用 `.requires_grad_(True)` 或 `.retain_grad()`。

---

## `.backward()`

> 从标量反向传播，把梯度写进各张量的 `.grad`。

```python
loss = (w * 2).sum()
loss.backward()
w.grad            # → tensor([2.])
```

⚠️ 只能对标量调用；向量要传 `gradient=` 参数。

---

## `.grad`

> 累积的梯度，**`backward()` 之前是 `None`**。

```python
w.grad is None     # True（还没 backward）
...
w.grad             # 有值
```

⚠️ **是累加（`+=`）不是覆盖**，所以每轮必须清零，否则梯度越滚越大。
⚠️ `.grad` 是 `Optional`，取值前最好 `assert w.grad is not None`（Pylance 也需要）。

---

## `torch.no_grad()`

> 在这个上下文里不记录计算图（省内存、提速）。

```python
with torch.no_grad():
    linear.weight.normal_(0, 0.01)    # 直接改值，不进图
```

⭐ 手动改参数、做评估、算指标时都要包一层。

---

## `.detach()`

> 返回一个**共享存储**但脱离计算图的张量。

```python
d = w.detach()
d.requires_grad        # → False
d.data_ptr() == w.data_ptr()   # → True（同一块内存）
```

---

## `.clone()`

> 复制一份**独立内存**。

```python
c = w.clone()
c.requires_grad        # → True（保留图关系）
c.data_ptr() == w.data_ptr()   # → False
```

⚠️ `g = w.grad` 只是别名；`g = w.grad.clone()` 才是快照。
⭐ 想"留住这一刻的梯度"就必须 clone。

---

## `.zero_grad()`

> 清零梯度。

```python
optimizer.zero_grad()
# 或
w.grad = None
```

⚠️ PyTorch 2.x 默认 `set_to_none=True` → 清完 `.grad` 是 **`None`**，不是全 0 张量。
⚠️ 位置：必须在 `backward()` **之前**，不能在 `backward → step` 之间。

---

## `nn.Parameter(tensor)`

> 把张量包成"可训练参数"。

```python
self.scale = nn.Parameter(torch.tensor(1.0))
```

⭐ 只有包成 `Parameter` 或赋给 `nn.Module` 的属性，才会自动进入 `parameters()`。

---

# 6 神经网络层与损失

## `nn.Linear(in_features, out_features, bias=True)`

> 全连接层：`y = x @ W.T + b`。

```python
linear = nn.Linear(2, 1)
linear.weight.shape    # → (1, 2)   ← (out, in)！转置的
linear.bias.shape      # → (1,)
```

⚠️ **`weight` 是 `(out, in)`**，与手写版 `(in, out)` 互为转置。
⚠️ 形状规则：最后一维 = `in_features`，输出最后一维 = `out_features`，前面的维度不变。

---

## `nn.Sequential(*layers)`

> 按顺序串起来的容器。

```python
net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
len(net)     # → 3
net[0]       # → 第 0 层（类型是 Module，没有 .weight 的类型信息）
```

⚠️ **存引用不存副本**：`net[0] is linear` → `True`。
⚠️ 只能表达"一条直线"，分支/残差做不到。
⭐ 只有 1 层时纯属多余；2 层以上才开始省代码。

---

## `nn.ReLU()` / `nn.Flatten()`

> 无参数的层：激活函数 / 拉平。

```python
nn.ReLU()(torch.tensor([-1., 2.]))        # → [0., 2.]
nn.Flatten()(torch.randn(2, 1, 28, 28))   # → (2, 784)
```

⚠️ 它们**没有参数**，但**仍然要登记进 `_modules`**，否则 forward 找不到。

---

## `nn.MSELoss()`

> 均方误差 `mean((ŷ-y)²)`。

```python
nn.MSELoss()(torch.tensor([1., 2.]), torch.tensor([1., 3.]))   # → 0.5
```

⚠️ **没有 1/2**，所以梯度和 loss 都是 d2l `squared_loss(...).mean()` 的 **2 倍**。

---

## `nn.CrossEntropyLoss()`

> 分类损失（内部 = `log_softmax` + 取正确类 + 取负）。

```python
logits = torch.tensor([[2.0, 1.0, 0.1]])     # (1, 3)
target = torch.tensor([0])                    # (1,)  int64
nn.CrossEntropyLoss()(logits, target)         # → 0.417030
```

⚠️⚠️ **只喂 logits，不喂概率！** 喂概率不报错，但内部会**再 softmax 一次**，结果静默算错。
⚠️ `target` 必须是 `(N,)` 的 `int64`，**不要**手动转 one-hot。

---

## `F.softmax(x, dim)` / `nn.Softmax(dim)`

> 沿指定维归一化成概率。

```python
F.softmax(torch.tensor([[1., 2., 3.]]), dim=1)   # → [[0.09, 0.245, 0.665]]
```

⚠️ `dim` 要指向**类别**那一维：`(N, C)` 用 `dim=1`。
⚠️ 形状不变（不降维），这和 `sum(dim=...)` 不同。
⚠️ 保序：`argmax(softmax(o)) == argmax(o)`。

---

## `F.log_softmax(x, dim)`

> 先 softmax 再取 log，但数值更稳。

```python
F.log_softmax(torch.tensor([[1., 2., 3.]]), dim=1)   # → [[-2.407, -1.407, -0.407]]
```

⭐ 比 `torch.log(F.softmax(x))` 稳：不会"先算出 0 再取 log"得到 `-inf`。

---

## `F.one_hot(tensor, num_classes)`

> 整数标签 → one-hot。

```python
F.one_hot(torch.tensor([2]), num_classes=3)    # → [[0, 0, 1]]
```

⚠️ 交叉熵**内部自己会转**，你一般不需要手动 one-hot。

---

## `.parameters()` / `.named_parameters()`

> 遍历所有参数（递归）。

```python
list(model.parameters())          # → [Parameter, Parameter]
dict(model.named_parameters())    # → {'0.weight': ..., '0.bias': ...}
```

⚠️ 返回 **generator**，只能遍历一次；递的是**引用**不是副本。
⭐ `optim.SGD(model.parameters(), lr)` 内部会 `list()` 存起来。

---

## `net.apply(fn)`

> 把 `fn` **递归**应用到模型自身和每个子模块上（顺序：先子模块，最后才是自己）。

```python
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

**实测遍历顺序**（`Sequential(Flatten(), Linear(784,10))`）：

```
第 1 次：Flatten
第 2 次：Linear      ← 只有这个会被初始化
第 3 次：Sequential
```

⚠️ `apply` 返回 **self**，所以 Jupyter 里不加分号会打印整个网络结构 → 末尾加 `;`。
⚠️ `type(m) == nn.Linear` 匹配不到**子类**（自定义继承 `nn.Linear` 的层）。用 `isinstance` 更稳。
⭐ 常见用法就是"统一初始化"：`net.apply(init_weights)`。

---

## `nn.init.normal_(tensor, mean=0, std=1)`

> **原地**把张量填成 `N(mean, std²)`。名字带下划线 = in-place。

```python
nn.init.normal_(m.weight, std=0.01)     # 就地改成 std=0.01 的正态
```

**对比 `nn.Linear` 的默认初始化：**

| | 分布 | std |
|---|---|---|
| `nn.Linear(784,10)` 默认 | 均匀 `U(±1/√784)` | 0.0205 |
| `nn.init.normal_(w, std=0.01)` | 正态 `N(0,0.01²)` | 0.0101 |

⭐ d2l 用它把 3.7 的初始化对齐 3.6 手写版（`torch.normal(0, 0.01, ...)`），两节训练曲线才可比。
⭐ 同族还有 `nn.init.zeros_` / `ones_` / `constant_` / `xavier_uniform_` / `kaiming_normal_`。

---

## `.state_dict()`

> 参数快照（`OrderedDict`）。

```python
model.state_dict().keys()    # → odict_keys(['0.weight', '0.bias'])
torch.save(model.state_dict(), 'm.pth')
```

⚠️ 里面是 `detach` 后的视图，**仍是同一块内存**，不是副本。

---

# 7 数据加载

## `data.TensorDataset(X, y)`

> 把特征和标签按第 0 维绑成对。

```python
ds = data.TensorDataset(features, labels)
len(ds)      # → 样本数
ds[0]        # → (features[0], labels[0])
```

⭐ 保证"取第 i 个永远拿到同一对"，不会错位。

---

## `data.DataLoader(dataset, batch_size, shuffle=True)`

> 分批迭代器。

```python
dl = data.DataLoader(ds, 10, shuffle=True)
len(dl)                       # → ceil(样本数 / batch_size)
X, y = next(iter(dl))         # → 只取第一批
```

⚠️ `shuffle=True` 打乱的是**样本行顺序**，不拆列、不改原张量。
⚠️ 除不尽时最后一批会小一些（`drop_last=True` 可丢掉）。
⚠️ `DataLoader` 自己是可迭代对象，但没有 `__next__` → `next(dl)` 会 `TypeError`，要先 `iter(dl)`。

---

## `transforms.ToTensor()`

> PIL 图像 → 张量，同时改数值范围。

```python
transforms.ToTensor()(pil_img)
# (28,28) uint8 [0,255]  →  (1,28,28) float32 [0,1]
```

⚠️ 它同时改**类型、形状、数值范围**三件事，忘了它后面全错。

---

## `transforms.Resize(size)` / `transforms.Compose([...])`

> 缩放 / 串联多个变换。

```python
trans = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])
```

⚠️ `Compose` 按**列表顺序**执行：`Compose([A,B])(x) == B(A(x))`。

---

# 8 优化器

## `torch.optim.SGD(params, lr, momentum=0)`

> 随机梯度下降。

```python
trainer = torch.optim.SGD(model.parameters(), lr=0.03)
trainer.zero_grad()
loss.backward()
trainer.step()
```

⚠️ **不除 batch_size**（d2l 手写 `sgd` 里除）。
⭐ 想用 `l.sum()` 配 `optim.SGD` 时，步长会被放大 `batch_size` 倍。

---

## `torch.optim.Adam(params, lr)`

> 自适应学习率优化器。

```python
torch.optim.Adam(model.parameters(), lr=1e-3)
```

---

## `optimizer.param_groups`

> 优化器的参数组（想中途改学习率就改这里）。

```python
trainer.param_groups[0]['lr'] = 0.01     # ✅ 立即生效
```

---

# 9 设备（CPU/GPU）

## `torch.cuda.is_available()`

> 有没有可用的 GPU。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

## `.to(device)` / `.device`

> 搬运张量/模型。

```python
x = x.to(device)
model = model.to(device)
x.device     # → device(type='cuda', index=0)
```

⚠️ 模型和张量必须在**同一个 device**，否则报错。

---

# 10 Python 内置函数

> 这些**不是 torch 的**，是 Python 自带的。但 `iter` / `next` 在 3.3、3.5 里反复出现，所以单列一章。

## `iter(x)`

> 向可迭代对象要一个"带游标的迭代器"。

```python
it = iter([10, 20])
it = iter(data.DataLoader(ds, 18))     # DataLoader 必须先 iter 才能 next
```

⚠️ `DataLoader` 有 `__iter__` 但**没有 `__next__`**，所以 `next(dl)` 会 `TypeError`，必须 `next(iter(dl))`。
⚠️ `iter()` 每次都**新建游标**：再 `iter(dl)` 又从第 1 批开始。

---

## `next(it, default)`

> 从迭代器取下一个；取完抛 `StopIteration`。

```python
X, y = next(iter(dl))       # 只取第一批，不跑完整个 epoch
next(it, '没了')            # 提供默认值就不抛异常
```

---

## `len(x)`

> 长度（张量取第 0 维大小）。

```python
len([1, 2, 3])              # → 3
len(torch.randn(4, 5))      # → 4      第 0 维
len(dl)                     # → 批数
```

---

## `range(start, stop, step)`

> 左闭右开的整数序列。

```python
range(5)          # → 0,1,2,3,4
range(0, 10, 2)   # → 0,2,4,6,8
```

---

## `enumerate(seq, start=0)`

> 同时给下标和值。

```python
list(enumerate(['a', 'b']))        # → [(0,'a'), (1,'b')]
list(enumerate(['a', 'b'], 1))     # → [(1,'a'), (2,'b')]
```

---

## `zip(*seqs)`

> 把多个序列"拉链"起来，**以最短的为准**。

```python
list(zip([1,2,3], ['a','b']))      # → [(1,'a'), (2,'b')]   第 3 个被丢掉
```

⚠️ 长度不等时**静默截断**，不报错。

---

## `round(x, n)`

> 四舍五入到 n 位 —— **但用的是"银行家舍入"**：`.5` 往偶数靠。

```python
round(0.5)      # → 0      （不是 1！）
round(1.5)      # → 2
round(2.5)      # → 2      （不是 3！）
round(2.675, 2) # → 2.67   （浮点表示导致的，不是 2.68）
```

⭐ 你练习文件里那 47 处 `round(v, 6)` 就是为打印好看，误差范围内无所谓。
⚠️ 要做**真正的四舍五入**（比如财务），别用 `round`。

---

## `abs(x)`

> 绝对值（Python 内置，不是 `torch.abs`）。

```python
abs(-3.5)                  # → 3.5
torch.abs(torch.tensor([-1., 2.]))   # → [1., 2.]   张量用这个
```

---

## `type(x)` / `isinstance(x, T)`

> 查类型 / 判断类型。

```python
type(3)                  # → <class 'int'>
isinstance(3, int)       # → True
isinstance(True, int)    # → True   ⚠️ bool 是 int 的子类
```

⭐ 调试时最常用：`type(net.parameters())`、`isinstance(p, torch.nn.Parameter)`。

---

## `sum(seq)`

> 求和（对列表/生成器，不是张量）。

```python
sum([1, 2, 3])                                  # → 6
sum(p.numel() for p in model.parameters())      # → 总参数量
```

⚠️ 张量要用 `.sum()`，不是 `sum()`。

---

## `int()` / `float()` / `list()` / `tuple()`

> 类型转换。

```python
int(3.9)                      # → 3     （截断，不四舍五入）
float('1.5')                  # → 1.5
tuple(torch.randn(2, 3).shape)  # → (2, 3)   打印形状用
list(net.parameters())        # → 把 generator 变成列表
```

⭐ `tuple(x.shape)` 是打印形状的标准写法。

---

## `open(path, encoding="utf-8")`

> 读文件。

```python
with open('a.txt', encoding='utf-8') as f:
    text = f.read()
```

⚠️ Windows 中文环境下**一定要写 `encoding='utf-8'`**，否则默认 GBK 会乱码。

---

# 11 自己写过的函数索引

> 这些是你（或 d2l）在练习文件里用 `def` 写的。忘了在哪定义、参数是什么，查这里。

## ⚠️ 先澄清一个容易混的名字：`data_iter`

**两个文件里同名，但性质完全不同：**

| 出现位置 | 是什么 |
|---|---|
| `02_线性回归的从零实现.py` 第 102 行 | **自定义函数**（生成器），`def data_iter(batch_size, features, labels)` |
| `03_线性回归的简洁实现.py` 里 | **变量**，装的是 `data.DataLoader(...)` |

**所以看到 `for X, y in data_iter` 时，要先确认是哪个文件。**

---

## `synthetic_data(w, b, num_examples)` —— `02` 第 63 行 / `03` 第 61 行

> 造线性回归的合成数据。

```python
features, labels = synthetic_data(true_w, true_b, 1000)
# features → (1000, 2)    labels → (1000, 1)
```

---

## `data_iter(batch_size, features, labels)` —— `02` 第 102 行

> 手写的分批生成器（`yield`）。

```python
for X, y in data_iter(10, features, labels):
    ...      # X → (10, 2)   y → (10, 1)
```

⚠️ `y` 是 `(batch, 1)`，因为手写版的损失要 `y.reshape(y_hat.shape)`。
⚠️ 用了 `yield`，所以是**生成器**：调一次返回一个迭代器，遍历完就没了。

---

## `linreg(X, w, b)` —— `02` 第 170 行

> 手写线性模型。

```python
linreg(X, w, b)      # → X @ w + b
```

---

## `squared_loss(y_hat, y)` —— `02` 第 175 行

> 手写平方损失，**带 1/2**。

```python
squared_loss(y_hat, y)      # → (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

⚠️ 和 `nn.MSELoss()` 差一个 1/2 → 梯度差 2 倍。

---

## `sgd(params, lr, batch_size)` —— `02` 第 212 行

> 手写随机梯度下降，**内部除以 batch_size**。

```python
sgd([w, b], lr=0.03, batch_size=10)
```

⚠️ `torch.optim.SGD` **不除** batch_size，两者步长差 `batch_size` 倍。

---

## `squared_loss_32(...)` —— `03` 第 255 行

> 3.3 里为了和 `nn.MSELoss()` 对比，复制的**无 1/2** 版本。

---

## `build()` —— `04` 第 204 行

> 实验 4 里造"错得很离谱的 logits"。

```python
build()      # → tensor([[10.0, 0.0, 0.0]], requires_grad=True)
```

---

## `get_fashion_mnist_labels(labels)` —— `05` 第 116 行

> 标签索引 → 英文名。

```python
get_fashion_mnist_labels(torch.tensor([9, 0]))    # → ['ankle boot', 't-shirt']
```

---

## `show_images(imgs, num_rows, num_cols, titles=None, scale=1.2)` —— `05` 第 121 行

> 画图片网格。

```python
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
```

---

## `load_data_fashion_mnist(batch_size, resize=None)` —— `05` 第 227 行

> 下载 + 变换 + 打包，返回两个 DataLoader。

```python
train_iter, test_iter = load_data_fashion_mnist(256)
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
```

⭐ 3.6 / 3.7 每节开头都用它。

---

## `cross_entropy(y_hat, y)` —— 3.6 教材版

> 手写交叉熵：`-log(每个样本给正确类的概率)`，返回 **`(n,)` 向量**。

```python
y_hat = torch.tensor([[0.1, 0.3, 0.6],
                      [0.3, 0.2, 0.5]])
y = torch.tensor([0, 2])

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)     # → tensor([2.3026, 0.6931])
```

**展开看就四步：**

| 步 | 代码 | 结果 |
|---|---|---|
| 1 | `len(y_hat)` | `2`（样本数） |
| 2 | `range(2)` | `[0, 1]`（行号） |
| 3 | `y_hat[[0,1], y]` | `[0.1, 0.5]`（正确类概率） |
| 4 | `-torch.log(...)` | `[2.3026, 0.6931]`（每样本损失） |

⚠️ **返回 `(n,)` 向量不是标量**，直接 `backward()` 会 `RuntimeError: grad can be implicitly created only for scalar outputs`，必须 `.mean()`。
⚠️ 输入是**概率**（`y_hat`），不是 logits —— 与 `nn.CrossEntropyLoss` 相反。
⭐ 等价对照：`F.cross_entropy(torch.log(y_hat), y, reduction='none')` 得到同一向量。
⭐ `range(len(y_hat))` 等价于 `torch.arange(n)`，就是"所有样本的行号"，和 `y` 按位置配对。

---

## `accuracy(y_hat, y)` —— 3.6 教材版

> 算这一批里预测对了几个：把概率矩阵压成类别编号，再和 `y` 逐个比。

```python
def accuracy(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)      # (n, C) → (n,)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)      # → 0.5   正确率
```

**逐行：**

| 步 | 操作 | 结果（`y_hat=[[.1,.3,.6],[.3,.2,.5]]`, `y=[0,2]`） |
|---|---|---|
| 1 | `argmax(axis=1)` | `[2, 2]` 预测类别 |
| 2 | `== y` | `[False, True]` |
| 3 | `.type(...).sum()` | `1`（猜对 1 个） |
| 4 | `float(...)` | `1.0` |

⚠️ **返回的是“猜对几个”，不是正确率** —— 要自己 `/ len(y)`。
⚠️ 那行 `if` 是防御：`(n,)` 或 `(n,1)` 传进来就跳过 `argmax`，不会瞎转。
⚠️ 忘了 `argmax` 时 `(n,3)==(n,)` 一般会报错；但**类别数恰好等于样本数**时会广播成功、静默算错。
⭐ 等价一行：`(y_hat.argmax(dim=1) == y).float().mean()`。

---

## `Accumulator(n)` —— 3.6 教材版

> `n` 个格子的累加器（d2l 自己写的类），用来在流式读取时"边读边记账"。

```python
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
```

**`train_epoch_ch3` 里用 `Accumulator(3)`，三格分别是：**

| 格子 | 累加（课本版） | 累加（网课版） | 含义 |
|---|---|---|---|
| `metric[0]` | `float(l.sum())` | `float(l) * len(y)` | 损失**总和** |
| `metric[1]` | `accuracy(y_hat, y)` | 同 | 猜对总个数 |
| `metric[2]` | `y.numel()` | `y.size().numel()` | 总样本数 |

```python
return metric[0] / metric[2], metric[1] / metric[2]
#      平均 loss            准确率
```

⚠️ **两种写法的前提不同，不能混用：**
> 课本 `cross_entropy` 返回**向量 `(n,)`** → 用 `l.sum()`；backward 用 `l.mean().backward()`
> 网课版 `l` 已是**标量 mean** → 用 `l * len(y)`；backward 用 `l.backward()`
> 混用会炸：`float(3元素向量)` → `ValueError: only one element tensors ...`
> 实测两版总损失一致：`[0.3567, 0.3567, 0.9163].sum() = 1.629641` = `0.543214 × 3`

⚠️ `float(...)` 还有个作用：把带梯度的张量转成普通 float，**脱离计算图**，避免整轮训练的计算图被一直引用。
⭐ `n` = 要返回几个统计量 + 1；最后一格永远是总样本数（当分母）。`evaluate_accuracy` 只要准确率，所以是 `Accumulator(2)`。
⭐ `metric[0]` 能取值是因为类里实现了 `__getitem__`。

---

## 辅助函数（一次性用的，不单列）

| 函数 | 文件 | 行 | 干什么 |
|---|---|---|---|
| `timeit(f)` | `01_线性回归` | 313 | 给函数计时 |
| `add_loop(x, y)` | `01_线性回归` | 319 | for 循环版逐元素加（测矢量化用） |
| `add_vec(x, y)` | `01_线性回归` | 326 | 张量版逐元素加 |
| `gauss_nll(x, mu, sigma)` | `01_线性回归` | 374 | 高斯负对数似然，验证 MSE 的来源 |
| `load_and_prepare(path)` | `02_数据预处理` | 357 | 读 csv 并清洗 |
| `numerical_lim(...)` | `04_微积分` | 86 | 数值极限演示 |
| `num_deriv(...)` | `04_微积分` | 126 | 数值导数演示 |
| `run_with_order(...)` | `03_简洁实现` | 391 | 验证 zero_grad 三种顺序 |
| `f` / `q` / `g` / `h_func` / `example` | `04_微积分` | 38–410 | 微积分章节的临时演示函数 |

---

# 12 形状速查表

| 场景 | 形状 |
|---|---|
| 单张图（Fashion-MNIST） | `(1, 28, 28)` |
| 一个 batch 的图 | `(batch, 1, 28, 28)` |
| 拉平后的图 | `(batch, 784)` |
| 一批分类标签 | `(batch,)` int64 |
| 一批 logits | `(batch, 10)` |
| 一批概率 | `(batch, 10)`，每行和为 1 |
| `nn.Linear(784, 10).weight` | `(10, 784)` = `(out, in)` |
| 回归标签 | `(batch, 1)` float |
| 手写 `W`（3.2 风格） | `(in, out)` = `(784, 10)` |

---

# 13 踩过的坑（浓缩版）

| # | 坑 | 正确做法 |
|---|---|---|
| 1 | `softmax` 直接 `exp` | 先减最大值，否则 `exp(89)` 溢出 |
| 2 | 把概率喂给 `CrossEntropyLoss` | 喂 **logits** |
| 3 | 分类 `y` 用 `(N,1)` 或 one-hot | `(N,)` 的 `int64` |
| 4 | `MSELoss` 以为有 1/2 | 没有，梯度是手写版 2 倍 |
| 5 | 忘了 `zero_grad` | `.grad` 是累加的 |
| 6 | `zero_grad` 放在 `backward` 后 | 必须在 `backward` 之前 |
| 7 | `(n,1) - (n,)` | 会广播成 `(n,n)`，先 `reshape` |
| 8 | `sum()` 不给 `dim` | 二维里要 `sum(dim=1, keepdim=True)` |
| 9 | `view` 用在非连续张量上 | 用 `reshape` |
| 10 | `.grad` 没判 `None` | `assert p.grad is not None` |
| 11 | `next(dl)` | 要 `next(iter(dl))` |
| 12 | `nn.Sequential` 只有 1 层 | 纯多余，直接调那一层 |
| 13 | `net[0].weight` | 类型检查器会标红，用命名的 `linear` |
| 14 | 浮点数用 `==` 比较 | 用 `torch.allclose` |

---

# 14 自己加条目

复制这个模板，粘到对应章节的末尾：

````markdown
## `torch.xxx(args)`

> 一句话作用。

```python
torch.xxx(...)      # → 结果
```

⚠️ 坑或注意事项（没有就删掉这行）。
⭐ 记忆技巧或常见用法（可选）。
````

**标题必须是 `## `**，这样 `Ctrl+Shift+O` 才能列出来（和章节形成两级大纲）。
分类对不上就新建一个 `# N 章节名`。

---

*最后更新：2026-10-24*
