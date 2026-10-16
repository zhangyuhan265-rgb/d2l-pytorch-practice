#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 62)
print(" 2.3 线性代数 —— 练习")
print("=" * 62)
print("""
【教材】https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html

【先说清楚：这节不是纯数学】
  标题叫"线性代数"，内容其实是【PyTorch 张量运算 API】。
  数学部分你 NumPy 里都见过，真正要学的是 API。

【三个重点】
  1. 降维求和 + keepdims=True      <- 最实用
  2. dot / mv / mm 三个乘法         <- 神经网络的核心
  3. norm 范数                      <- 损失函数和正则化的基础

【怎么用】光标放在某格里按 Shift+Enter，只跑这一格。
         第 0 格必须先跑。
""")

from numpy import float32, shape
import torch

print("torch 版本:", torch.__version__)


#%% 实验 1：元素、形状、长度 —— 别搞混这三个
print("--- 实验 1：元素数 vs 形状 vs 长度 ---")

x = torch.arange(4)
y = torch.arange(4,dtype = torch.float32)
print(y[0])
print(y[0].item())
print("x =", x)
print()
print(x.shape)
print("  x.shape  =", tuple(x.shape), "   <- 形状（元组）")
print("  x.numel()=", x.numel(), "              <- 元素总数（一个数）")
print("  len(x)   =", len(x), "              <- 第 0 维的长度")
print("  x[3]     =", x[3], "              <- 索引取元素")
print()

A = torch.arange(20).reshape(5, 4)
print("A.shape =", tuple(A.shape))
print("  A.numel() =", A.numel(), "  <- 5*4 = 20")
print("  len(A)    =", len(A), "  <- 只看第 0 维（行数）！不是总数")
print()

print("=" * 50)
print(" 坑：len() 和 numel() 完全不是一回事")
print("=" * 50)
print("""
  x = torch.arange(4)
      x.shape   -> (4,)      形状
      x.numel() -> 4         元素总数
      len(x)    -> 4         第 0 维长度（一维时和 numel 恰好相同）

  A = torch.arange(20).reshape(5, 4)
      A.shape   -> (5, 4)
      A.numel() -> 20        元素总数 = 5*4
      len(A)    -> 5         第 0 维长度，【不是 20】！

  numel = number of elements = 所有维度乘起来
  len   = 只数第一个维度
""")
print()

# TODO 先猜答案，再取消注释
B = torch.arange(24).reshape(2, 3, 4)
print("B.shape =", tuple(B.shape))     #(2,3,4)
print("B.numel() =", B.numel())   #24
print("len(B) =", len(B))   #2
# 猜猜 len(B) 是几？为什么？


#%% 实验 2：转置和对称矩阵
print("--- 实验 2：转置 .T ---")

A = torch.arange(20).reshape(5, 4)
print(A)
print("A.shape     =", tuple(A.shape))
print("A.T.shape   =", tuple(A.T.shape), "   <- 行和列换了")
print()
print("A 的前两行:")
print(A[:2])
print()
print("A.T 的前两行（就是 A 的前两列）:")
print(A.T[:2])
print()

print("=" * 50)
print(" 对称矩阵：转置等于自己")
print("=" * 50)
print()
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print("B =")
print(B)
print()
print("B == B.T 的结果:")
print(B == B.T)
print()
print("  全是 True -> B 是【对称矩阵】")
print("  什么叫对称：B[i,j] == B[j,i]，沿对角线镜像相等")
print()

# TODO 先猜：A == A.T 会怎样？  #报错？
# 提示：A 是 (5,4)，A.T 是 (4,5)，形状都不一样
print("A == A.T ->", A == A.T)#猜对了，跟reshape()里的值乘起来必须等于arange类似


#%% 实验 3：clone —— 和 2.1 学的 reshape 对比
print("--- 实验 3：clone 是复制，不是视图 ---")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()          # 真的复制一份新内存

print("A.data_ptr() =", A.data_ptr())
print("B.data_ptr() =", B.data_ptr())
print("共用内存吗   :", A.data_ptr() == B.data_ptr(), "   <- False，是两份独立数据")
print()

B[0, 0] = 999
print("改了 B[0,0] = 999 之后:")
print("  A[0,0] =", A[0, 0].item(), "  <- 没被影响")
print("  B[0,0] =", B[0, 0].item())
print()

print("=" * 50)
print(" 对比 2.1 学的 reshape")
print("=" * 50)
print("""
  A.reshape(4, 5)   ->  返回【视图】，共用内存，改一个影响另一个
  A.clone()         ->  返回【副本】，独立内存，互不影响

  判断方法永远是同一个:  a.data_ptr() == b.data_ptr()
""")
print()

C = A.reshape(4, 5)
print("验证:")
print("  A.data_ptr() == A.reshape(4,5).data_ptr() :", A.data_ptr() == C.data_ptr(), "  <- reshape 共用")
print("  A.data_ptr() == A.clone().data_ptr()      :", A.data_ptr() == A.clone().data_ptr(), "  <- clone 独立")
# ============ 判断「共用内存」还是「独立副本」 ============
#
# id()          -> 看 Python 对象外壳。视图和副本都是新壳，全都不同，❌ 分不出
# data_ptr()    -> 看第 0 个元素的地址。reshape 有效，但切片会偏移起始位置，
#                  所以切片会被误判成「独立」，⚠️ 有坑
# untyped_storage().data_ptr()  -> 看底层存储块本身，不受偏移影响，✅ 最可靠
#
# 结论：
#   判断「改一个会不会影响另一个」-> untyped_storage().data_ptr() 相同即共用
#   判断「是不是同一个对象」      -> 用 is
#
# 各类操作的实际情况：
#   y = x                   共用    同一对象
#   y = x.reshape(...)      共用    视图
#   y = x[2:6]              共用    切片视图（但 data_ptr 会不同！）
#   y = x.reshape(3,4).t()  共用    转置视图
#   y = x.clone()           独立    副本
#   y = torch.cat((x,x))    独立    新张量
#
# ⚠️ 比较前先把切片存进变量，别写成 x.data_ptr() == x[1:3].data_ptr()
#    临时对象会被回收，结果不可靠
# ==========================================================

#%% 实验 4：按元素运算 + 标量广播
print("--- 实验 4：按元素运算 ---")
#没啥意思，核心是矩阵运算，注意函数调用
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()

print("A + B 的形状:", tuple((A + B).shape), "  <- 形状不变")
print("A * B 的形状:", tuple((A * B).shape), "  <- 是【逐元素相乘】，不是矩阵乘法！")
print()

print("=" * 50)
print(" 坑：A * B 是逐元素乘，矩阵乘法是 A @ B 或 torch.mm")
print("=" * 50)
print()
print("  A * B     -> 对应位置相乘，形状不变")
print("  A @ B     -> 矩阵乘法（需要形状匹配）")
print("  torch.mm  -> 同 A @ B")
print()

print("验证 A * B 是逐元素:")
print("  A[0] =", A[0].tolist())
print("  B[0] =", B[0].tolist())
print("  (A*B)[0] =", (A * B)[0].tolist(), "  <- 就是对应位置乘起来的")
print()

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print("标量广播: a + X")
print("  X.shape     =", tuple(X.shape))
print("  (a + X).shape =", tuple((a + X).shape), "  <- 标量不改变形状")
print("  (a * X).shape =", tuple((a * X).shape))
print("  验证 a*X 每个元素都乘了 2:", (a * X)[0, 0].tolist(), " vs X[0,0]*2 =", (X[0, 0] * 2).tolist())
print()

# TODO 先猜：X + a 和 a + X 结果一样吗？
# print(torch.equal(a + X, X + a))


#%% 实验 5：降维求和 —— axis 的三种用法
print("--- 实验 5：降维求和 ---")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print("A (5行4列):")
print(A)
print()

print("1) 全加起来，得到标量:")
print("   A.sum() =", A.sum().item(), "  shape:", tuple(A.sum().shape), " <- 零维标量")
print()

print("2) axis=0 —— 消灭第 0 维（行）:")
s0 = A.sum(dim=0)
print("   A.sum(axis=0) =", s0.tolist())
print("   shape =", tuple(s0.shape), "  <- 5 行没了，剩 4 个数")
print("   含义: 每一列各自相加")
print()

print("3) axis=1 —— 消灭第 1 维（列）:")
s1 = A.sum(dim=1)
print("   A.sum(axis=1) =", s1.tolist())
print("   shape =", tuple(s1.shape), "  <- 4 列没了，剩 5 个数")
print("   含义: 每一行各自相加")
print()

print("4) axis=[0,1] —— 一次消灭两维:")
s01 = A.sum(dim=[0, 1])
print("   A.sum(axis=[0,1]) =", s01.item(), "  <- 和 A.sum() 一样")
print("shape:   ",tuple(s01.shape))
print()

print("=" * 50)
print(" 规律：axis=k 就是消灭第 k 维（可以一次给多个）")
print("=" * 50)
print()
print("  A.sum()            -> 标量     全部相加")
print("  A.sum(axis=0)      -> (4,)     消第0维")
print("  A.sum(axis=1)      -> (5,)     消第1维")
print("  A.sum(axis=[0,1])  -> 标量     两个都消掉")
print()
print("  三维也一样:")
B3 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
for ax in [0, 1, 2, [0, 1], [1, 2]]:
    r = B3.sum(dim=ax)
    print(f"    (2,3,4).sum(axis={str(ax):<7}) -> {tuple(r.shape)}")
print()

# TODO 先猜：(2,3,4).sum(axis=[0,1,2]) 是什么形状？  0维？shape:()
print(B3.sum(dim=[0,1,2]))
print(tuple(B3.sum(dim=[0,1,2]).shape))


#%% 实验 6：mean 和 sum 的关系
print("--- 实验 6：均值 = 和 / 个数 ---")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)
print("A.sum()   =", A.sum().item())
print("A.numel() =", A.numel())
print("A.sum() / A.numel() =", (A.sum() / A.numel()).item())
print("A.mean()  =", A.mean().item())
print("两者相等吗:", torch.allclose(A.mean(), A.sum() / A.numel()))
print()

print("按维度求均值:")
print(A.sum(dim=0))
print("  A.mean(axis=0) =", A.mean(dim=0).tolist())
print("  A.sum(axis=0) / A.shape[0] =", (A.sum(dim=0) / A.shape[0]).tolist())
print("  相等吗:", torch.allclose(A.mean(dim=0), A.sum(dim=0) / A.shape[0]))
print()
print("  ^^^ 注意除的是 A.shape[0]（行数=5），因为 axis=0 把 5 行压成了 1 个")
print()

# TODO 先猜：A.mean(dim=1) 应该除以几？ 除4？
print("A.mean(axis=1) =", A.mean(dim=1).tolist())
print("A.sum(axis=1) / A.shape[1] =", (A.sum(dim=1) / A.shape[1]).tolist())


#%% 实验 7 ⭐：keepdims —— 本节最实用的一格
print("--- 实验 7：keepdims （重点）---")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print("A (5,4):")
print(A)
print()

s_no = A.sum(dim=1)
s_keep = A.sum(dim=1, keepdim=True)
print("不加 keepdims:")
print("  值   :", s_no.tolist())
print("  形状 :", tuple(s_no.shape), "     <- 一维")
print()
print("加 keepdims=True:")
print("  值   :", s_keep.tolist())
print("  形状 :", tuple(s_keep.shape), "   <- 还是二维！")
print()
print("  值完全一样，只有形状不同")
print()

print("=" * 50)
print(" 形状有什么用？看 A / sum_A")
print("=" * 50)
print()
print("不加 keepdims 会怎样:")
try:
    r = A / s_no
    print("  居然成功了?", tuple(r.shape))
except RuntimeError as e:
    print("  ✗ 报错:", str(e)[:95])
print()
print("加 keepdims 就通:")
r = A / s_keep
print("  A / A.sum(axis=1, keepdims=True)")
print("  形状 =", tuple(r.shape))
print(r)
print()
print("  验证：每一行加起来 = 1.0")
for i in range(5):
    print(f"    第{i}行之和 = {r[i].sum().item():.4f}")
print()

print("=" * 50)
print(" 一句话理解 keepdims")
print("=" * 50)
print("""
  keepdims=True 让"被压掉的那一维"变成 1，而不是消失。

    A.sum(axis=1)                 -> (5,)    一维
    A.sum(axis=1, keepdims=True)  -> (5,1)   二维

  为什么要保留：为了【广播】。
    (5,4) / (5,1)  -> 广播成功，每行除以自己那行的和   ✅
    (5,4) / (5,)   -> 对齐规则不同，报错             ❌

  什么时候用：算一个统计量、然后拿它去除原张量的时候。
    这就是 softmax、批归一化、注意力机制的共同套路。
""")
print()

# TODO 自己试试 axis=0 配 keepdims
m = A.mean(dim=0, keepdim=True)
print("A.mean(dim=0, keepdims=True).shape =", tuple(m.shape))   # 猜猜是几？
print("A - m  的形状 =", tuple((A - m).shape))
print("这叫【中心化】：每一列减去该列的均值")


#%% 实验 8：cumsum 累加
print("--- 实验 8：cumsum 累加 ---")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print("A:")
print(A)
print()
print("A.cumsum(axis=0)  <- 沿着第 0 维累加:")
print(A.cumsum(dim=0))
print()
print("  就是每一列从上往下累加:")
for j in range(4):
    col = A[:, j].tolist()
    cum = A.cumsum(dim=0)[:, j].tolist()
    print(f"    第{j}列: {[int(v) for v in col]}  ->  累加后 {[int(v) for v in cum]}")
print()
print("A.cumsum(axis=1)  <- 沿着第 1 维累加:")
print(A.cumsum(dim=1))
print()

# TODO 先猜：A.cumsum(axis=0) 的最后一行等于什么？
print("最后一行:", A.cumsum(dim=0)[-1].tolist())
print("A.sum(axis=0):", A.sum(dim=0).tolist())
# 它们相等吗？为什么？


#%% 实验 9 ⭐：dot / mv / mm 三个乘法
print("--- 实验 9：三个乘法（神经网络的核心）---")
print()

print("=" * 50)
print(" ① torch.dot(x, y) —— 两个一维向量 -> 一个数")
print("=" * 50)
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print("  x =", x.tolist())
print("  y =", y.tolist())
print("  torch.dot(x, y) =", torch.dot(x, y).item())
print()
print("  ⭐ 等价写法：torch.sum(x * y)")
print("  torch.sum(x * y) =", torch.sum(x * y).item())
print()
print("  点积的本质：对应位置相乘，再全部加起来")
print("    0*1 + 1*1 + 2*1 + 3*1 =", int(torch.dot(x, y).item()))
print()

print("=" * 50)
print(" ② torch.mv(A, x) —— 矩阵 × 向量")
print("=" * 50)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print("  A.shape =", tuple(A.shape), "  x.shape =", tuple(x.shape))
r = torch.mv(A, x)
print("  torch.mv(A, x).shape =", tuple(r.shape))
print("  torch.mv(A, x) =", r.tolist())
print()
print("  规则: (5,4) × (4,) -> (5,)   矩阵的列数必须等于向量的长度")
print("  验算第 0 行: ", end="")
print(" + ".join(f"{int(A[0,j])}*{int(x[j])}" for j in range(4)), "=", int(r[0].item()))
print()
#矩阵*向量就是把矩阵的每一行抽出来和向量做内积⭐⭐⭐
print("=" * 50)
print(" ③ torch.mm(A, B) —— 矩阵 × 矩阵")
print("=" * 50)
Bm = torch.ones(4, 3)
print("  A.shape =", tuple(A.shape), "  B.shape =", tuple(Bm.shape))
rm = torch.mm(A, Bm)
print("  torch.mm(A, B).shape =", tuple(rm.shape))
print("  规则: (5,4) × (4,3) -> (5,3)   中间的 4 必须一致，然后消失")
print()

print("=" * 50)
print(" 对照表（记住这张表）")
print("=" * 50)
print("""
  函数          输入               输出        比喻
  -----------   ----------------   ---------   ------------------
  torch.dot     (n,) 和 (n,)       标量        两个向量"对齐相乘再求和"
  torch.mv      (m,n) 和 (n,)      (m,)        矩阵作用于向量
  torch.mm      (m,n) 和 (n,p)     (m,p)       两个矩阵相乘

  形状规则：【中间那个 n 必须一致，然后消失】
      (5,4) @ (4,3) -> (5,3)
          ↑     ↑
        必须相等

  ⭐ 神经网络的前向传播就是 mv:
      y = W @ x + b
      W 是 (输出维度, 输入维度)，x 是 (输入维度,)
""")
print()

# TODO 先猜：torch.mm(A, A.T) 是什么形状？
print(tuple(torch.mm(A, A.T).shape))
# TODO: 那 torch.mm(A, A) 能跑吗？为什么？
#不能，需要（5*4）*（4*5）这样的


#%% 实验 10：范数 norm
print("--- 实验 10：范数 ---")

u = torch.tensor([3.0, -4.0])
print("u =", u.tolist())
print()
print("torch.norm(u)      =", torch.norm(u).item(), "   <- L2 范数（欧几里得长度）")
print("torch.abs(u).sum() =", torch.abs(u).sum().item(), "   <- L1 范数（绝对值之和）")
print()
print("  验算 L2: sqrt(3^2 + (-4)^2) = sqrt(9+16) = sqrt(25) =", (3**2 + 4**2) ** 0.5)
print("  验算 L1: |3| + |-4| = 3 + 4 =", 3 + 4)
print()

big = torch.ones((4, 9))
print(big)
print("torch.norm(torch.ones((4, 9))) =", torch.norm(big).item())
print("  验算: 36 个 1，sqrt(36) =", 36 ** 0.5)
print()

print("=" * 50)
print(" 范数是什么、为什么重要")
print("=" * 50)
print("""
  范数 = "把一个向量/矩阵变成一个数"的度量方式，衡量它"有多大"。

  L2 范数 (默认): sqrt(所有元素平方和)     <- 直线距离，最常用
  L1 范数:       所有元素绝对值之和         <- 对异常值更宽容

  为什么重要（不是纯数学）：
    1. 损失函数里用它衡量"预测和真实差多远"
    2. 权重衰减（4.5 节）用它惩罚过大的权重，防过拟合
    3. 梯度裁剪用它限制梯度大小，防梯度爆炸
""")
print()

# TODO 先猜：torch.norm(torch.tensor([1.0, 1.0])) 是多少？
# 提示：sqrt(1+1)
print(torch.norm(torch.tensor([1.0, 1.0])).item())


#%% 实验 11：串起来 —— 一个真正的前向传播
print("--- 实验 11：用学到的拼一次前向传播 ---")
print()

torch.manual_seed(0)

# 假装有 3 个样本，每个样本 4 个特征
X = torch.randn(3, 4)
print("输入 X (3个样本, 4个特征):")
print(X)
print()

# 一个线性层：4 个输入 -> 2 个输出
W = torch.randn(2, 4)      # (输出维度, 输入维度)
b = torch.randn(2)         # 每个输出一个偏置
print("权重 W.shape =", tuple(W.shape), "  偏置 b.shape =", tuple(b.shape))
print()

print("怎么算这个线性层？")
print("  单个样本: y = W @ x + b   -> 用 torch.mv")
print("  一批样本: Y = X @ W.T + b -> 用 torch.mm")
print()

# 方法 1：循环每个样本，用 mv
outs1 = []
for i in range(3):
    outs1.append(torch.mv(W, X[i]) + b)
outs1 = torch.stack(outs1)
print("方法1（循环 + mv）结果 shape:", tuple(outs1.shape))
print(outs1)
print()

# 方法 2：一次算完，用 mm
outs2 = torch.mm(X, W.T) + b
print("方法2（一次 mm）结果 shape:", tuple(outs2.shape))
print(outs2)
print()

print("两种方法结果一样吗:", torch.allclose(outs1, outs2))
print()
print("=" * 50)
print(" 这里用到了本节所有重点")
print("=" * 50)
print("""
  · W.T        -> 转置（实验 2）
  · torch.mm   -> 矩阵乘法（实验 9）
  · + b        -> 广播（标量/向量加到矩阵上，实验 4）
  · 一批数据   -> 一次算完，比循环快得多

  接下来 3.1 线性回归，做的就是这件事 —— 只不过还要算损失、
  求梯度、更新参数。
""")
print()

print("=" * 50)
print(" 为什么用 mm 而不是循环 mv")
print("=" * 50)
print("""
  循环: 每个样本调一次 mv，Python 层面循环一次 -> 慢
  一次 mm: 整个批次一次算完 -> GPU 能并行 -> 快几十倍

  所以真实训练里永远是 "整个 batch 一起算"。
  这就是为什么你的张量总是带一个 batch 维度：
      (batch_size, 特征数)
""")


#%% 踩坑记录（自己填）

# 格式： 现象 -> 原因
#
# 例：
#   A / A.sum(axis=1) 报错 -> (5,4) 和 (5,) 广播对不上，要加 keepdims=True
#
# 你的记录：
#
#
#


#%% 最后一格：自检
print("=" * 62)
print(" 如果上面每一格都能跑通，2.3 就过关了")
print("=" * 62)
print()
print("对照一下，这些你能说清楚吗：")
print("  1. numel() 和 len() 有什么区别？")
print("  2. A * B 和 torch.mm(A, B) 有什么区别？")
print("  3. 对称矩阵是什么条件？")
print("  4. reshape 和 clone 有什么区别？（提示：data_ptr）")
print("  5. axis=k 的通用含义是什么？axis=[0,1] 呢？")
print("  6. keepdims=True 改变了什么？为什么要它？")
print("  7. dot / mv / mm 的输入输出形状分别是什么？")
print("  8. 点积的本质是什么？")
print("  9. L1 范数和 L2 范数分别怎么算？")
print(" 10. 为什么训练时要整个 batch 一起算，而不是循环？")
print()
print("说不清楚的那条，回去重跑对应的实验。")
print("=" * 62)
