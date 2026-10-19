#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 64)
print(" 3.1 线性回归 —— 练习")
print("=" * 64)
print("""
【教材】https://zh-v2.d2l.ai/chapter_linear-networks/linear-regression.html

【这是全书第一个真正的模型】
  第 2 章学的全是工具，这一节终于开始「做机器学习」。
  而且你会发现：第二章的每一节都在这里用上了。

     2.1 张量      →  数据 X、标签 y、参数 w
     2.3 线性代数  →  ŷ = wᵀx + b 是矩阵乘法；解析解 (XᵀX)⁻¹Xᵀy
     2.4 微积分    →  损失函数对 w 的偏导数
     2.5 自动微分  →  backward() 帮你算这些偏导数
     2.6 概率      →  平方损失的来源（高斯最大似然）

【任何机器学习模型的三个零件】
  ① 模型：怎么从输入算输出         ŷ = wᵀx + b
  ② 损失：怎么衡量预测有多差        L = (1/2n)·Σ(ŷᵢ-yᵢ)²
  ③ 优化：怎么把损失降下去          梯度下降

  这三件事，后面所有模型（CNN、Transformer）都是同一个套路，
  只是把第 ① 件换成更复杂的函数。

【这一节其实只有两个代码演示】
  教材 3.1 是理论为主，代码只有：
     · 矢量化加速（for 循环 vs 张量运算）
     · 正态分布的曲线
  但【背后】的解析解和梯度公式才是重点，所以本练习把那些全做成实验。

【怎么用】光标放在某格里按 Shift+Enter。第 0 格必须先跑。
""")

import torch
import math
import time

torch.manual_seed(0)
print("torch 版本:", torch.__version__)


#%% 实验 1：线性模型到底长什么样
print("--- 实验 1：用一个具体例子看线性模型 ---")
print()

print("  场景：预测房价")
print("     输入 x = [面积(㎡), 房龄(年)]")
print("     参数 w = [w₁, w₂]     （每个特征一个权重）")
print("     偏置 b")
print()
print("  模型:  ŷ = w₁·面积 + w₂·房龄 + b")
print()

w = torch.tensor([2.0, -0.5])      # 每平米 2 万，每老一年减 0.5 万
b = torch.tensor(10.0)             # 基础价 10 万

houses = torch.tensor([
    [80.0, 5.0],
    [120.0, 10.0],
    [60.0, 0.0],
])
print("  房子数据 X（3 套，每套 2 个特征）:")
for i, row in enumerate(houses):
    print(f"    第{i+1}套: 面积={row[0].item():.0f}㎡, 房龄={row[1].item():.0f}年")
print()

pred = houses @ w + b
print("  预测价格 ŷ = X @ w + b:")
for i, p in enumerate(pred):
    h = houses[i]
    manual = w[0] * h[0] + w[1] * h[1] + b
    print(f"    第{i+1}套: {p.item():>8.2f} 万   （手算 {manual.item():>8.2f} 万）")
print()

print("  ⭐ 注意 X @ w 这一下：")
print(f"     X 的形状 = {tuple(houses.shape)}    (3 套, 2 个特征)")
print(f"     w 的形状 = {tuple(w.shape)}       (2 个权重)")
print(f"     结果形状 = {tuple((houses @ w).shape)}       (3 个预测值)")
print()
print("     这就是 2.3 学的矩阵乘法：(3,2) @ (2,) → (3,)")
print("     中间那个 2 消掉了 —— 特征维被「加权求和」掉了")
print()


#%% 实验 2：损失函数 —— 平方损失怎么算
print("--- 实验 2：平方损失 ---")
print()

print("  假设真实价格 y = [200, 300, 120] 万")
y_true = torch.tensor([200.0, 300.0, 120.0])
print(f"     真实 y     = {y_true.tolist()}")
print(f"     预测 ŷ     = {[round(v, 2) for v in pred.tolist()]}")#round(v,2)保留两位小数，银行家取舍法
print()

resid = pred - y_true
print(f"  ① 残差 (ŷ - y) = {[round(v, 2) for v in resid.tolist()]}")
print()
sq = resid ** 2
print(f"  ② 平方    (ŷ-y)² = {[round(v, 2) for v in sq.tolist()]}")
print()
print(f"  ③ 求和           = {sq.sum().item():.2f}")
print()
n = len(y_true)
L = sq.sum() / (2 * n)
print(f"  ④ 除以 2n (n={n}) → L = {L.item():.2f}")
print()

print("  ⭐ 教材的平方损失定义：")
print("        L(w, b) = (1/(2n)) · Σ (ŷᵢ - yᵢ)²")
print()
print("  为什么除以 2？")
print("     因为求导时 (x²)' = 2x，那个 2 会跟这里的 1/2 抵消，")
print("     梯度式子更干净：∂L/∂ŷ = (ŷ - y) / n ，没有 2 了。")
print("     （纯粹为了好看，不影响最优解在哪）")
print()

print("  验证一下这个说法：用 torch 自动求导看看")
pred2 = pred.clone().requires_grad_(True)
loss_2n = ((pred2 - y_true) ** 2).sum() / (2 * n)
loss_2n.backward()
assert pred2.grad is not None
print(f"     ∂L/∂ŷ = {[round(v, 4) for v in pred2.grad.tolist()]}")
print(f"     (ŷ-y)/n = {[round(v, 4) for v in ((pred - y_true) / n).tolist()]}")
print(f"     一致？ {torch.allclose(pred2.grad, (pred - y_true) / n)}   ← 果然没有 2")
print()


#%% 实验 3：解析解 —— 一步算出最优参数
print("--- 实验 3：解析解（闭式解）---")
print()

print("  线性回归有个特殊之处：最优解可以直接算出来，不用迭代。")
print()
print("  推导思路：")
print("     把 b 吸收进 w（在 X 左边加一列全 1），变成 ŷ = X_aug · w_aug")
print("     令梯度为 0：  X_augᵀ(X_aug·w_aug - y) = 0")
print("     解出：        w_aug = (X_augᵀ X_aug)⁻¹ X_augᵀ y")
print()

torch.manual_seed(0)
n3, d3 = 100, 3
X3 = torch.randn(n3, d3, dtype=torch.float64)
w_true = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
b_true = 3.0
noise = 0.01 * torch.randn(n3, dtype=torch.float64)
y3 = X3 @ w_true + b_true + noise

print(f"  造一份数据：n={n3} 个样本，{d3} 个特征，噪声 σ=0.01")
print(f"     真实 w = {w_true.tolist()}")
print(f"     真实 b = {b_true}")
print()

X_aug = torch.cat([torch.ones(n3, 1, dtype=torch.float64), X3], dim=1)
print(f"  ① 造增广矩阵：在 X 左边加一列全 1")
print(f"     X_aug 形状 = {tuple(X_aug.shape)}   （原来是 {tuple(X3.shape)}）")
print()

XtX = X_aug.T @ X_aug
Xty = X_aug.T @ y3
w_aug = torch.linalg.solve(XtX, Xty)
print(f"  ② 算 w_aug = (X_augᵀ X_aug)⁻¹ X_augᵀ y")
print(f"     X_augᵀ @ X_aug 形状 = {tuple(XtX.shape)}")
print(f"     X_augᵀ @ y     形状 = {tuple(Xty.shape)}")
print(f"     w_aug = {[round(v, 6) for v in w_aug.tolist()]}")
print()

print(f"  ③ 对照真实值：")
print(f"     第 0 个是 b： {w_aug[0].item():.6f}   vs 真实 {b_true}")
print(f"     后面是 w：    {[round(v, 6) for v in w_aug[1:].tolist()]}")
print(f"                          vs 真实 {w_true.tolist()}")
print()

print("  ⭐ 有噪声所以不完全相等，但非常接近（误差 ∝ 噪声大小）")
print()

print("  交叉验证：PyTorch 有内置的 lstsq")
sol = torch.linalg.lstsq(X_aug, y3).solution
print(f"     torch.linalg.lstsq(X_aug, y).solution")
print(f"     = {[round(v, 6) for v in sol.tolist()]}")
print(f"     跟我们手算一致？ {torch.allclose(w_aug, sol, atol=1e-10)}")
print()

print("  ⚠️ 那为什么实际训练还用梯度下降？")
print("     · (XᵀX)⁻¹ 的复杂度是 O(d³)，特征多时很慢")
print("     · XᵀX 可能不可逆（特征共线 / 样本比特征少）")
print("     · 神经网络根本没有解析解，只能迭代")
print("     → 所以梯度下降才是通用方法，解析解只在线性回归这种特例里好用")
print()


#%% 实验 4：梯度公式 —— 手推 vs autograd
print("--- 实验 4：平方损失的梯度 ---")
print()

print("  这是这一节最该亲手推一遍的东西。")
print()
print("  损失:  L = (1/(2n)) · Σᵢ (wᵀxᵢ + b - yᵢ)²")
print()
print("  令残差 rᵢ = ŷᵢ - yᵢ = wᵀxᵢ + b - yᵢ，则")
print()
print("     ∂L/∂w = (1/n) · Σᵢ rᵢ · xᵢ  =  (1/n) · Xᵀ r")
print("     ∂L/∂b = (1/n) · Σᵢ rᵢ        =  (1/n) · sum(r)")
print()
print("  ⭐ 记住这两个式子。第 3.2 节「从零实现」你就要亲手写它们。")
print()

torch.manual_seed(7)
n4 = 10
X4 = torch.randn(n4, 2)
w4 = torch.tensor([1.5, -2.0], requires_grad=True)
b4 = torch.tensor(0.5, requires_grad=True)
y4 = torch.randn(n4)

pred4 = X4 @ w4 + b4
loss4 = ((pred4 - y4) ** 2).sum() / (2 * n4)
loss4.backward()

r = pred4.detach() - y4
g_w_hand = X4.T @ r / n4
g_b_hand = r.sum() / n4
assert w4.grad  is not None
assert b4.grad  is not None
print("  实测对比：")
print()
print(f"     autograd  w.grad = {[round(v, 6) for v in w4.grad.tolist()]}")#round()
print(f"     手推公式  w_grad = {[round(v, 6) for v in g_w_hand.tolist()]}")
print(f"     一致？ {torch.allclose(w4.grad, g_w_hand, atol=1e-6)}  ✅")
print()
print(f"     autograd  b.grad = {b4.grad.item():.10f}")
print(f"     手推公式  b_grad = {g_b_hand.item():.10f}")
print(f"     一致？ {abs(b4.grad.item() - g_b_hand.item()) < 1e-6}  ✅")
print()

# ——— 留给你做 ———
print("=" * 52)
print(" ✏️ 自己推：如果损失不除以 n，只写成 L = Σ(ŷᵢ-yᵢ)²")
print("    那 ∂L/∂w 是什么？用 autograd 验证一下。")
print("    （提示：比上面那个式子多乘了什么？）")
print("=" * 52)
print()


#%% 实验 5：小批量 SGD —— 自己写一遍训练循环
print("--- 实验 5：小批量随机梯度下降 ---")
print()

print("  有了梯度就能训练了。算法：")
print()
print("     重复很多轮（epoch）：")
print("        ① 打乱数据")
print("        ② 每次取一小批 B（比如 10 个）")
print("        ③ 只用这一批算梯度  →  估计出全量梯度")
print("        ④ 沿反方向走一小步：w ← w - lr·grad")
print()
print("  ⭐ 为什么用「一小批」而不是全部？")
print("     全量要读完所有数据才算一次梯度，太慢。")
print("     小批量只用 10 个样本就能大致判断方向 —— 快得多。")
print("     （这就是 2.6 学的「用样本估计期望」）")
print()

torch.manual_seed(42)
n5 = 1000
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
X5 = torch.normal(0, 1, (n5, 2))#torch.normal(均值, 标准差, 形状)
y5 = X5 @ true_w + true_b
y5 = y5 + torch.normal(0, 0.01, y5.shape)#加噪声

w5 = torch.normal(0, 0.01, (2,), requires_grad=True)
b5 = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w5, b5], lr=0.03)

print(f"  真实参数: w = {[round(v,2) for v in true_w.tolist()]}, b = {true_b}")
print(f"  初始参数: w = {[round(v,4) for v in w5.tolist()]}, b = {b5.item():.4f}")
print(f"  样本数 {n5}，批大小 10，学习率 0.03")
print()
print(f"  {'epoch':>6}  {'全量loss':>14}  {'w':>22}  {'b':>9}")
print("  " + "-" * 58)

batch_size = 10
for epoch in range(6):
    perm = torch.randperm(n5)#打乱 0~999 的编号，返回一个「排列」
    for i in range(0, n5, batch_size):
        idx = perm[i:i + batch_size]
        xb, yb = X5[idx], y5[idx]
        loss_b = ((xb @ w5 + b5 - yb) ** 2).mean() / 2
        loss_b.backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        full = ((X5 @ w5 + b5 - y5) ** 2).mean() / 2
    wstr = str([round(v, 4) for v in w5.tolist()])
    print(f"  {epoch+1:>6}  {full.item():>14.8f}  {wstr:>22}  {b5.item():>9.4f}")

print()
print("  ✅ 6 轮之后 w 已经接近 [2.0, -3.4]，b 接近 4.2")
print()
print("  ⭐ 注意这个循环的结构，以后所有训练都是这五步：")
print("       ① forward    pred = xb @ w + b")
print("       ② loss       ((pred - yb)**2).mean() / 2")
print("       ③ backward   loss.backward()")
print("       ④ step       optimizer.step()")
print("       ⑤ zero_grad  optimizer.zero_grad()")
print()


#%% 实验 6：矢量化加速 —— 教材第一个代码演示
print("--- 实验 6：for 循环 vs 张量运算 ---")
print()


def timeit(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def add_loop(a, b):
    c = torch.zeros_like(a)# 先造零向量
    for i in range(len(a)): # 逐个下标
        c[i] = a[i] + b[i]# 取、加、写回
    return c


def add_vec(a, b):
    return a + b


print(f"  {'n':>8}  {'for 循环':>14}  {'张量运算':>14}  {'倍数':>10}")
print("  " + "-" * 52)

for n in [1000, 10000, 100000]:
    a = torch.ones(n)
    b = torch.ones(n)
    t1 = timeit(add_loop, a, b)
    t2 = timeit(add_vec, a, b)
    ratio = t1 / t2 if t2 > 0 else float("inf")
    print(f"  {n:>8}  {t1*1000:>12.2f}ms  {t2*1000:>12.4f}ms  {ratio:>9.0f}x")

print()
print("  ⭐ 同样的运算，张量版快成千上万倍。原因：")
print("     · for 循环是【逐个】处理，每次都有 Python 解释器开销")
print("     · 张量运算走 C++ / CUDA，一次处理整批数据，还能向量化指令")
print()
print("  ⚠️ 但注意：n 很小时差距没那么夸张，因为")
print("     启动并行运算本身有固定开销（这就是「加速比随 n 增长」的原因）")
print()
print("  💡 结论：写深度学习代码，【永远不要用 for 循环遍历样本】。")
print("     能写成矩阵运算就写成矩阵运算 —— 这就是「矢量化」。")
print()


#%% 实验 7：正态分布与平方损失的关系
print("--- 实验 7：为什么平方损失是「对的」---")
print()

print("  教材最后讲了一个重要的联系：")
print()
print("     假设 y = ŷ + ε，其中 ε 是高斯噪声 ε ~ N(0, σ²)")
print()
print("     那给定 x 时 y 的概率密度是：")
print("        P(y|x) = 1/(σ√(2π)) · exp( -(y-ŷ)² / (2σ²) )")
print()
print("     取负对数（我们要最大化似然 = 最小化负对数似然）：")
print("        -log P(y|x) = (y-ŷ)² / (2σ²) + log(σ√(2π))")
print()
print("     ⭐ 后一项跟 w 无关（常数），前一项正比于 (y-ŷ)²")
print()
print("     所以【最小化平方误差】≡【最大化高斯似然】")
print()


def gauss_nll(resid, sigma):
    n = len(resid)
    return (resid ** 2).sum() / (2 * sigma ** 2) + n * math.log(sigma * math.sqrt(2 * math.pi))


resid = torch.tensor([0.1, -0.2, 0.3, 0.0, -0.5, 0.4])
sigma = 0.5
print(f"  一组残差: {[round(v, 2) for v in resid.tolist()]}")
print(f"  σ = {sigma}")
print()
print(f"  {'平方和':>10}  {'负对数似然':>14}")
print("  " + "-" * 28)
for s in [0.0, 0.1, 0.5, 1.0, 2.0]:
    print(f"  {s:>10.2f}  {gauss_nll(torch.full((1,), s ** 0.5), sigma):>14.4f}")
print()
print("  ⭐ 平方和越小，负对数似然越小 —— 严格单调")
print("     所以「最小化平方损失」和「最大化似然」是同一个目标")
print()

print("  ── 顺便：如果 σ 也未知，它的最大似然估计是什么？──")
print()
resid = torch.tensor([0.1, -0.2, 0.3, 0.0, -0.5, 0.4])
print("  对 σ 求导令为 0，可得   σ² = 平均残差平方")
print()
mse_like = (resid ** 2).mean()
print(f"     残差平方和/n = {mse_like.item():.6f}")
print(f"     开方         = {mse_like.item() ** 0.5:.6f}   ← 这就是 σ 的最大似然估计")
print()
print(f"  {'σ':>8}  {'负对数似然':>14}")
print("  " + "-" * 26)
for sg in [0.1, 0.2, 0.303, 0.4, 0.6, 1.0]:
    print(f"  {sg:>8.3f}  {gauss_nll(resid, sg):>14.4f}")
print()
print("  ⭐ 在 0.303 附近最小 —— 跟理论值一致")
print()


#%% 实验 8：小测验 —— 先猜再验证
print("=" * 64)
print(" 小测验（先在心里答，再改代码验证）")
print("=" * 64)
print("""
  Q1. ŷ = wᵀx + b 里，如果 x 是 (n, d) 的矩阵，
      w 是 (d,)，那 ŷ 是什么形状？

  Q2. 平方损失为什么除以 2？（不影响最优解的话，为什么还要除？）

  Q3. 解析解 (XᵀX)⁻¹Xᵀy 在什么情况下会失败？

  Q4. 小批量 SGD 里，batch_size 越大越好吗？为什么？

  Q5. 为什么说「最小化平方损失」等价于「最大化高斯似然」？
      关键假设是什么？
""")
print()
print("  --- 逐条验证 / 补充 ---")
print()

print("  Q1: (n,) —— n 个预测值，特征维被加权求和掉")
print(f"      实测: X(3,2) @ w(2,) → {tuple((torch.randn(3,2) @ torch.randn(2)).shape)}")
print()

print("  Q2: 求导时 (x²)' = 2x，除以 2 正好抵消，梯度式子变干净")
print("      ∂L/∂ŷ = (ŷ-y)/n，没有多余的 2")
print()

print("  Q3: 三种情况：")
print("      ① XᵀX 不可逆 —— 特征之间完全共线（比如 x₂ = 2·x₁）")
print("      ② 样本数 < 特征数 —— n < d，方程比未知数少")
print("      ③ 特征维度极高 —— (XᵀX)⁻¹ 复杂度 O(d³)，算不动")
print()

# 实测共线
print("      实测①：造一个完全共线的 X")
X_bad = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])   # 第二列 = 2 × 第一列
print(f"         X = {X_bad.tolist()}")
_det = torch.linalg.det(X_bad.T @ X_bad).item()
print(f"         det(XᵀX) = {_det:.3e}   绝对值 {abs(_det):.1e} → 就是 0")
print(f"         秩 = {torch.linalg.matrix_rank(X_bad.T @ X_bad).item()}   ← 应该满秩是 2")
print()

print("  Q4: 不是。batch 越大梯度越准，但每步也越慢，而且")
print("      泛化未必更好（小批量的噪声反而有正则化效果）")
print("      常见取 32 / 64 / 128 / 256 —— 显存和速度的折中")
print()

print("  Q5: 关键假设是【噪声服从高斯分布】，而且【各样本独立】")
print("      如果噪声是拉普拉斯分布，对应的就不是平方损失，")
print("      而是绝对值损失（L1）—— 这就是「损失函数的选择 = 对噪声的假设」")
print()


#%% 第 N 格：本节小结
print("=" * 64)
print(" 3.1 线性回归 —— 小结")
print("=" * 64)
print("""
【三个零件】

  ① 模型   ŷ = wᵀx + b        （矩阵乘法，2.3）
  ② 损失   L = (1/2n)·Σ(ŷ-y)²  （来自高斯似然，2.6）
  ③ 优化   梯度下降 w ← w - lr·∂L/∂w  （2.4 + 2.5）

【两个公式（务必记住）】

  ∂L/∂w = (1/n)·Xᵀ(ŷ - y)
  ∂L/∂b = (1/n)·Σ(ŷ - y)

  第 3.2 节「从零实现」你就要亲手敲这两行。

【两条路，各有用途】

  解析解  w* = (XᵀX)⁻¹Xᵀy
      优点：一步到位，无超参数
      缺点：O(d³)、可能不可逆、神经网络没这玩意

  梯度下降
      优点：通用，能扩展到任何模型
      缺点：要调学习率、要迭代很多轮

  ⭐ 深度学习里【永远用梯度下降】，因为神经网络没有解析解。

【一句话理解矢量化】

  不要用 for 循环遍历样本 —— 写成矩阵运算，快成千上万倍。

【第二章的工具在这里全用上了】

  2.1 张量      →  X, y, w, b 的数据容器
  2.2 pandas    →  读数据（后面处理真实数据集时用）
  2.3 线性代数  →  X @ w、XᵀX、求解
  2.4 微积分    →  偏导数公式
  2.5 自动微分  →  backward() 自动算这些偏导数
  2.6 概率      →  平方损失 = 高斯最大似然

【下一节：3.2 从零开始实现】

  会亲手写：
     · 造数据
     · 定义模型
     · 写损失函数
     · 写小批量采样器
     · 手写梯度下降（不用 optimizer）

  做完 3.2，再回头看这一节的公式，会特别通透。
""")


# ────────────────────────────────────────────────────────────────────
#  📐 Q1 的形状对照表（务必记住）
#
#     场景          数学写法            代码写法           结果形状
#     ─────────────────────────────────────────────────────────────
#     一个样本      ŷ = wᵀx + b         w @ x + b          标量
#                                       torch.dot(w,x)+b
#
#     一批样本      ŷ = Xw + b·1        X @ w + b          (n,)
#
#     ❌ 错误写法        —              w.T @ X            报错 / 静默算错
#
#  🔑 形状规则
#
#     X 是 (n, d)    n 个样本，每行一个样本，d 个特征
#     w 是 (d,)      每个特征一个权重
#
#     (n, d) @ (d,)  →  (n,)      中间那个 d 消掉
#     └──┬──┘   └┬┘     └┬┘
#       a, b     b       a        (a,b) @ (b,c) → (a,c)
#
#     ⭐ n 个样本各出一个预测 → 结果就是 n 个数
#
#  🤔 为什么数学写 wᵀx，代码写 X @ w？
#
#     数学惯例：  向量是【列向量】(d,1)
#                求内积必须把左边转成行：(1,d) @ (d,1) = (1,1)
#                所以才写成 wᵀx
#
#     代码现实：  数据是【一批样本】，X 的每行就是样本
#                (n,d) @ (d,1) = (n,1)
#                X 的每行本来就对齐，不需要转置
#
#     ⭐ 数学的转置是「为了凑维度」；
#        代码里 X 每行本来就是样本，维度天然对齐，不用转。
#
#     ⚠️ 教材写的 wᵀx 是【单个样本】的写法，
#        不是让你在代码里写 w.T @ X！
#
#  🚨 危险点：n == d 时会「不报错但算错」
#
#     如果 X 恰好是 (3,3)、w 是 (3,)：
#        (1,3) @ (3,3) → (1,3)     ← 不报错！
#
#     但它算的是「每一列和 w 的内积」（按列加权求和），
#     而你要的是「每一行和 w 的内积」（每个样本一个预测）——方向反了。
#
#     实测（X 是 3×3）：
#        用 wᵀ @ X 算 = [-6.7349, 0.0043, -3.9498]   ← 错
#        正确的 X @ w = [-6.3119, 0.7269, -4.1967]
#
#     ⭐ 这是最危险的错误类型：跑得通、不报错、结果全错。
#        所以每次写完矩阵乘法，【先打印 shape 确认】。
# ────────────────────────────────────────────────────────────────────
#  📝 Q1 小结
# ────────────────────────────────────────────────────────────────────
#
#     ŷ 的形状是 (n,)，因为 n 个样本各出一个预测值。
#
#     wᵀx 是【单个样本】的写法，结果是标量 (1,1)。
#
#     代码里 n 个样本一起算，写 X @ w，不要写 w.T @ X。
#
#     写错的形式在 n != d 时直接报错（好事），
#     在 n == d 时静默算错（灾难）；
#     所以【形状检查比结果检查更早、更可靠】。
#
# ────────────────────────────────────────────────────────────────────
