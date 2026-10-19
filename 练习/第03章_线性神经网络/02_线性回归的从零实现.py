#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 64)
print(" 3.2 线性回归的从零开始实现 —— 练习")
print("=" * 64)
print("""
【教材】https://zh-v2.d2l.ai/chapter_linear-networks/linear-regression-scratch.html

【这一节到底在干什么】
  3.1 里我们用了一行 optimizer.step() 就更新了参数。
  但 optimizer 内部到底做了什么？这一节【全部自己写】。

  不用：
     ❌ nn.Module        （模型类）
     ❌ nn.Linear        （现成的线性层）
     ❌ torch.optim.SGD  （现成的优化器）
     ❌ DataLoader       （现成的数据加载器）

  只用：张量 + autograd + 自己写的 for 循环。

【五个零件，自己造一遍】
  ① 造数据    synthetic_data()
  ② 采样器    data_iter()        ← 自己实现打乱 + 切批
  ③ 模型      linreg()
  ④ 损失      squared_loss()
  ⑤ 优化器    sgd()              ← 自己实现 w -= lr*grad

【⚠️ 一个必须先讲清楚的事：形状不一样了】

  3.1 的写法（一维）        3.2 的写法（列向量）
  ─────────────────────    ─────────────────────
  w 是 (2,)                w 是 (2, 1)
  y 是 (n,)                y 是 (n, 1)
  ŷ 是 (n,)                ŷ 是 (n, 1)
  X @ w → (n,)             X @ w → (n, 1)

  两种都对！3.2 用列向量是因为这样更接近数学写法 ŷ = Xw（w 是列向量）。
  好处：形状永远对得上，不会出现 (n,) 和 (n,1) 广播出 (n,n) 的意外。
  坏处：多一个维度，打印时是 [[...]] 而不是 [...]

【跟 3.1 的关系】
  3.1：理解「模型/损失/优化」三个零件在干什么
  3.2：亲手把它们造出来
  3.3：用 nn.Module + optimizer 重写一遍（简洁实现）

  依次是：懂原理 → 自己实现 → 用现成的

【怎么用】光标放在某格里按 Shift+Enter。第 0 格必须先跑。
""")

import random
import torch
from torch import Tensor  # 写 : Tensor 类型标注时要用

torch.manual_seed(42)
print("torch 版本:", torch.__version__)


#%% 实验 1：造数据 —— 注意 y 的形状是 (n, 1)
print("--- 实验 1：synthetic_data ---")
print()


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print("  调用了 synthetic_data(true_w, true_b, 1000)")
print()
print(f"  features.shape = {tuple(features.shape)}      ← 1000 个样本，2 个特征")
print(f"  labels.shape   = {tuple(labels.shape)}         ← ⚠️ 是 (1000, 1)，不是 (1000,)")
print()
print("  第 1 个样本:")
print(f"     features[0] = {features[0].tolist()}")
print(f"     labels[0]   = {labels[0].tolist()}        ← 是个长度 1 的列表")
print()
print("  ⚠️ 为什么 y 要 reshape 成 (n,1)？")
print("     因为 w 是 (2,)，X @ w 得到 (n,)，再加上 b 还是 (n,)")
print("     reshape((-1, 1)) 把它变成列向量，跟后面 w 改成 (2,1) 配套")
print("     （-1 的意思是「这一维自动算」）")
print()
print("  验证：labels 和 features 一一对应吗？")
manual = features[0, 0] * true_w[0] + features[0, 1] * true_w[1] + true_b
print(f"     手算: {manual.item():.6f}")
print(f"     标签: {labels[0].item():.6f}   （差一点是噪声）")
print(f"     差 = {abs(manual.item() - labels[0].item()):.6f}   ← 噪声量级 0.01，对得上")
print()


#%% 实验 2：小批量采样器 —— 自己实现「打乱 + 切批」
print("--- 实验 2：data_iter 采样器 ---")
print()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)          # 用 Python 的 random 打乱下标
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:i + batch_size])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
print(f"  调用 data_iter({batch_size}, features, labels)")
print()
print("  ⭐ 关键：这个函数用了 yield，所以它是个【生成器】")
print("     yield 的意思：产出一个值，然后【暂停在这里】，下次再继续")
print("     所以 for 循环每转一圈，它才现算一批，不用一次把所有批都造出来")
print()
print("  取前 2 批看看:")
for k, (X, y) in enumerate(data_iter(batch_size, features, labels)):
    print(f"     第{k+1}批: X.shape={tuple(X.shape)}  y.shape={tuple(y.shape)}")
    print(f"             X 前 3 个样本: {[round(v,3) for v in X[:3,0].tolist()]}")
    print(f"             y 前 3 个标签: {[round(v,3) for v in y[:3].flatten().tolist()]}")
    if k == 1:
        break
print()
print("  一共多少批？")
n_batches = sum(1 for _ in data_iter(batch_size, features, labels))
print(f"     {n_batches} 批 × {batch_size} = {n_batches * batch_size} 个样本")
print()

# ——— 留给你做 ———
print("=" * 52)
print(" ✏️ 自己试：把 random.shuffle(indices) 注释掉，")
print("    连跑两次 data_iter，看输出的第一批是不是一样。")
print("    如果打乱被去掉，每轮顺序固定，会有什么问题？")
print("=" * 52)
print()


#%% 实验 3：初始化参数 —— w 是 (2, 1) 不是 (2,)
print("--- 实验 3：初始化参数 ---")
print()

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print("  w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)")
print(f"     shape = {tuple(w.shape)}")
print(f"     值    = {w.flatten().tolist()}")
print()
print("  b = torch.zeros(1, requires_grad=True)")
print(f"     shape = {tuple(b.shape)}")
print(f"     值    = {b.tolist()}")
print()
print("  ⚠️ 注意 w 的形状是 (2,1)，不是 3.1 里的 (2,)")
print()
print("  验证：用 (2,1) 的 w 算前向，形状是什么？")
X_demo = features[:5] # 等价于 features[0:5]  →  下标 0,1,2,3,4
out = X_demo @ w + b
print(f"     X {tuple(X_demo.shape)} @ w {tuple(w.shape)} + b {tuple(b.shape)}")
print(f"     = {tuple(out.shape)}      ← 跟 labels[:5] 的 {tuple(labels[:5].shape)} 一致 ✅")
print()


#%% 实验 4：模型和损失 —— 两个只有一行的函数
print("--- 实验 4：linreg 和 squared_loss ---")
print()


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


print("  ① linreg(X, w, b) = X @ w + b")
X_batch, y_batch = next(iter(data_iter(batch_size, features, labels)))
pred = linreg(X_batch, w, b)
print(f"     输入 X {tuple(X_batch.shape)} → 输出 {tuple(pred.shape)}")
print()
print("  ② squared_loss(y_hat, y) = (y_hat - y)² / 2")
l = squared_loss(pred, y_batch)
print(f"     输入 pred {tuple(pred.shape)}, y {tuple(y_batch.shape)}")
print(f"     输出 l   {tuple(l.shape)}      ← 每个样本一个损失，还没求和")
print(f"     值 = {[round(v,4) for v in l.flatten().tolist()]}")
print()
print("  ⚠️ 注意 y.reshape(y_hat.shape) 这一句：")
print("     它是为了「万一 y 是 (n,) 而 y_hat 是 (n,1)」时也能对齐。")
print("     实测：y 已经是 (n,1) 时，reshape 不改变任何东西")
print(f"     y_batch.shape = {tuple(y_batch.shape)}, reshape 之后 = {tuple(y_batch.reshape(pred.shape).shape)}")
print()
print("  ⚠️ 另一个陷阱：如果不 reshape，直接用 (n,) 减 (n,1) 会广播成 (n,n)！")
y_bad = y_batch.flatten()          # (10,)
try:
    bad = (pred - y_bad)
    print(f"     pred {tuple(pred.shape)} - y {tuple(y_bad.shape)} → {tuple(bad.shape)}   ❌ 变成了 10×10 矩阵！")
except RuntimeError as e:
    print(f"     报错: {e}")
print("     所以 reshape 是护栏，别删。")
print()


#%% 实验 5：手写优化器 —— 三行替代 optimizer.step()
print("--- 实验 5：自己实现 sgd ---")
print()


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            g = param.grad
            assert g is not None, "梯度是 None —— 是不是忘了 backward()？"
            param -= lr * g / batch_size
            if param.grad is not None:
                param.grad.zero_()


print("  这就是全部代码：")
print()
print("     def sgd(params, lr, batch_size):")
print("         with torch.no_grad():")
print("             for param in params:")
print("                 param -= lr * param.grad / batch_size")
print("                 param.grad.zero_()")
print()
print("  ⭐ 三个细节，每个都有原因：")
print()
print("  ① with torch.no_grad()")
print("     因为 param -= ... 是【修改参数本身】。")
print("     如果不禁用梯度，这个「减法」会被记进计算图，")
print("     下次 backward 就会沿着更新操作继续往回求导 —— 完全错误。")
print()
print("  ② 除以 batch_size")
print("     因为训练时用的是 l.sum().backward()，梯度是【求和】的梯度。")
print("     要变回「平均梯度」，就得除以批大小。")
print("     （如果用 l.mean()，就不用在这里除）")
print()
print("  ③ param.grad.zero_()")
print("     清空梯度，准备下一批。等价于 optimizer.zero_grad()。")
print("     注意它是【原地清零】，设成 0 而不是 None。")
print()

print("  实测走一步：")
torch.manual_seed(1)
w_test = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b_test = torch.zeros(1, requires_grad=True)
before_w = w_test.flatten().tolist()
before_b = b_test.tolist()

l_test = squared_loss(linreg(X_batch, w_test, b_test), y_batch)
l_test.sum().backward()
assert w_test is not None
assert b_test is not None
gw = w_test.grad
gb = b_test.grad
assert gw is not None and gb is not None, "backward 之后应该有梯度"
print(f"     backward 后 w.grad = {gw.flatten().tolist()}")
print(f"                 b.grad = [{gb.item():.4f}]")
sgd([w_test, b_test], 0.03, batch_size)
print(f"     sgd 之后   w: {before_w} → {w_test.flatten().tolist()}")
print(f"                b: {before_b} → {b_test.tolist()}")
print(f"     梯度清空了吗？ w.grad = {w_test.grad}")
print()


#%% 实验 6：完整训练 —— 五步循环
print("--- 实验 6：训练 ---")
print()
"""
def synthetic_data(w, b, num_examples):
    生成 y = Xw + b + 噪声
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
"""
torch.manual_seed(42)
features, labels = synthetic_data(true_w, true_b, 1000)
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.03
num_epochs = 3

print(f"  lr = {lr}, num_epochs = {num_epochs}, batch_size = {batch_size}")
print(f"  真实值: w={true_w.tolist()}  b={true_b}")
print()
print(f"  {'epoch':>6}  {'平均loss':>12}  {'w':<26}  {'b':>9}")
print("  " + "-" * 60)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = squared_loss(linreg(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels)
    print(f"  {epoch+1:>6}  {train_l.mean().item():>12.8f}  "
          f"{str([round(v,4) for v in w.flatten().tolist()]):<26}  {b.item():>9.4f}")

print()
print("  ⭐ 注意这个循环跟 3.1 的对比：")
print()
print("     3.1（用 optimizer）              3.2（自己写）")
print("     ─────────────────────────────    ─────────────────────────────")
print("     loss = ((pred-y)**2).mean()/2    l = squared_loss(...)")
print("     loss.backward()                  l.sum().backward()")
print("     optimizer.step()                 sgd([w,b], lr, batch_size)")
print("     optimizer.zero_grad()            （已包含在 sgd 里）")
print()


#%% 实验 7：验证结果
print("--- 实验 7：学出来的参数对不对 ---")
print()

print(f"  真实 w = {true_w.tolist()}")
print(f"  学出 w = {[round(v,6) for v in w.flatten().tolist()]}")
print(f"  误差   = {[round(v,6) for v in (true_w - w.flatten()).tolist()]}")
print()
print(f"  真实 b = {true_b}")
print(f"  学出 b = {b.item():.6f}")
print(f"  误差   = {true_b - b.item():.6f}")
print()
print("  ⭐ 误差都在 1e-3 量级 —— 因为噪声 σ=0.01，样本 1000 个，")
print("     理论上误差应该在 σ/√n 量级，也就是 0.01/31.6 ≈ 3e-4。对得上。")
print()


#%% 实验 8：手写 sgd 和 optimizer 到底差在哪
print("--- 实验 8：自己写的 sgd vs torch.optim.SGD ---")
print()

print("  ⚠️ 先纠正一个常见误解：这两个【不】完全等价。")
print()
print("     torch.optim.SGD 的 step 做的事：")
print("         param -= lr * param.grad           <- 不除以 batch_size")
print()
print("     教材的 sgd 做的事：")
print("         param -= lr * param.grad / batch_size   <- 自己除了")
print()
print("     ⭐ 所以手写版的实际步长 = optim 版的 1/batch_size。")
print()

torch.manual_seed(3)
Xa = torch.randn(10, 2)
ya = torch.randn(10, 1)

w_h = torch.tensor([[0.1], [0.2]], requires_grad=True)
b_h = torch.zeros(1, requires_grad=True)
w_o = w_h.detach().clone().requires_grad_(True)
b_o = b_h.detach().clone().requires_grad_(True)

init_w = w_h.detach().flatten().tolist()
init_b = b_h.item()

print("  用【同一份数据、同一个初始值】严格对比：")
print()
print(f"     初始 w = {init_w},  b = {init_b}")
print()
"""def squared_loss(y_hat, y):
    均方损失
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
"""
l1 = squared_loss(linreg(Xa, w_h, b_h), ya)
l1.sum().backward()
l2 = squared_loss(linreg(Xa, w_o, b_o), ya)
l2.sum().backward()

gh = w_h.grad
go = w_o.grad
assert gh is not None and go is not None, "backward 之后应该有梯度"
print(f"  ① 两边 backward（都用 l.sum()）后，梯度一致吗？ {torch.allclose(gh, go)}")
print(f"     w.grad = {[round(v,6) for v in gh.flatten().tolist()]}")
print()

sgd([w_h, b_h], 0.03, 10)
opt = torch.optim.SGD([w_o, b_o], lr=0.03)
opt.step()

d_hand = [round(v - k, 8) for v, k in zip(w_h.detach().flatten().tolist(), init_w)]
d_opt = [round(v - k, 8) for v, k in zip(w_o.detach().flatten().tolist(), init_w)]

print("  ② 各自更新一步，看【步长】（不是最终值）：")
print()
print(f"     手写 sgd 步长  w: {d_hand}     b: {round(b_h.item()-init_b, 8)}")
print(f"     optim 步长     w: {d_opt}     b: {round(b_o.item()-init_b, 8)}")
print(f"     比值           : {[round(a/b, 6) if b else None for a, b in zip(d_hand, d_opt)]}")
print()
print(f"  ⭐ 比值 = 0.1 = 1/{10}，正好是 1/batch_size")
print()

# —— 再验证：optim 那边改用 l.mean()，就跟手写版完全一致 ——
w_o2 = torch.tensor([[0.1], [0.2]], requires_grad=True)
b_o2 = torch.zeros(1, requires_grad=True)
l3 = squared_loss(linreg(Xa, w_o2, b_o2), ya)
l3.mean().backward()
opt2 = torch.optim.SGD([w_o2, b_o2], lr=0.03)
opt2.step()
d2 = [round(v - k, 8) for v, k in zip(w_o2.detach().flatten().tolist(), init_w)]

print("  ③ 如果 optim 那边把 l.sum() 换成 l.mean() 呢？")
print()
print(f"     optim + l.mean() 步长 w: {d2}")
print(f"     跟手写版完全相等吗？ {all(abs(a-b) < 1e-7 for a, b in zip(d_hand, d2))}")
print()
print("  ⭐ 相等！因为 l.mean() 的梯度本来就已经是平均值了。")
print()

print("  📋 四种配对方式（一定要记牢）")
print()
print("     梯度来源       优化器        效果")
print("     ──────────────────────────────────────────────────────────")
print("     l.sum()        sgd 里除 bs   步长 = 平均梯度 * lr     OK 教材")
print("     l.mean()       optim.SGD     同上                     OK 等价")
print("     l.sum()        optim.SGD     步长放大 batch 倍        X  学崩")
print("     l.mean()       sgd 里也除    步长缩小 batch 倍        X  学不动")
print()
print("  ⚠️ 最容易踩的坑：把 3.1 的 l.mean() 写法搬过来，")
print("     却继续用教材的 sgd（它还在除 batch_size）——")
print("     等于 lr 被除了两次批大小，学习会慢到看不出来。")
print()


#%% 实验 9：小测验
print("=" * 64)
print(" 小测验（先在心里答，再改代码验证）")
print("=" * 64)
print("""
  Q1. 为什么 labels 的形状是 (1000, 1) 而不是 (1000,)？

  Q2. data_iter 里的 yield 和 return 有什么区别？

  Q3. sgd 里为什么必须写 with torch.no_grad()？

  Q4. 为什么反向传播要写 l.sum().backward()，不能直接 l.backward()？

  Q5. sgd 里 param.grad.zero_() 和 optimizer.zero_grad() 等价吗？
""")
print()
print("  --- 逐条验证 / 补充 ---")
print()

print("  Q1: 因为 w 是 (2,1)，X@w 是 (n,1)。若 y 是 (n,) 会跟 (n,1) 广播成 (n,n)。")
print(f"      实测: (10,1) - (10,) → {tuple((torch.randn(10,1) - torch.randn(10)).shape)}")
print()

print("  Q2: return 一次返回就结束函数；yield 产出后【暂停】，下次接着跑。")
print(f"      data_iter 是生成器吗？ {type(data_iter(10, features, labels)).__name__}")
print()

print("  Q3: 因为 param -= lr*grad 是原地修改参数。如果不禁用梯度，")
print("      这个操作会被记录进计算图，下次 backward 会沿着它继续求导 → 错。")
print()

print("  Q4: l 的形状是 (batch,1)，不是标量。backward() 要求标量输出。")
print(f"      实测 l.shape = {tuple(squared_loss(linreg(X_batch, w, b), y_batch).shape)}")
try:
    squared_loss(linreg(X_batch, w, b), y_batch).backward()
except RuntimeError as e:
    print(f"     直接 backward() → ❌ {str(e)[:60]}...")  # noqa: BLE001
print()

print("  Q5: 等价（都清空梯度）。区别：")
print("      param.grad.zero_()   → 原地清零，.grad 是 tensor([0.])")
print("      optimizer.zero_grad()→ 默认设成 None，省内存")
print()


#%% 第 N 格：本节小结
print("=" * 64)
print(" 3.2 线性回归的从零开始实现 —— 小结")
print("=" * 64)
print("""
【五个零件，都是几行代码】

  ① 造数据     synthetic_data(w, b, n)   →  X (n,d), y (n,1)
  ② 采样器     data_iter(bs, X, y)       →  yield 一批一批
  ③ 模型       linreg(X, w, b)           →  X @ w + b
  ④ 损失       squared_loss(y_hat, y)    →  (ŷ-y)²/2
  ⑤ 优化器     sgd(params, lr, batch)    →  param -= lr*grad/batch

【三个必须记住的细节】

  · labels 是 (n,1)，w 是 (d,1) —— 列向量风格，形状永远对齐
  · l.sum().backward() —— 因为 l 是 (batch,1)，不是标量
  · with torch.no_grad() —— 更新参数时必须禁用梯度记录

【跟 optimizer 的关系】

  optimizer.step()  内部就是：param -= lr * grad
  optimizer.zero_grad() 内部就是：param.grad = None

  ⭐ 你现在知道那一行背后是什么了。

【这一节的意义】

  第 3 章后面所有模型（softmax、MLP、CNN）都是这个五步循环。
  只是把 linreg 换成更复杂的函数。

  做完这一节，你看任何训练代码都不会再觉得「黑箱」。

【下一节：3.3 简洁实现】

  同样的东西，用 nn.Module + DataLoader + optimizer 重写。
  你会发现：代码少了 2/3，但每一行你都知道它在干什么。
""")
