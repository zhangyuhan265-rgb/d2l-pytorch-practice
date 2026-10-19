#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 64)
print(" 3.3 线性回归的简洁实现 —— 练习")
print("=" * 64)
print("""
【教材】https://zh-v2.d2l.ai/chapter-linear-networks/linear-regression-concise.html

【这一节在做什么】
  3.2 我们把所有零件都手写了一遍。
  3.3 把它们全部换成 PyTorch 现成的 —— 代码少一大半。

  这不是「偷懒」，是【工程实践】：现成的更稳、更快、更通用。

【三节的递进】

              3.1            3.2              3.3
  模型       手写公式        手写函数          nn.Linear
  损失       手写公式        手写函数          nn.MSELoss
  优化       optimizer      手写 sgd          optimizer
  采样       手动切片        手写 data_iter    DataLoader

【这一节的重点：三个坑】

  ⚠️ 坑1：nn.Linear 的 weight 形状是 (out, in)，是【转置】的
  ⚠️ 坑2：nn.MSELoss 里【没有】1/2，所以梯度是 3.2 的 2 倍
  ⚠️ 坑3：zero_grad 的位置，放错参数就永远不动

【⚠️ 本文件的一个写法约定（很重要）】

  教材写的是：
      net = nn.Sequential(nn.Linear(2, 1))
      net[0].weight.data.normal_(0, 0.01)       ← 访问参数

  但 net[0] 的类型是 Module，Module 没有 .weight 属性，
  所以 VS Code 的类型检查器（Pylance）会在每个 net[0].weight 上标红。

  ⭐ 本文件改成：先把 Linear 存一个变量，再套进 Sequential：
      linear = nn.Linear(2, 1)
      net = nn.Sequential(linear)
      linear.weight.data.normal_(0, 0.01)       ← 用 linear 访问，类型明确

  功能完全一样，但类型检查器满意了。

【怎么用】光标放在某格里按 Shift+Enter。第 0 格必须先跑。
""")

import random
import torch
from torch import nn
from torch.utils import data

torch.manual_seed(42)
print("torch 版本:", torch.__version__)


#%% 实验 1：造数据
print("--- 实验 1：造数据 ---")
print()


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print(f"  features.shape = {tuple(features.shape)}")
print(f"  labels.shape   = {tuple(labels.shape)}")
print()
print("  ⭐ 造数据这一步 3.3 没得省 —— 它是「问题定义」，不是模型零件。")
print()


#%% 实验 2：DataLoader —— 替代手写的 data_iter
print("--- 实验 2：TensorDataset + DataLoader ---")
print()

dataset = data.TensorDataset(features, labels)
print("  ① dataset = data.TensorDataset(features, labels)")
print(f"     len(dataset) = {len(dataset)}      ← 样本数")
print()
item = dataset[0]
print("     dataset[0] = 一个元组，第 0 个样本的 (X, y)")
print(f"        X = {[round(v,3) for v in item[0].tolist()]}")
print(f"        y = {[round(v,3) for v in item[1].tolist()]}")
print()
print("  ⭐ TensorDataset 就是把几个张量按第 0 维「捆」在一起，")
print("     取第 i 项会同时取出每个张量的第 i 行。")
print()

batch_size = 10
data_iter = data.DataLoader(dataset, batch_size, shuffle=True)
print("  ② data_iter = data.DataLoader(dataset, batch_size, shuffle=True)")
print(f"     len(data_iter) = {len(data_iter)}     ← 批数 = ceil(1000/10) = 100")
print()
print("  ③ 取一批看看：")
for Xb, yb in data_iter:
    print(f"     X.shape = {tuple(Xb.shape)}")
    print(f"     y.shape = {tuple(yb.shape)}")
    break
print()

print("  ④ 跟 3.2 手写的 data_iter 对比：")
print()
print("     3.2 手写：")
print("         indices = list(range(n))")
print("         random.shuffle(indices)")
print("         for i in range(0, n, bs):")
print("             yield features[idx], labels[idx]     ← yield 一批一批")
print()
print("     3.3 现成：")
print("         DataLoader(dataset, bs, shuffle=True)    ← 一行")
print()
print("  ⭐ 功能完全一样：打乱 + 切批 + 一批一批给。")
print("     DataLoader 更快，因为它用 C++ 多线程在后台预取下一批。")
print()


#%% 实验 3：nn.Linear —— 注意权重的形状是【转置】的！
print("--- 实验 3：nn.Linear(2, 1) ---")
print()
'''输入（2 个特征）
    面积 ──┐
    房龄 ──┤
           ▼
     ┌───────────┐
     │ nn.Linear │
     │   (2, 3)  │
     └───────────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
  成交价 月租金 单价      ← 输出（3 个预测）'''
linear = nn.Linear(2, 1)
print("  linear = nn.Linear(2, 1)")
print("             ↑           ↑")
print("        输入特征数   输出特征数")
print()
print(f"  linear.weight.shape = {tuple(linear.weight.shape)}      ← ⚠️ 是 (1, 2)，不是 (2, 1)！")
print(f"  linear.bias.shape   = {tuple(linear.bias.shape)}")
print(f"  linear.weight = {linear.weight}")
print(f"  linear.bias   = {linear.bias}")
print()
print("  ⭐ 为什么是 (1,2)？因为 nn.Linear 内部算的是：")
print()
print("        y = x @ W.T + b")
print()
print("     你给它 W 是 (out, in) = (1, 2)，它内部转置成 (2, 1) 再乘。")
print()

x_demo = features[:5]
out = linear(x_demo)
print(f"  验证：X {tuple(x_demo.shape)} 输入 linear → 输出 {tuple(out.shape)}")
print()
manual = x_demo @ linear.weight.T + linear.bias
print("  手工复现 x @ W.T + b：")
print(f"     跟 linear(x) 一致吗？ {torch.allclose(out, manual)}   ✅")
print()

print("  ⚠️⚠️ 这就是 3.2 和 3.3 的【参数形状约定不同】：")
print()
print("     3.2 手写：  w 是 (2, 1)，算 X @ w")
print("     3.3 现成：  W 是 (1, 2)，算 X @ W.T")
print()
print("     ⭐ 两者数学上完全相同，只是「谁是行谁是列」的约定不同。")
print()


#%% 实验 4：nn.Sequential —— 把层串起来
print("--- 实验 4：nn.Sequential ---")
print()

net = nn.Sequential(linear)# 注：此处只有 1 层，Sequential 无实际作用，仅为与后续多层网络保持写法一致
print("  net = nn.Sequential(linear)")
print(f"     {net}")
print()
print("  ⭐ Sequential 就是「按顺序串起来的容器」：")
print("     输入 → 第1层 → 第2层 → ... → 输出")
print()
print("  只有一层时，看起来多此一举。但后面多层网络就是这样写的：")
print("     nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))")
print()
print(f"  len(net) = {len(net)}      ← 有一层")
print(f"  type(net[0]) = {type(net[0]).__name__}")
print()
print("  ⚠️ 注意：net[0] 的类型是 Module，不是 Linear。")
print("     所以 net[0].weight 在类型检查器看来是错的（Module 没这属性）。")
print("     ⭐ 这就是为什么本文件用 linear.weight 而不是 net[0].weight。")
print()
print("  ⭐ 前向等价性验证：")
print(f"     net(x) 跟 linear(x) 一样吗？ {torch.allclose(net(x_demo), linear(x_demo))}")
print()


#%% 实验 5：初始化参数 —— 用 .data 或 no_grad
print("--- 实验 5：初始化模型参数 ---")
print()

print("  PyTorch 自动做的默认初始化：")
print(f"     weight = {linear.weight.data.flatten().tolist()}")
print(f"     bias   = {linear.bias.data.tolist()}")
print()

linear.weight.data.normal_(0, 0.01)
linear.bias.data.fill_(0)
print("  手动改成「均值 0、标准差 0.01」和「bias 全 0」：")
print(f"     weight = {[round(v,6) for v in linear.weight.data.flatten().tolist()]}")
print(f"     bias   = {linear.bias.data.tolist()}")
print()
print("  ⚠️ 三个细节：")
print()
print("  ① 为什么有 .data？")
print("     .data 返回「不追踪梯度的版本」，可以直接改值。")
print("     如果写 linear.weight.normal_(0, 0.01)，会被当成一次运算记进计算图。")
print()
print("  ② 末尾的下划线 normal_ / fill_ 是什么意思？")
print("     下划线 = 原地操作，直接改这个张量。")
print("     normal(0,0.01) 返回新张量，normal_(0,0.01) 直接改自己。")
print()
print("  ③ 更现代的写法（推荐）：")
print("     with torch.no_grad():")
print("         linear.weight.normal_(0, 0.01)")
print("         linear.bias.fill_(0)")
print("     ⭐ 用 no_grad 比 .data 更安全（.data 会绕过安全检查）。")
print()

print("  验证两种写法等价：")
net_a = nn.Linear(2, 1)
net_b = nn.Linear(2, 1)
torch.manual_seed(7)
#print(net_a.weight.data.tolist())
#print(net_a.bias.data.tolist())
net_a.weight.data.normal_(0, 0.01)
net_a.bias.data.fill_(0)
#print(net_a.weight.data.tolist())
#print(net_a.bias.data.tolist())
torch.manual_seed(7)
with torch.no_grad():
    net_b.weight.normal_(0, 0.01)
    net_b.bias.fill_(0)
print(f"     .data 版 vs no_grad 版，weight 一样吗？ {torch.allclose(net_a.weight, net_b.weight)}")
print()


#%% 实验 6：nn.MSELoss —— 注意它【没有】1/2
print("--- 实验 6：nn.MSELoss ---")
print()


def squared_loss_32(y_hat, y):
    """3.2 用的（含 1/2）"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


mse = nn.MSELoss()
qh = torch.tensor([[1.0], [2.0], [3.0]])
qt = torch.tensor([[0.5], [2.5], [2.0]])
print(f"  样例：y_hat = {qh.flatten().tolist()},  y = {qt.flatten().tolist()}")
print()
print(f"  3.2 的 squared_loss(...).mean()  = {squared_loss_32(qh,qt).mean().item():.8f}")
print(f"  nn.MSELoss()                     = {mse(qh,qt).item():.8f}")
print(f"  比值 = {mse(qh,qt).item() / squared_loss_32(qh,qt).mean().item():.4f}")
print()
print("  ⭐ nn.MSELoss 的公式是：")
print()
print("        L = (1/n) · Σ (ŷᵢ - yᵢ)²")
print()
print("     只有【平均】，没有那个 1/2。")
print()

print("  ⚠️⚠️ 后果：梯度差 2 倍")
print()
torch.manual_seed(3)
Xa = torch.randn(10, 2)
Ya = torch.randn(10, 1)

w1 = torch.tensor([[0.1],[0.2]], requires_grad=True)
b1 = torch.zeros(1, requires_grad=True)
l1 = squared_loss_32(Xa @ w1 + b1, Ya).mean()
l1.backward()
assert w1.grad is not None
g1 = w1.grad.clone()

w2 = torch.tensor([[0.1],[0.2]], requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
l2 = mse(Xa @ w2 + b2, Ya)
l2.backward()
assert w2.grad is not None
g2 = w2.grad.clone()

print(f"     squared_loss 的 w.grad = {[round(v,6) for v in g1.flatten().tolist()]}")
print(f"     MSELoss 的 w.grad      = {[round(v,6) for v in g2.flatten().tolist()]}")
print(f"     比值 = {[round(a/b,4) for a,b in zip(g2.flatten().tolist(), g1.flatten().tolist())]}")
print()
print("  ⭐ 同样 lr 下，3.3 的实际步子就是 3.2 的 2 倍。")
print("     教材两边都用 lr=0.03，都能收敛 —— 因为 0.03 还没超出稳定范围。")
print("     但这【不是等价】，是恰好都稳。")
print()


#%% 实验 7：优化器 —— net.parameters() 直接喂进去
print("--- 实验 7：torch.optim.SGD ---")
print()

print("  trainer = torch.optim.SGD(net.parameters(), lr=0.03)")
print("                            ↑")
print("                      不是 [w, b]，是 net.parameters()")
print()
print("  实测 net.parameters() 返回什么：")
print(f"     type = {type(net.parameters()).__name__}")
for i, p in enumerate(net.parameters()):
    print(f"     params[{i}]: shape {tuple(p.shape)}  requires_grad={p.requires_grad}")
print()
print("  ⭐ 生成器，里面是模型的所有参数（weight 和 bias）。")
print("     这跟 3.2 的 sgd([w, b], ...) 是同一个约定 —— 都要一个可迭代的参数集合。")
print()
print("  ⚠️ net.parameters() 是【生成器】，只能遍历一次！")
g = net.parameters()
first = list(g)
second = list(g)
print(f"     同一个生成器遍历两次：第一次 {len(first)} 个，第二次 {len(second)} 个")
print("     ⭐ 第二次是空的 —— 要用多次就先 list() 存下来")
print()


#%% 实验 8：训练 —— 注意 zero_grad 的位置
print("--- 实验 8：训练循环 ---")
print()

torch.manual_seed(42)
features, labels = synthetic_data(true_w, true_b, 1000)

linear = nn.Linear(2, 1)
net = nn.Sequential(linear)
linear.weight.data.normal_(0, 0.01)
linear.bias.data.fill_(0)

loss_fn = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
data_iter = data.DataLoader(data.TensorDataset(features, labels), 10, shuffle=True)
'''DataLoader(dataset, batch_size, shuffle)
                ↑         ↑         ↑
            数据集    每批几个    是否打乱
'''
num_epochs = 3
print("  ⭐ 教材的写法（zero_grad 在 backward 之前）：")
print()
print("     for X, y in data_iter:")
print("         l = loss(net(X), y)")
print("         trainer.zero_grad()     ← 先清零")
print("         l.backward()            ← 再算梯度")
print("         trainer.step()          ← 再更新")
print()
print(f"  {'epoch':>6}  {'loss':>16}  {'w':<24}  {'b':>9}")
print("  " + "-" * 60)

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss_fn(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.no_grad():
        full = loss_fn(net(features), labels)
    wstr = str([round(v, 4) for v in linear.weight.data.flatten().tolist()])
    print(f"  {epoch+1:>6}  {full.item():>16.10f}  {wstr:<24}  {linear.bias.data.item():>9.4f}")

print()
print("  ⭐ loss 最终落在 1e-4 量级，跟 3.2 的 5e-5 是同一水平。")
print("     两者都降到了【噪声地板】σ²/2 = 0.01²/2 = 5e-5 附近，再降不动了。")
print()
print("  ⚠️ 那「3.3 的梯度是 2 倍」体现在哪？")
print("     体现在【前期的轨迹】上 —— 步子更大，下降更猛。")
print("     但一旦都到了噪声地板，最终 loss 就看不出差别了。")
print()
print("     ⭐ 所以「梯度差 2 倍」影响的是【收敛速度 / 稳定性】，")
print("        不是最终的解（只要 lr 没超范围）。")
print()


#%% 实验 9：三种 zero_grad 顺序 —— 哪个错？
print("--- 实验 9：zero_grad 放错位置会怎样 ---")
print()


def run_with_order(order, name):
    torch.manual_seed(0)
    lin_ = nn.Linear(2, 1)
    lin_.weight.data.normal_(0, 0.01)
    lin_.bias.data.fill_(0)
    net_ = nn.Sequential(lin_)
    opt = torch.optim.SGD(net_.parameters(), lr=0.03)
    lossf = nn.MSELoss()
    init = lin_.weight.data.flatten().clone()
    dl = data.DataLoader(data.TensorDataset(Xa, Ya), 10, shuffle=False)
    for epoch in range(2):
        for Xb, Yb in dl:
            l = lossf(net_(Xb), Yb)
            for action in order:
                if action == "zero":
                    opt.zero_grad()
                elif action == "back":
                    l.backward()
                elif action == "step":
                    opt.step()
    delta = (lin_.weight.data.flatten() - init).tolist()
    print(f"     {name:<26} 两步后变化 = {[round(v,6) for v in delta]}")


print("  三种顺序，实测结果：")
print()
run_with_order(["zero", "back", "step"], "zero → back → step")
run_with_order(["back", "step", "zero"], "back → step → zero")
run_with_order(["back", "zero", "step"], "back → zero → step")
print()
print("  ⭐ 前两种都能训练，结果完全一样。")
print("     第三种（back → zero → step）参数【一点没动】！")
print()
print("  ⚠️ 原因：")
print("     optimizer.zero_grad() 会把 .grad 设成 None，")
print("     如果放在 backward 之后、step 之前，等于把刚算好的梯度扔了。")
print()
print("  📋 两种正确的顺序（记住任一种都行）：")
print("     A: zero_grad → backward → step     ← 教材用这个")
print("     B: backward → step → zero_grad     ← 我们 3.2 用这个")
print()


#%% 实验 10：验证结果
print("--- 实验 10：学出来的参数对不对 ---")
print()

w_learn = linear.weight.data.flatten()
b_learn = linear.bias.data
print(f"  真实 w = {true_w.tolist()}")
print(f"  学出 w = {[round(v,6) for v in w_learn.tolist()]}")
print(f"  误差   = {[round(v,6) for v in (true_w - w_learn).tolist()]}")
print()
print(f"  真实 b = {true_b}")
print(f"  学出 b = {b_learn.item():.6f}")
print(f"  误差   = {true_b - b_learn.item():.6f}")
print()
print("  ⚠️ 注意：linear.weight 的形状是 (1, 2)，true_w 是 (2,)")
blat = linear.weight - true_w
print(f"     直接相减会广播！{tuple(linear.weight.shape)} - {tuple(true_w.shape)} = {tuple(blat.shape)}")
print("     所以要先 flatten()，上面已经做了。")
print()


#%% 实验 11：小测验
print("=" * 64)
print(" 小测验（先在心里答，再改代码验证）")
print("=" * 64)
print("""
  Q1. nn.Linear(2, 1) 的 weight 形状是 (2,1) 还是 (1,2)？

  Q2. nn.MSELoss() 跟 3.2 的 squared_loss 差在哪？

  Q3. zero_grad 放在 backward 之后、step 之前，会发生什么？

  Q4. net.parameters() 返回的是什么？能遍历两次吗？

  Q5. 为什么初始化要用 .data 或 with torch.no_grad()？
""")
print()
print("  --- 逐条验证 ---")
print()

q_lin = nn.Linear(2, 1)
print(f"  Q1: (1, 2)。因为 nn.Linear 内部算 x @ W.T + b。实测 {tuple(q_lin.weight.shape)}")
print()

qh2 = torch.tensor([[1.0], [2.0], [3.0]])
qt2 = torch.tensor([[0.5], [2.5], [2.0]])
print("  Q2: MSELoss 没有 1/2，所以 loss 和梯度都是 squared_loss 的 2 倍。")
print(f"      实测比值: {mse(qh2,qt2).item() / squared_loss_32(qh2,qt2).mean().item():.1f}")
print()

print("  Q3: 参数永远不动（梯度被清空了）。实测上面第三种顺序：变化 = [0.0, 0.0]")
print()

gg = net.parameters()
a1 = list(gg)
a2 = list(gg)
print(f"  Q4: 生成器，只能遍历一次。同一个生成器：第一次 {len(a1)} 个，第二次 {len(a2)} 个")
print()

print("  Q5: 因为直接改 weight.normal_() 会被记进计算图（当成一次运算）。")
print("      .data 或 no_grad 跳过记录，直接改值。")
print(f"      实测两种写法等价: {torch.allclose(net_a.weight, net_b.weight)}")
print()


#%% 第 N 格：本节小结
print("=" * 64)
print(" 3.3 线性回归的简洁实现 —— 小结")
print("=" * 64)
print("""
【一个对照表（3.2 → 3.3）】

  3.2 手写                        3.3 现成
  ──────────────────────────────────────────────────────
  data_iter(bs, X, y)             DataLoader(TensorDataset(X,y), bs)
  w = normal(0,0.01,(2,1))        nn.Linear(2,1).weight  → (1,2)!
  linreg(X, w, b)                 net(X)
  squared_loss(ŷ,y)               nn.MSELoss()  ← 没有 /2
  sgd([w,b], lr, bs)              torch.optim.SGD(net.parameters(), lr)

【三个坑（考试/面试常问）】

  ⚠️ 1. nn.Linear 的 weight 是 (out, in)，跟 3.2 的 (in, out) 互为转置
        y = x @ W.T + b

  ⚠️ 2. nn.MSELoss 没有 1/2
        3.2:  L = (1/2n)·Σ(ŷ-y)²    梯度 ∝ 1
        3.3:  L = (1/n)·Σ(ŷ-y)²     梯度 ∝ 2

  ⚠️ 3. zero_grad 的位置
        ✅ zero → backward → step
        ✅ backward → step → zero
        ❌ backward → zero → step   （梯度被扔掉，参数不动）

【一个代码风格建议】

  用 nn.Sequential 时，参数访问会失去类型信息：
      net = nn.Sequential(nn.Linear(2,1))
      net[0].weight              ← 类型检查器不认识

  推荐：
      linear = nn.Linear(2, 1)
      net = nn.Sequential(linear)
      linear.weight              ← 类型明确，也不会标红

  ⭐ 功能一样，但 VS Code 不再满屏红波浪线。

【这一节真正的收获】

  你现在看到一个训练脚本，能立刻拆成五个零件：
     数据 → 模型 → 损失 → 优化器 → 循环

  而且知道每一行背后在干什么（因为 3.2 亲手写过）。

【下一节：3.4 softmax 回归】

  线性回归只能输出一个数（比如预测房价）。
  softmax 回归能输出【多个类别的概率】（比如判断是猫还是狗还是鸡）。

  结构完全一样，只是：
     · 模型多了一步 softmax（把输出变成概率）
     · 损失换成交叉熵（比较两个概率分布）

  ⭐ 所以 3.1-3.3 学扎实了，3.4 只是换零件。
""")
