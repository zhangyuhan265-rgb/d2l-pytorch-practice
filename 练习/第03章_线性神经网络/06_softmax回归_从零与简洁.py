#%% 第 0 格：说明 + 导入 + 数据（先跑这格）
print("=" * 68)
print(" 3.6 / 3.7 softmax 回归：从零实现 vs 简洁实现 —— 练习")
print("=" * 68)
print("""
【这一份是什么】

  3.6 和 3.7 讲的是**同一件事**：用 Fashion-MNIST 做 10 类分类。
  3.6 全部手写，3.7 全换成 nn. 现成的。

  所以这份练习不重复讲两遍，而是：
     你先手写 4 个零件  →  再和 nn. 版并排对照

【⚠️ 这是练习文件，不是教学文件】

  下面 4 个格里有 TODO，**填之前会抛 NotImplementedError，这是正常的**。
  一格一格来：填完 → Shift+Enter → 验收代码告诉你过没过。

【你要手写的 4 样东西】

  ┌────┬──────────────────┬────────────────────────────────────┐
  │ 任务│ 函数              │ 关键点                              │
  ├────┼──────────────────┼────────────────────────────────────┤
  │ 1  │ 初始化 W, b       │ W 形状 (784, 10)，b 形状 (10,)      │
  │ 2  │ softmax(X)        │ 必须减去最大值                      │
  │ 3  │ net(X)            │ 先 reshape 成 (n, 784)             │
  │ 4  │ cross_entropy     │ 高级索引取正确类概率，返回向量        │
  │ 5  │ accuracy + 训练    │ argmax → 比较 → 数个数              │
  └────┴──────────────────┴────────────────────────────────────┘

  ⭐ 忘了公式就翻 `04_softmax回归.py`（那里有手算和验证）。

【怎么用】光标放在某格里按 Shift+Enter。第 0 格必须先跑。
""")

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms

torch.manual_seed(42)

root = os.path.expanduser("~/projects/06_深度学习/data")


def load_data_fashion_mnist(batch_size):
    """下载 + ToTensor + 打包（05 里写过，这里直接用）"""
    trans = transforms.ToTensor()
    mnist_tr = datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
    mnist_te = datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_tr, batch_size, shuffle=True),
            data.DataLoader(mnist_te, batch_size, shuffle=False))


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

print(f"  train_iter: {len(train_iter)} 批，每批 {batch_size} 张")
print(f"  test_iter : {len(test_iter)} 批")

X_demo, y_demo = next(iter(train_iter))
print(f"  第一批 X{X_demo.shape}  y{y_demo.shape}")
print()
print("  数据就绪，开始任务 1")


#%% 任务 1：初始化参数
print("--- 任务 1：初始化 W 和 b ---")
print()
print("  要求：")
print("     num_inputs  = 784      （28×28，一张图拉平后）")
print("     num_outputs = 10       （10 个类别）")
print("     W 形状 (784, 10)，用 normal(0, 0.01) 填充")
print("     b 形状 (10,)，全 0")
print("     两者都要 requires_grad=True")
print()
print("  ⚠️ 这里的 W 是 (in, out)，和 nn.Linear 的 (out, in) 相反")
print()

num_inputs = 784
num_outputs = 10

# ────────────────────────── 你写 ──────────────────────────
# 提示 1：torch.normal(mean, std, size, requires_grad=True)
# 提示 2：torch.zeros(size, requires_grad=True)
W = None    # ← 换成你的代码
b = None    # ← 换成你的代码
# ──────────────────────────────────────────────────────────

if W is None or b is None:
    raise NotImplementedError("任务 1 还没写：把 W 和 b 填上")

print("  验收：")
print(f"     W.shape = {tuple(W.shape)}      期望 (784, 10)     {'✅' if tuple(W.shape) == (784, 10) else '❌'}")
print(f"     b.shape = {tuple(b.shape)}         期望 (10,)         {'✅' if tuple(b.shape) == (10,) else '❌'}")
print(f"     W.requires_grad = {W.requires_grad}   {'✅' if W.requires_grad else '❌'}")
print(f"     b.requires_grad = {b.requires_grad}   {'✅' if b.requires_grad else '❌'}")
print(f"     W.std() = {W.std().item():.6f}      期望约 0.01        {'✅' if abs(W.std().item() - 0.01) < 0.002 else '❌'}")
print(f"     b 全 0？ {bool((b == 0).all())}          {'✅' if bool((b == 0).all()) else '❌'}")
print()
print("  ⭐ 全部 ✅ 才往下走")


#%% 任务 2：softmax + 模型
print("--- 任务 2：softmax 和 net ---")
print()
print("  2a) 手写 softmax(X)：对每一行归一化")
print("      ⚠️ 必须减去最大值，否则 logits 一大就 nan")
print("      提示：X.max(dim=1, keepdim=True).values")
print()
print("  2b) net(X)：X 是 (n,1,28,28)，先拉平成 (n,784)，再算 X@W+b，最后 softmax")
print("      提示：X.reshape(-1, num_inputs)")
print()

# ────────────────────────── 你写 ──────────────────────────
def softmax(X):
    """沿第 1 维归一化，返回概率，形状不变"""

    raise NotImplementedError("任务 2a：写 softmax")


def net(X):
    """(n,1,28,28) → (n,10) 的概率"""

    raise NotImplementedError("任务 2b：写 net")
# ──────────────────────────────────────────────────────────

print("  验收：")
# 2a 用一组正常输入
z = torch.tensor([[-1.2, 3.4, 0.5]])
try:
    got = softmax(z)
    good = torch.allclose(got, torch.softmax(z, dim=1))
    print(f"     softmax([[-1.2,3.4,0.5]]) = {[round(v,6) for v in got[0].tolist()]}")
    print(f"     和 torch.softmax 一致？ {good}   {'✅' if good else '❌'}")
    print(f"     行和 = {got.sum().item():.6f}   {'✅' if abs(got.sum().item()-1) < 1e-6 else '❌'}")
except NotImplementedError as e:
    print(f"     ❌ {e}")

# 2a 关键：极端输入不能出 nan
try:
    big = torch.tensor([[1000.0, 0.0, 0.0]])
    got_big = softmax(big)
    ok_big = torch.isfinite(got_big).all().item()
    print(f"     softmax([[1000,0,0]]) = {[round(v,6) for v in got_big[0].tolist()]}")
    print(f"     全是有限值？ {ok_big}   {'✅ 减最大值了' if ok_big else '❌ 忘了减最大值'}")
except NotImplementedError:
    pass

# 2b
try:
    out = net(X_demo)
    shape_ok = tuple(out.shape) == (batch_size, 10)
    sum_ok = torch.allclose(out.sum(dim=1), torch.ones(batch_size))
    print(f"     net(X).shape = {tuple(out.shape)}   期望 {(batch_size, 10)}   {'✅' if shape_ok else '❌'}")
    print(f"     每行和为 1？ {sum_ok}   {'✅' if sum_ok else '❌'}")
except NotImplementedError as e:
    print(f"     ❌ {e}")
print()


#%% 任务 3：交叉熵
print("--- 任务 3：cross_entropy(y_hat, y) ---")
print()
print("  要求：返回【向量】shape (n,)，每个样本一个 loss")
print("        公式 L = -log( ŷ_y )")
print()
print("  提示：y_hat[range(len(y_hat)), y] 一次取出所有「正确类概率」")
print("        忘了这个索引怎么工作，翻速查表搜「高级索引」")
print()

# ────────────────────────── 你写 ──────────────────────────
def cross_entropy(y_hat, y):
    """返回 (n,) 的向量"""

    raise NotImplementedError("任务 3：写 cross_entropy")
# ──────────────────────────────────────────────────────────

print("  验收：")
demo_y_hat = torch.tensor([[0.1, 0.3, 0.6],
                           [0.3, 0.2, 0.5]])
demo_y = torch.tensor([0, 2])
try:
    got = cross_entropy(demo_y_hat, demo_y)
    print(f"     cross_entropy([[.1,.3,.6],[.3,.2,.5]], [0,2])")
    print(f"       = {[round(v,6) for v in got.tolist()]}")
    print(f"       期望 [2.302585, 0.693147]")
    shape_ok = tuple(got.shape) == (2,)
    print(f"     形状 {tuple(got.shape)}   {'✅（是向量，不是标量）' if shape_ok else '❌ 必须是 (n,) 向量'}")
    print(f"     数值对吗？ {torch.allclose(got, torch.tensor([2.302585, 0.693147]), atol=1e-5)}")

    # 用官方实现交叉验证（把概率反推成 logits）
    ref = F.cross_entropy(torch.log(demo_y_hat), demo_y, reduction='none')
    print(f"     和 F.cross_entropy 一致？ {torch.allclose(got, ref, atol=1e-6)}")
except NotImplementedError as e:
    print(f"     ❌ {e}")
print()


#%% 任务 4：准确率 + 训练
print("--- 任务 4：accuracy + 训练循环 ---")
print()
print("  4a) accuracy(y_hat, y)：返回【猜对的个数】（不是比例）")
print("      提示：y_hat.argmax(dim=1) == y  →  .sum()")
print()
print("  4b) 训练循环：")
print("      每批：net(X) → loss → mean() → backward → step → zero_grad")
print("      ⚠️ 用你手写的 W、b 做参数，updater 用 SGD([W, b], lr)")
print("      ⚠️ 记住 02 里手写 sgd 要传 batch_size，而 optim.SGD 不用")
print()

# ────────────────────────── 你写 4a ──────────────────────────
def accuracy(y_hat, y):
    """返回猜对的个数（int/float）"""

    raise NotImplementedError("任务 4a：写 accuracy")
# ──────────────────────────────────────────────────────────

# 4a 的验收数据（本格自己定义，不依赖第 3 格）
demo_y_hat = torch.tensor([[0.1, 0.3, 0.6],
                           [0.3, 0.2, 0.5]])
demo_y = torch.tensor([0, 2])

print("  4a 验收：")
try:
    got_acc = accuracy(demo_y_hat, demo_y)
    print(f"     accuracy([[.1,.3,.6],[.3,.2,.5]], [0,2]) = {got_acc}   期望 1")
    print(f"     （模型猜 [2,2]，真实 [0,2] → 只对 1 个）")
except NotImplementedError as e:
    print(f"     ❌ {e}")
print()

# ────────────────────────── 你写 4b ──────────────────────────
def train(num_epochs=5, lr=0.1):
    """训练并打印每个 epoch 的 loss / train_acc / test_acc

    提示（把下面这几步写出来，然后删掉最后的 raise）：
      步骤 1：trainer = torch.optim.SGD([W, b], lr=lr)
      步骤 2：for epoch in range(num_epochs):
                  total_loss, correct, count = 0.0, 0, 0
                  for X, y in train_iter:
                      y_hat = net(X)
                      l = cross_entropy(y_hat, y)
                      trainer.zero_grad()
                      l.mean().backward()          ← 用 mean 求梯度
                      trainer.step()
                      total_loss += l.sum().item() ← 记账用 sum
                      correct += accuracy(y_hat, y)
                      count += len(y)
      步骤 3：用 test_iter 算测试准确率（记得 torch.no_grad()）
      步骤 4：print 这一轮的 loss / train_acc / test_acc
    """

    raise NotImplementedError("任务 4b：写训练循环（把上面 4 步实现出来，再删掉这行）")
# ──────────────────────────────────────────────────────────

# 训练完再验（demo 数据在本格自己定义，不依赖其他格）
print("  4b 验收：")
try:
    train(num_epochs=5, lr=0.1)
    print()
    print("     第 1 轮 loss 应该在 0.8 左右（刚离开 ln(10)≈2.30）")
    print("     第 5 轮 loss 应该降到 0.6 以下")
    print("     train_acc / test_acc 应该 > 0.80")
except NotImplementedError as e:
    print(f"     ❌ {e}")
print()


#%% 对照格：nn. 简洁版（这格是给你的，不用填）
print("=" * 68)
print(" 对照：3.7 简洁版做了哪些事")
print("=" * 68)
print()
print("  同样的任务，nn. 版只需要 4 行：")
print()
print("     net_nn = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))")
print("     loss_nn = nn.CrossEntropyLoss()")
print("     trainer_nn = torch.optim.SGD(net_nn.parameters(), lr=0.1)")
print("     # 训练循环和第 4 格完全一样")
print()
print("  对照表（这是这一章真正的收获）：")
print()
print("     你手写的                   nn. 帮你做的")
print("     ─────────────────────     ──────────────────────────────")
print("     W, b 两个张量              nn.Linear(784, 10) 的 .weight/.bias")
print("     X.reshape(-1, 784)         nn.Flatten()")
print("     softmax(X@W+b)             不写！CrossEntropyLoss 内部做")
print("     cross_entropy(y_hat, y)    nn.CrossEntropyLoss()(logits, y)")
print("     l.mean() 才是标量          默认 reduction='mean'")
print()
print("  ⚠️ 最大的差别：nn. 版【不要】softmax")
print("     因为 CrossEntropyLoss 内部 = log_softmax + nll")
print("     你手写版是 softmax 完再 -log，殊途同归")
print()
print("  下面直接跑 nn. 版验证（这格不需要你填）：")
print()

net_nn = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
loss_nn = nn.CrossEntropyLoss()
trainer_nn = torch.optim.SGD(net_nn.parameters(), lr=0.1)

for epoch in range(3):
    net_nn.train()
    total_loss, correct, count = 0.0, 0, 0
    for X, y in train_iter:
        logits = net_nn(X)
        l = loss_nn(logits, y)
        trainer_nn.zero_grad()
        l.backward()
        trainer_nn.step()
        total_loss += l.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += len(y)

    net_nn.eval()
    te_correct, te_count = 0, 0
    with torch.no_grad():
        for X, y in test_iter:
            te_correct += (net_nn(X).argmax(dim=1) == y).sum().item()
            te_count += len(y)

    print(f"     epoch {epoch+1}: loss={total_loss/count:.4f}  "
          f"train_acc={correct/count:.4f}  test_acc={te_correct/te_count:.4f}")
print()
print("  ⭐ 你应该在手写版里看到几乎一样的曲线。")
print("     如果差很多，检查：初始化、lr、softmax 有没有减最大值。")
print()
print("=" * 68)
print(" 小结")
print("=" * 68)
print("""
  手写 6 个零件 → 换成 nn. 4 行，**训练循环一个字都不用改**。

  因为梯度还是 ŷ - y（3.4 推过），参数还是那些 Parameter。

  ⭐ 这就是这一章的意义：
     理解透一层，框架只是让你少写字，不是让你不思考。
""")
