#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 64)
print(" 4.1 多层感知机 —— 练习")
print("=" * 64)
print("""
【教材】https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/mlp.html

【这一节其实只回答两个问题】

  ① 为什么线性模型不够用，非要往网络里塞「隐藏层 + 非线性」
  ② 激活函数是什么，三个常用的长什么样、导数是什么

  注意：教材 4.1 几乎没有代码，整节只有 5 段画图的代码。
  真正的内容是论证——所以这个练习的重点是「把论证跑成数字」，
  而不是写模型。真正的模型在 4.2 和 4.3。

【先想一个问题，别急着往下翻】
  如果我只是把两个线性层首尾接起来：
        y = W₂(W₁x + b₁) + b₂
  这个「两层网络」的表达能力，比单层强吗？

  先在心里回答，再看第 1 格。

【这一节的三个实验 + 两个任务】
  实验 1  两个线性层叠起来，能不能等价成一个线性层
  实验 2  三个激活函数的曲线（ReLU / sigmoid / tanh）
  实验 3  三个激活函数的导数（交给 autograd 算）
  任务 1  自己写 sigmoid 的导数公式，和 autograd 对答案
  任务 2  课本练习题：证明 tanh(x) + 1 = 2·sigmoid(2x)

【怎么用】光标放在某格里按 Shift+Enter。第 0 格必须先跑。
  有 TODO 的格子会抛 NotImplementedError，那是正常的，填完就不抛了。
""")

import torch
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
print("torch 版本:", torch.__version__)


#%% 第 1 格：实验 1 —— 两个线性层叠起来，还是一个线性层
print("--- 实验 1：如果中间不加非线性，两层就等于一层 ---")
print()

torch.manual_seed(0)
X = torch.randn(5, 3)                       # 5 个样本，每个 3 个特征

net2 = nn.Sequential(
    nn.Linear(3, 4),                        # 第一层：3 → 4（所谓"隐藏层"）
    nn.Linear(4, 2),                        # 第二层：4 → 2
)
y_two = net2(X)
print("两层网络输出 y_two.shape =", tuple(y_two.shape))
print()

# 把两层的权重拿出来，手工合成一个等效的线性层
W1, b1 = net2[0].weight, net2[0].bias       # (4,3) 和 (4,)
W2, b2 = net2[1].weight, net2[1].bias       # (2,4) 和 (2,)

W_eq = W2 @ W1                              # (2,4)@(4,3) → (2,3)
b_eq = W2 @ b1 + b2                         # (2,4)@(4,) + (2,) → (2,)
y_one = X @ W_eq.T + b_eq                   # 一个线性层，参数是 (W_eq, b_eq)

print("等效单层的输出 y_one.shape =", tuple(y_one.shape))
print("两者最大差值:", (y_two - y_one).abs().max().item())
print()

print("  结论：差值接近 0，说明这两个网络算的是同一个函数。")
print("  也就是说，加了隐藏层，表达能力一点没涨——")
print("  因为「仿射变换的复合还是仿射变换」。")
print()
print("  + b₁ 这一步别忘了：没有偏置时 W₂W₁ 也是一个矩阵，同样合得起来。")
print("  所以必须有东西破坏线性，这就是激活函数的全部作用。")


#%% 第 2 格：实验 2 —— 三个激活函数长什么样
print("--- 实验 2：三个激活函数 ---")
print()

x = torch.arange(-8.0, 8.0, 0.1)
relu = torch.relu(x)
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)

print("  挑几个点看看具体数值：")
print(f"    {'x':>6} {'ReLU':>10} {'sigmoid':>10} {'tanh':>10}")
for v in (-8.0, -2.0, 0.0, 2.0, 8.0):
    i = int((v + 8.0) / 0.1)
    i = min(i, len(x) - 1)
    print(f"    {x[i].item():6.1f} {relu[i].item():10.4f} {sigmoid[i].item():10.4f} {tanh[i].item():10.4f}")
print()

print("  三个函数的输出范围：")
print(f"    ReLU    [{relu.min().item():.1f}, {relu.max().item():.1f}]   —— 负的全砍成 0，正的原样通过")
print(f"    sigmoid [{sigmoid.min().item():.4f}, {sigmoid.max().item():.4f}] —— 挤压到 (0,1)")
print(f"    tanh    [{tanh.min().item():.4f}, {tanh.max().item():.4f}] —— 挤压到 (-1,1)，且关于原点对称")
print()

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, y, name in zip(axes, (relu, sigmoid, tanh), ("ReLU", "sigmoid", "tanh")):
    ax.plot(x.numpy(), y.numpy())
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("  教材是分开画三张图的，这里拼成一张方便横向比。")
print("  看形状记两件事：ReLU 是折线（分段线性），sigmoid 和 tanh 是 S 形（平滑）。")
print("  sigmoid 的中心在 0.5，tanh 的中心在 0——这是两者最大的区别。")


#%% 第 3 格：实验 3 —— 导数交给 autograd
print("--- 实验 3：三个函数的导数 ---")
print()


def grad_table(fn):
    """对 fn 在 -8..8 上逐点求导，返回 (x, 导数)"""
    xg = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    yg = fn(xg)
    yg.backward(torch.ones_like(xg))        # 对每个点求导，所以传全 1 的种子
    return xg.detach(), xg.grad.clone()


xr, g_relu = grad_table(torch.relu)
xs, g_sigmoid = grad_table(torch.sigmoid)
xt, g_tanh = grad_table(torch.tanh)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, y, name in zip(axes, (g_relu, g_sigmoid, g_tanh),
                       ("ReLU'", "sigmoid'", "tanh'")):
    ax.plot(x.numpy(), y.numpy())
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

nz = int((g_relu > 0).sum())
print(f"  ReLU 的导数在 {len(g_relu)} 个点里，有 {nz} 个是 1，其余是 0。")
print(f"  sigmoid 的导数最大值 = {g_sigmoid.max().item():.4f}")
print(f"  tanh 的导数最大值    = {g_tanh.max().item():.4f}")
print()
print("  两件事值得记：")
print("    · ReLU 的导数只有 0 和 1 两种取值——这就是它训练快的原因，")
print("      梯度要么原样传回去，要么直接掐断，不会越乘越小。")
print("    · sigmoid 的导数最大只有 0.25，tanh 的最大是 1。")
print("      层一多，0.25 连乘几次就没了——这就是梯度消失的来历。")


#%% 第 4 格：任务 1 —— 自己写 sigmoid 的导数，和 autograd 对答案
print("--- 任务 1：sigmoid 的导数 ---")
print()
print("  教材给的结论是：")
print("      d/dx sigmoid(x) = sigmoid(x) · (1 - sigmoid(x))")
print()
print("  先别信，自己验证一遍。下面已经用 autograd 算好了「标准答案」，")
print("  你要做的是用那个公式手算一遍，看两者差多少。")
print()

xg = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
yg = torch.sigmoid(xg)
yg.backward(torch.ones_like(xg))
auto_grad = xg.grad.clone()                  # autograd 给的答案
s = torch.sigmoid(xg.detach())               # s(x) 的值

# ──────────────────────────────────────────────────────────
#  留给你做：
#    用 s 算出导数的解析值，赋值给 mine_grad
#    提示：公式里的 sigmoid(x) 就是 s，不用再算一遍 torch.sigmoid
# ──────────────────────────────────────────────────────────
mine_grad = None

if mine_grad is None:
    raise NotImplementedError("任务 1 还没写：用 s*(1-s) 算出导数，赋给 mine_grad")

i0 = int(s.argmax())
err = (auto_grad - mine_grad).abs().max().item()
print("  验收：")
print(f"     autograd 在峰值处给 {auto_grad[i0].item():.6f}   （应该正好是 0.25）")
print(f"     你算的   在峰值处给 {mine_grad[i0].item():.6f}")
print(f"     最大误差 {err:.3e}   {'✅ 对上' if err < 1e-6 else '❌ 不对，再想想公式'}")


#%% 第 5 格：任务 2 —— tanh 和 sigmoid 的关系；再看 ReLU 在 0 点
print("--- 任务 2：课本练习——证明 tanh(x) + 1 = 2·sigmoid(2x) ---")
print()

x2 = torch.arange(-8.0, 8.0, 0.1)
lhs = torch.tanh(x2) + 1

# ──────────────────────────────────────────────────────────
#  留给你做：
#    算右边，赋给 rhs
#    提示：torch.sigmoid(2 * x2) 再乘 2
# ──────────────────────────────────────────────────────────
rhs = None

if rhs is None:
    raise NotImplementedError("任务 2 还没写：算出 2*sigmoid(2x)，赋给 rhs")

j0 = int(x2.abs().argmin())
err2 = (lhs - rhs).abs().max().item()
print("  验收：")
print(f"     左边 tanh(x)+1  在 x≈0 处 = {lhs[j0].item():.6f}")
print(f"     右边 2·sigmoid(2x) 在 x≈0 处 = {rhs[j0].item():.6f}")
print(f"     最大误差 {err2:.3e}   {'✅ 恒等成立' if err2 < 1e-6 else '❌ 不对'}")
print()
print("  这个结论的意思是：tanh 不过是 sigmoid 平移缩放了一下，")
print("  所以两者都会梯度消失，只是 tanh 的导数上限是 1、sigmoid 是 0.25。")
print()

print("--- 顺便看一个教材提到的边界情况：ReLU 在 0 点 ---")
z = torch.tensor([0.0], requires_grad=True)
torch.relu(z).backward()
print(f"  autograd 在 x=0 处给的 ReLU 导数是: {z.grad.item()}")
print("  教材说：0 点严格来说不可导，但它默认取左边的导数（0）。")
print("  实际训练时几乎碰不到精确的 0，所以这条不影响什么——")
print("  但如果哪天你的梯度里出现一堆 0，先想想是不是神经元全死了。")
print()

print("=" * 64)
print(" 小结")
print("=" * 64)
print("""
  1. 没有激活函数，堆多少层都等价于一层。隐藏层本身不带来任何新能力。
  2. 激活函数的作用是「破坏线性」，让网络能表达非单调、非线性的关系。
  3. 三个常用激活函数：
        ReLU     max(x,0)              导数 0 或 1，最常用
        sigmoid  1/(1+e^-x)            导数 s(1-s)，最大 0.25
        tanh     (e^x-e^-x)/(e^x+e^-x) 导数 1-tanh²，最大 1
  4. 隐藏层里基本不用 sigmoid（导数上限太小，容易梯度消失），
     它现在主要用在输出层做二分类概率。

  【还能自己往下想的（教材练习里的）】
    · 只用 ReLU 的 MLP，为什么得到的是一个「连续的分段线性函数」？
    · 如果把同一个非线性单元应用在整个小批量上（而不是每个样本各一个），
      会出什么问题？
""")
