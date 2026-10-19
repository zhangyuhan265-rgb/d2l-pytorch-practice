#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 64)
print(" 3.4 softmax 回归 —— 练习（精简版 6 格）")
print("=" * 64)
print("""
【教材】https://zh-v2.d2l.ai/chapter-linear-networks/softmax-regression.html

【这一节的性质】

  课本 3.4 是【纯理论】—— 整个小节 0 行代码，只有公式和概念。
  所以这份文件不做模型、不跑训练，只做一件事：
  把三个关键数字亲手算一遍。

【回归 vs 分类】

                线性回归（3.1-3.3）       softmax 回归（3.4）
  ─────────────────────────────────────────────────────
  输出           1 个数                    每类一个分数（logit）
  损失           MSE                      交叉熵
  评估           误差大小                  准确率（argmax）

【这一节只要记住四句】

  ① 线性输出不能当概率（会出负数、和不为 1）→ 所以要 softmax
  ② softmax 里有 exp，直接算会溢出 → 必须减去最大值（数学上等价）
  ③ 分类不能用 MSE：预测越离谱，梯度反而越小 → 学不动
  ④ 交叉熵对 logits 的梯度 = ŷ - y（和线性回归同形，后面代码不用改）

【怎么用】光标放在某格里按 Shift+Enter。第 0 格必须先跑。
""")

import math
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)
print("torch 版本:", torch.__version__)


#%% 实验 1：softmax 手算 —— 把分数变成概率
print("--- 实验 1：softmax 怎么算 ---")
print()

print("  判断一张图是 猫 / 狗 / 鸡。线性层吐出 3 个分数（logit）：")
o = torch.tensor([-1.2, 3.4, 0.5])
print(f"     o = {o.tolist()}")
print()
print("  ⚠️ 直接当概率的两个问题：")
print(f"     有负数：{o.min().item():.2f}")
print(f"     和 = {o.sum().item():.2f}，不是 1")
print()

print("  softmax：先 exp，再除以总和")
exps = torch.exp(o)
total = exps.sum()
print(f"     exp(o) = {[round(v,6) for v in exps.tolist()]}")
print(f"     Σ      = {total.item():.6f}")
names = ["猫", "狗", "鸡"]
for i in range(3):
    print(f"     {names[i]}：{exps[i].item():.6f} / {total.item():.6f} = {(exps[i]/total).item():.6f}")
print()
print(f"  用 torch.softmax 核对：{[round(v,6) for v in torch.softmax(o,0).tolist()]}")
print(f"  一致？ {torch.allclose(exps/total, torch.softmax(o, 0))}")
print()

print("  三个性质：")
p = torch.softmax(o, 0)
print(f"     ① 非负：{bool((p >= 0).all())}")
print(f"     ② 和为 1：{p.sum().item():.10f}")
print(f"     ③ 保序：分数序 {o.argsort().tolist()} == 概率序 {p.argsort().tolist()} → {bool((o.argsort() == p.argsort()).all())}")
print()
print("  ⭐ 第 ③ 条很重要：softmax 不改大小顺序，")
print("     所以「分类」本身不需要 softmax（argmax 就行）。")
print("     softmax 是【算损失】才需要的 —— 要把分数变成概率才能和标签比。")
print()

print("  ⚠️ 副作用：exp 是指数的，分数差一点，概率差很多")
for d in (1, 5, 10):
    pr = torch.softmax(torch.tensor([float(d), 0.0]), 0)
    print(f"     分数差 {d:>2} → 概率比 {(pr[0]/pr[1]).item():>10.2f} 倍")
print("     所以 softmax 输出往往很极端（接近 1 或接近 0）")
print()

# ——— 留给你做 ———
print("=" * 52)
print(" ✏️ 自己验：softmax([0, 0, 100]) 的 argmax 和 [0,0,100] 一样吗？")
print("    再试 [-5, 0, -5]。保序这件事有没有例外？")
print("=" * 52)
print()


#%% 实验 2：数值稳定性 —— 为什么必须减去最大值
print("--- 实验 2：exp 溢出 ---")
print()

print("  极端情形：logits = [1000, 0, 0]")
o = torch.tensor([1000.0, 0.0, 0.0])
print()
print("  ① 天真写法（先 exp 再除）：")
exps = torch.exp(o)
print(f"     exp(o)   = {exps.tolist()}")
print(f"     Σ        = {exps.sum().item()}")
print(f"     exp/Σ    = {(exps/exps.sum()).tolist()}   ← ⚠️ nan")
print()

print("  ② 减去最大值再算：")
shifted = o - o.max()
exps2 = torch.exp(shifted)
print(f"     o - max  = {shifted.tolist()}")
print(f"     exp(...) = {exps2.tolist()}")
print(f"     结果     = {(exps2/exps2.sum()).tolist()}   ← ✅ 正常")
print()

print("  ③ 数学上完全等价：")
print()
print("        exp(oᵢ)/Σexp(oⱼ) = exp(oᵢ-m)/Σexp(oⱼ-m)")
print()
print("     分子分母同乘 exp(-m)，比例不变。")
print(f"     但对普通大小的输入，两种算法结果一致？")
for _ in range(3):
    z = torch.randn(3) * 2
    a = torch.exp(z) / torch.exp(z).sum()
    print(f"        z = {[round(v,2) for v in z.tolist()]}  一致？ {torch.allclose(a, torch.softmax(z, 0))}")
print()
print("  ⭐ 小输入两者无差别；大输入只有减最大值能活。")
print(f"     torch.softmax 内部就是这么实现的：{[round(v,6) for v in torch.softmax(o,0).tolist()]}")
print()

# ——— 留给你做 ———
print("=" * 52)
print(" ✏️ 自己验：logits = [-1000, 0, 0] 天真写法会不会炸？")
print("    提示：exp(-1000) 下溢成 0，但 exp(0) 是 1，所以 exp 得 [0, 1, 1]。")
print("    分母 = 0+1+1 = 2，结果 = [0, 0.5, 0.5] —— 和 softmax 一致，居然没炸。")
print("    那为什么还要减最大值？把同样的天真写法用到 [1000, 0, 0] 上再想。")
print("=" * 52)
print()


#%% 实验 3：交叉熵 —— 一个数字衡量预测有多离谱
print("--- 实验 3：交叉熵 ---")
print()

print("  单样本公式：L = -log( ŷ_y )      ŷ_y = 模型给【正确类】的概率")
print()
print(f"  {'给正确类的概率':<18} {'-log(p)':<12} {'直觉'}")
print("  " + "-" * 58)
for pv, desc in [(0.999, "几乎确信猜对"), (0.9, "很确信"),
                 (0.5, "半信半疑"), (0.1, "基本没猜中"),
                 (0.01, "完全没猜中")]:
    print(f"  {pv:<18} {-math.log(pv):<12.4f} {desc}")
print()
print("  ⚠️ p → 0 时 -log(p) → ∞。")
print("     softmax 输出永远 > 0，所以交叉熵永远算得出来 —— 这是它俩的配合。")
print()

print("  算一个完整的：")
logits = torch.tensor([[2.0, 1.0, 0.1]])
target = 0
p = torch.softmax(logits, dim=1)
#loss = nn.CrossEntropyLoss()(net(X), y)   该行就是交叉熵的简洁实现
print(f"     logits = {logits.tolist()[0]}，正确类 = 第 {target} 类")
print(f"     概率   = {[round(v,6) for v in p[0].tolist()]}")
print(f"     L      = -log({p[0,target].item():.6f}) = {(-torch.log(p[0,target])).item():.6f}")
print()
print(f"  nn.CrossEntropyLoss 核对：{nn.CrossEntropyLoss()(logits, torch.tensor([target])).item():.6f}")
print(f"  一致？ {torch.allclose(-torch.log(p[0,target]), nn.CrossEntropyLoss()(logits, torch.tensor([target])))}")
print()
print("  ⚠️⚠️ 大坑：nn.CrossEntropyLoss 内部【已经含 softmax】")
print("     所以喂给它的是 logits，不是概率！")
wrong = nn.CrossEntropyLoss()(p, torch.tensor([target]))
print(f"     如果错误地喂概率：{wrong.item():.6f}  ← 数字变了，但这【不是】交叉熵")
print()

print("  另外一句：交叉熵的下界是【熵】")
pt = torch.tensor([0.5, 0.3, 0.2])
H = -(pt * torch.log(pt)).sum()
print(f"     真实分布 p = {pt.tolist()}")
print(f"     熵 H(p)        = {H.item():.6f}   ← 交叉熵能降到的最低值")
print(f"     交叉熵 p,p     = {(-(pt*torch.log(pt)).sum()).item():.6f}")
print(f"     交叉熵 p,q偏   = {(-(pt*torch.log(torch.tensor([0.4,0.35,0.25])))).sum().item():.6f}")
print("     ⭐ q 越接近 p，交叉熵越接近熵；多出来的部分叫 KL 散度 ≥ 0。")
print()
# 熵 H(p) 是「真实分布自己描述自己」的代价，也就是 loss 的地板；
# 交叉熵 H(p,q) 是「用预测分布 q 去描述真实分布 p」的代价，也就是实际 loss，永远 ≥ 熵，多出来的那部分就是 KL。
# ——— 留给你做 ———
print("=" * 52)
print(" ✏️ 自己算：概率 [0.7, 0.2, 0.1]，正确类是索引 1。损失多少？")
print("    先手算 -log(0.2)，再写代码核对。")
print("    ⭐ 再想：如果正确类是索引 0 呢？为什么这么小？")
print("=" * 52)
print()


#%% 实验 4：为什么分类不能用 MSE
print("--- 实验 4：softmax + MSE 的致命问题 ---")
print()

print("  构造一个「错得很离谱」的预测：logits = [10, 0, 0]，正确类是第 1 类")
print()

y_oh = torch.tensor([[0.0, 1.0, 0.0]])

def build():
    return torch.tensor([[10.0, 0.0, 0.0]], requires_grad=True)

# 交叉熵那一路
oA = build()
pA = torch.softmax(oA, dim=1)
lossA = -torch.log(pA[0, 1])
lossA.backward()
assert oA.grad is not None
gA = oA.grad.clone()

# MSE 那一路
oB = build()
pB = torch.softmax(oB, dim=1)
lossB = ((pB - y_oh) ** 2).mean()
lossB.backward()
assert oB.grad is not None
gB = oB.grad.clone()

print(f"  模型给正确类的概率只有 {pA[0,1].item():.8f} —— 几乎没猜中")
print()
print(f"  {'方案':<22} {'loss':<12} {'梯度范数':<16} {'对 logits 的梯度'}")
print("  " + "-" * 84)
print(f"  {'softmax + 交叉熵':<22} {lossA.item():<12.4f} {gA.norm().item():<16.6f} {[round(v,6) for v in gA[0].tolist()]}")
print(f"  {'softmax + MSE':<22} {lossB.item():<12.6f} {gB.norm().item():<16.8f} {[round(v,8) for v in gB[0].tolist()]}")
print()
print(f"  ⚠️ 梯度范数差 {gA.norm().item()/gB.norm().item():.0f} 倍")
print()
print("  原因：MSE 那一路梯度里多乘了一个 ŷ(1-ŷ)：")
print(f"     正确类 ŷ = {pB[0,1].item():.8f}")
print(f"     ŷ(1-ŷ)   = {(pB[0,1]*(1-pB[0,1])).item():.10f}   ← 几乎为 0")
print()
print("  ⭐ softmax 在两端【饱和】：输出接近 0 或 1 时导数接近 0。")
print("     预测越离谱 → 越饱和 → 梯度越小 → 越学不动。这叫梯度消失。")
print()

print("  而交叉熵刚好把这个 ŷ(1-ŷ) 抵消掉，梯度化简成 ŷ - y：")
print()
print(f"  {'logits':<26} {'autograd 梯度':<32} {'ŷ - y':<32} {'一致'}")
print("  " + "-" * 96)
for _ in range(4):
    z = torch.randn(1, 3) * 2
    yl = int(torch.randint(0, 3, (1,)).item())
    yy = torch.zeros(1, 3); yy[0, yl] = 1.0
    oz = z.clone().requires_grad_(True)
    lp = -torch.log(torch.softmax(oz, dim=1)[0, yl])
    lp.backward()
    assert oz.grad is not None
    gf = (torch.softmax(z, dim=1) - yy).detach()
    print(f"  {[round(v,2) for v in z[0].tolist()]!s:<26} "
          f"{[round(v,6) for v in oz.grad[0].tolist()]!s:<32} "
          f"{[round(v,6) for v in gf[0].tolist()]!s:<32} "
          f"{torch.allclose(oz.grad, gf, atol=1e-6)}")
print()
print("  ⭐ 全部一致。所以 softmax 回归的「误差」也是 ŷ - y ——")
print("     跟线性回归同形，3.2 的 sgd 循环、3.3 的 trainer.step() 一个字都不用改。")
print()

# ——— 留给你做 ———
print("=" * 52)
print(" ✏️ 自己推（这一步最值钱）：")
print("    L = -log(ŷ_y)，ŷ_y = exp(o_y)/Σexp(oⱼ)")
print("    对 o_j 求导，分 j == y 和 j != y 两种情况，")
print("    证出 ∂L/∂o_j = ŷ_j - y_j。")
print()
print("    ⭐ 推通了，3.6 从零实现交叉熵你会写得很轻松。")
print("=" * 52)
print()


#%% 第 N 格：本节小结
print("=" * 64)
print(" 3.4 softmax 回归 —— 小结")
print("=" * 64)
print("""
【一条主线】

   分数 logits        概率 softmax          损失 交叉熵
    [2.0,1.0,0.1]  →  [0.659,0.242,0.099] → -log(0.659) = 0.417
          ↑                                        ↑
      线性层算出来                           只看正确类那一项

【四个结论】

  ① softmax(o) 非负、和为 1、保序 → argmax 不需要 softmax
  ② softmax(z) == softmax(z - max(z))        ← 数值稳定的依据
  ③ ∂(交叉熵)/∂logits = ŷ - y                 ← 和线性回归同形
  ④ 交叉熵 ≥ 熵，取等号当且仅当预测分布 == 真实分布

【三个坑】

  ⚠️⚠️⚠️ nn.CrossEntropyLoss 内部已含 softmax，喂 logits 不是概率
  ⚠️⚠️   softmax 必须减最大值，否则 exp 溢出成 nan
  ⚠️     分类不用 MSE：softmax 饱和 → 梯度消失

【和前面对照】

            3.1-3.3（回归）        3.4（分类）
  模型      线性层                 线性层 + softmax
  输出      1 个数                 K 个概率
  损失      MSE                    交叉熵
  评估      误差                   准确率（argmax）

  ⭐ 损失对输出的梯度都是 ŷ-y 的形状 —— 所以训练框架一行都不用改。

【课本 3.4 的三道练习（数学证明，可以跳过，知道存在即可）】

  1. 交叉熵的二阶导数；softmax 分布的方差与它的关系
  2. 三类等概率 (1/3,1/3,1/3) 时怎么设计二进制编码
  3. RealSoftMax(a,b) = log(exp(a)+exp(b)) 为什么叫「误称」——
     证明它 > max(a,b)，且 λ→∞ 时 λ⁻¹RealSoftMax(λa,λb) → max(a,b)

  ⭐ 第 3 题回答了「softmax 这名字其实是错的」，属趣味题，
     不影响后面写代码。真要读，重点看第 1 题（跟 Hessian、二阶优化有关）。

【下一节：3.5 图像分类数据集】

  3.4 没有任何真实数据。3.5 会下 Fashion-MNIST：
     · 10 类衣物，每张 28×28 灰度图 = 784 个数
     · 6 万训练 + 1 万测试
  3.5 自己不训练模型，只做三件事：下载、看长什么样、包成 DataLoader。

【去 3.5 之前，这三件事你都会了吗】

  ✓ 写出 softmax 公式，说明为什么要减最大值
  ✓ 分清 nn.CrossEntropyLoss 该喂 logits 还是概率
  ✓ 解释「分类为什么不用 MSE」

  三条都答得上，3.6 会非常顺。
""")
