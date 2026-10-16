#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 62)
print(" 2.5 自动微分 —— 练习")
print("=" * 62)
print("""
【教材】https://zh-v2.d2l.ai/chapter_preliminaries/autodiff.html

【先说清楚：这节到底在讲什么】
  2.4 你学了「导数是什么」——那是数学。
  这一节讲的是「PyTorch 怎么帮你算导数」——那是工具。

  核心机制只有一句：
      requires_grad=True  →  开始记录计算图
      backward()          →  沿着图反向走一遍，用链式法则算出梯度
      .grad               →  结果存在【被求导的那个张量】上

【这一节最容易困惑的三件事】
  ① 为什么 y.grad 是 None？（梯度不在 y 上，在 x 上）
  ② 为什么跑两次 backward 梯度变成两倍？（梯度是【累加】的）
  ③ 为什么训练循环里必须写 zero_grad()？（因为第 ② 条）

  这三条是 2.5 的全部精华。

【怎么用】光标放在某格里按 Shift+Enter。第 0 格必须先跑。
""")

import torch

print("torch 版本:", torch.__version__)


#%% 实验 1：先看一个「非标量」backward 会发生什么（教材第一个坑）
print("--- 实验 1：向量能直接 backward 吗？---")
print()

x = torch.arange(4.0, requires_grad=True)
print("  x =", x)
print("  x.requires_grad =", x.requires_grad)
print("  x.grad =", x.grad, "  ← 还没算，所以是 None")
print()

y = 2 * torch.dot(x, x)
print("  y = 2 * torch.dot(x, x) =", y)
print("  y 的维度 ndim =", y.ndim, "   ← 0 维 = 标量")
print()

y.backward()
print("  ✅ backward() 成功了")
print("  x.grad =", x.grad)
print()

# ——— 先猜再看 ———
print("=" * 50)
print(" 🤔 先猜：现在 y.grad 是什么？")
print("=" * 50)
print("     猜完再往下跑下一格！")
print()


#%% 实验 2：揭晓 —— y.grad 是 None
print("--- 实验 2：y.grad 到底是什么 ---")
print()
print("  y.grad =", y.grad)
print()
if y.grad is None:
    print("  ❗ 是 None！为什么？")
    print()
    print("  因为：y = 2*torch.dot(x,x) 这个式子本身【又做了一次乘法】")
    print("        所以 y 是一个【中间结果】，不是叶子节点。")
    print()
    print("  规则：")
    print("    · requires_grad=True 直接创建的张量  →  叶子节点 (leaf)    →  存 .grad")
    print("    · 由运算产生出来的张量              →  非叶子 (non-leaf)  →  .grad 是 None")
    print()
    print("  【记这句话就够了】")
    print("      梯度存在「被求导的那个张量」上，不在「结果」上。")
    print("      你要的是 dx，不是 dy。所以看 x.grad。")
print()

print("  验证一下谁是叶子：")
print(f"    x.is_leaf = {x.is_leaf}")
print(f"    y.is_leaf = {y.is_leaf}")
print(f"    y.grad_fn = {y.grad_fn}   ← 有 grad_fn 说明它是算出来的")
print()


#%% 实验 3：用标量函数把整条链走一遍（教材的主要例子）
print("--- 实验 3：教材的例子 y = 2 * xᵀx ---")
print()

x = torch.arange(4.0, requires_grad=True)
print("  x =", x)
print()

y = 2 * torch.dot(x, x)
print("  y = 2 * torch.dot(x, x)")
print("    = 2 * (0² + 1² + 2² + 3²)")
print("    = 2 * 14")
print("    =", y.item())
print()

y.backward()
print("  y.backward() 之后：")
print("    x.grad =", x.grad)
print()
print("  手算验证：d(2xᵀx)/dx = 4x")
print("    4 * x = 4 * [0,1,2,3] = [0,4,8,12]")
print("    对上了 ✅")
print()


#%% 实验 4：连续用 x 做运算，看梯度怎么「攒」起来
print("--- 实验 4：同一个 x 被用两次 ---")
print()

x = torch.arange(4.0, requires_grad=True)
print("  x =", x)

y = x * x            # 注意：先算 y
z = y * y            # 再用 y 算 z
print()
print("  y = x * x")
print("  z = y * y")
print()

z = z.sum()
print("  z = z.sum() =", z.item(), "  ← 变成标量了，才能 backward")
print()

z.backward()
print("  z.backward() 之后：")
print("    z.grad =", z.grad, "   ← 不是叶子，所以是 None（会看到一条 Warning）")
print("    y.grad =", y.grad, "   ← 也不是叶子")
print("    x.grad =", x.grad, "   ← ✅ 只有它有")
print()

print("  手算验证：z = (x²)² = x⁴，dz/dx = 4x³")
print("    4 * [0,1,2,3]³ = 4 * [0,1,8,27] = [0,4,32,108]")
print()


#%% 实验 5：想让中间变量也留下梯度？用 retain_grad()
print("--- 实验 5：retain_grad() 救回中间变量 ---")
print()

x = torch.arange(4.0, requires_grad=True)
y = x * x
y.retain_grad()          # 👈 就加这一句
z = (y * y).sum()

z.backward()
print("  x.grad =", x.grad)
print("  y.grad =", y.grad, "   ← 现在有值了！")
print()
print("  手算验证 y.grad：z = y²，dz/dy = 2y = 2x² = 2*[0,1,4,9] = [0,2,8,18]")
print()
print("  【什么时候用】想调试「梯度在哪一层消失了」的时候。")
print("               平时不用，纯粹为了看。")
print()


#%% 实验 6：梯度是【累加】的 —— 这节最重要的坑
print("--- 实验 6：梯度累积 ---")
print()

x = torch.ones(1, requires_grad=True)
print("  x =", x.item())
print()

for i in range(3):
    y = x * 3
    y.backward()
    print(f"    第 {i+1} 次 backward 后，x.grad = {x.grad.item()}")

print()
print("  ❗ 注意看：3 → 6 → 9，每次加 3，不是一直是 3")
print()
print("  为什么？因为 backward() 是【累加】到 .grad 上的，不是覆盖。")
print("  这是设计如此——因为计算图有分支时，梯度本来就该相加。")
print()


#%% 实验 7：所以必须手动清零 —— 用梯度下降验证
print("--- 实验 7：忘了 zero_grad() 会怎样 ---")
print()

print("  目标：让 x 下降到 2/3。真解 = 0.6667")
print("  起点 x0 = -3，学习率 lr = 0.1，跑 20 步")
print()

for label, zero in [("不写 zero_grad", False), ("写了 zero_grad", True)]:
    x = torch.tensor([-3.0], requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=0.1)

    for step in range(20):
        y = 3 * x ** 2 - 4 * x
        y.backward()
        optimizer.step()          # ① 先用梯度更新参数
        if zero:
            optimizer.zero_grad()  # ② 再清零，准备下一轮

    print(f"  {label:<16} 20 步后 x = {x.item():>12.4f}")

print()
print("  不写的那行会跑飞（梯度被重复累加，越走越远）。")
print("  写了的那行正好落到 0.6667 —— 就是 2/3。")
print()
print("  ⭐ 这就是训练循环里那句 optimizer.zero_grad() 存在的【唯一理由】。")
print()


#%% 实验 7.5：顺序错了会怎样（一个真实的坑）
print("--- 实验 7.5：step() 和 zero_grad() 谁在前？---")
print()

print("  正确顺序：  backward() -> step() -> zero_grad()")
print("  错误顺序：  backward() -> zero_grad() -> step()   ← 参数永远不动！")
print()

x = torch.tensor([-3.0], requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

y = 3 * x ** 2 - 4 * x
y.backward()
print(f"  backward 后     x.grad = {x.grad}   ← 正常，-22")
optimizer.zero_grad()
print(f"  zero_grad 后    x.grad = {x.grad}   ← ❗不是变成 0，是变成 None")
optimizer.step()
print(f"  step 后         x      = {x}   ← 没梯度可用，参数纹丝不动")
print()

print("  所以：zero_grad() 的作用是【把 .grad 整个丢掉（设为 None）】，")
print("        不是把里面的数字填成 0。效果一样，但机制不同。")
print()
print("  【记忆口诀】")
print("      先走一步（step），再擦黑板（zero_grad）。")
print("      顺序反了，等于擦完黑板才走路 —— 什么都看不见，哪也去不了。")
print()


#%% 实验 8：对非标量做 backward —— 必须给「上游梯度」
print("--- 实验 8：向量 backward 的正确姿势 ---")
print()

x = torch.arange(4.0, requires_grad=True)
y = x * x
print("  x =", x)
print("  y = x * x =", y, "   shape =", tuple(y.shape))
print()
print("  直接 y.backward() 会报错：")
print("      RuntimeError: grad can be implicitly created only for scalar outputs")
print()
print("  为什么？因为 y 是向量，链式法则需要知道「每个 y_i 对最终结果贡献多少」。")
print("  这个信息就是你要传进去的参数（叫 grad_outputs 或 v）。")
print()

y.backward(torch.ones(4))
print("  y.backward(torch.ones(4)) 之后：")
print("    x.grad =", x.grad, "   ← 等于 2x")
print()

print("  理解方式：ones(4) 相当于让 y 变成 sum(y) = x₀²+x₁²+x₂²+x₃²")
print("            那么 d(sum y)/dx = 2x ✅")
print()

# ——— 留给你做 ———
print("=" * 50)
print(" ✏️ 改成 y.backward(torch.tensor([1., 2., 3., 4.])) 会得到什么？")
print("    先猜，再跑。")
print("=" * 50)
print()
print("  提示：等价于对 sum([1*y₀, 2*y₁, 3*y₂, 4*y₃]) 求导")
print("        答案应该是 [0, 4, 12, 24] —— 自己验证一下")
print()


#%% 实验 9：切断计算图 —— detach() 和 no_grad()
print("--- 实验 9：怎么让梯度「到此为止」 ---")
print()

x = torch.tensor([1.0], requires_grad=True)
y = x * 2
print("  x =", x.item())
print("  y = x * 2")
print(f"    y.requires_grad          = {y.requires_grad}")
print(f"    y.detach().requires_grad = {y.detach().requires_grad}")
print()

print("  detach() 的意思：把这块从计算图上「撕下来」，")
print("  之后的运算跟 x 再无关系，梯度传不回去。")
print()

print("  with torch.no_grad(): 是整块关闭梯度记录 ——")
x = torch.ones(1, requires_grad=True)
with torch.no_grad():
    y = x * 2
print(f"    no_grad 里算出来的 y.requires_grad = {y.requires_grad}")
print(f"    y.grad_fn = {y.grad_fn}   ← None 说明没在记录")
print()

print("  【用在哪】")
print("    · 推理/评估阶段：torch.no_grad()  省显存、更快")
print("    · 只想取值不想影响梯度：.detach()")
print("    · 把 tensor 转成 numpy：必须 .detach().numpy()，否则报错")
print()


#%% 实验 10：小测验 —— 先猜再验证
print("=" * 62)
print(" 小测验（先在心里答，再改代码验证）")
print("=" * 62)
print("""
  Q1. x = torch.arange(4.0, requires_grad=True)
      y = x.sum()
      y.backward()
      x.grad 是几？

  Q2. x = torch.tensor(2.0, requires_grad=True)
      y = x ** 3
      y.backward()
      x.grad 是几？

  Q3. 下面这段跑完，x.grad 是几？
      x = torch.ones(1, requires_grad=True)
      (x * 5).backward()
      (x * 5).backward()

  Q4. 为什么不能写 x.numpy() 而要写 x.detach().numpy()？
""")
print()
print("  --- 逐条验证 ---")
print()

x = torch.arange(4.0, requires_grad=True)
(x.sum()).backward()
print(f"  Q1: x.grad = {x.grad}")

x = torch.tensor(2.0, requires_grad=True)
(x ** 3).backward()
print(f"  Q2: x.grad = {x.grad}   （3x² = 3*4 = 12）")

x = torch.ones(1, requires_grad=True)
(x * 5).backward()
(x * 5).backward()
print(f"  Q3: x.grad = {x.grad}   （5+5 = 10，累积！）")

x = torch.ones(1, requires_grad=True)
try:
    x.numpy()
except RuntimeError as e:
    print(f"  Q4: 直接 .numpy() 报错 → {str(e)[:70]}...")
print(f"      改成 .detach().numpy() → {x.detach().numpy()}  ✅")
print()


#%% 第 N 格：本节小结
print("=" * 62)
print(" 2.5 自动微分 —— 小结")
print("=" * 62)
print("""
【三个必背的结论】

  ① requires_grad=True  →  从这个张量开始的运算会被记录成计算图
  ② backward()          →  沿图反向走，每步乘上局部导数（链式法则）
  ③ .grad               →  存在【叶子节点】上，且是【累加】的

【两条铁律】

  · 要看导数，看被求导的那个张量（通常是权重），不看结果
  · 顺序是 backward -> step -> zero_grad，位置别放错

【和 2.4 的对应关系】

  2.4 学的是：导数的【定义】           （数学）
  2.5 学的是：导数怎么【算出来】        （工具）
  第 3 章会用：拿着梯度去【更新参数】    （应用：梯度下降）

  一条线：定义 → 计算 → 使用

【下一步】
  2.6 概率（最后一节预备知识），然后就进第 3 章线性回归了。
""")
