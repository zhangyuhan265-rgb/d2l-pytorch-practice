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
    assert x.grad is not None
    print(f"    第 {i+1} 次 backward 后，x.grad = {x.grad.item()}")
    #x.grad.zero_()      加入zero则清除累加

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

# ════════════════════════════════════════════════════════════════════════
#  📖 为什么 y.backward() 会报错？
#  —— 用「考试总评」的比喻讲清楚
# ════════════════════════════════════════════════════════════════════════
#
#  【场景】
#      你手上有 4 门课的成绩：
#
#          y = [0, 1, 4, 9]
#              ↑  ↑  ↑  ↑
#              y₀ y₁ y₂ y₃      （当作：数学、语文、英语、物理）
#
#  backward() 想回答的问题是：
#      「总评提高一点，每门课要考多好？」
#      （数学语言：∂总评 / ∂各科成绩）
#
#  【卡住的地方】
#      慢着 —— 「总评」到底是什么？你还没定义！
#
#        · 平均制？    总评 = (y₀+y₁+y₂+y₃) / 4
#        · 理工加权？  总评 = 2·y₀ + 2·y₁ + 1·y₂ + 1·y₃
#        · 只看数学？  总评 = y₀
#
#      三种「总评」定义 → 算出三种完全不同的答案。
#
#  PyTorch 不知道你们学校怎么定总评，
#  所以它干脆拒绝执行，甩出这句：
#
#      RuntimeError: grad can be implicitly created only for scalar outputs
#                    「我只能为【一个数】隐式创建梯度」
#
#  ⭐ 那个必须传的参数 v，就是你的【权重表】：
#
#          总评 L = v₀·y₀ + v₁·y₁ + v₂·y₂ + v₃·y₃
#
#      传 ones(4) = [1, 1, 1, 1]  →  L = 0+1+4+9 = 14     （全部一视同仁）
#      传 [1, 2, 3, 4]            →  L = 0+2+12+36 = 50   （越来越看重后面）
#
#      定了 L，backward 才知道要算 ∂L/∂x，问题才有唯一答案。
#
#  ────────────────────────────────────────────────────────────────────
#  【一张图看清流程】
#
#       x ──[平方]──► y ──[加权和 v]──► L
#       ↑             ↑               ↑
#       │             │               └─ 必须是【一个数】，才能当优化目标
#       │             └─ 4 个读数，直接当目标会"分裂"
#       └─ 我们真正想要的是 ∂L/∂x（每个 x 该改多少）
#
#      报错的意思是：你还没走到 L 那一步，就想往回倒推，路断在半空。
#
#  ────────────────────────────────────────────────────────────────────
#  【三种解法，任选一】
#
#       ① y.sum().backward()                  ← 最常用，等于 v = [1,1,1,1]
#       ② y.backward(torch.ones_like(y))      ← 等价，但更显式
#       ③ y.backward(v)                       ← 想自定义权重时用
#
#  ────────────────────────────────────────────────────────────────────
#  【一句话记住】
#
#      y 是向量  =  有 4 个「候选目标」
#      你得先【压成 1 个数】= 定一个总评公式
#      压的方式就是 v（权重表）
#      压完才能「回溯」到 x，算出梯度
#
# ════════════════════════════════════════════════════════════════════════
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
print(f"    y.requires_grad          = {y.requires_grad}")  #y明明没有声明---→requires_grad可以向下传递
print(f"    y.detach().requires_grad = {y.detach().requires_grad}")
#print(y.requires_grad)
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
# ────────────────────────────────────────────────────────────────────
#  ❓ numpy 是干啥的？是不是相当于 delete？
#
#  ❌ 不是！完全无关。
#
#  📖 numpy = Numerical Python，Python 的【数值计算基础库】
#     版本：你环境里是 2.5.2
#     pandas / matplotlib / scipy / sklearn 全都建立在它上面
#     （你在 2.2 数据预处理用的 pandas、2.4 用的 matplotlib，
#       底层数据本质都是 numpy 数组）
#
#  📖 tensor.numpy() = 【格式转换】，把 PyTorch 张量 → numpy 数组
#
#     最好的比喻：把 .docx「另存为」.pdf
#        · 原文件还在
#        · 内容一样
#        · 只是换了个格式，能用别的软件打开
#        · 什么都没删
#
#     实测：
#        t = tensor([1., 2., 3.])   →   t 是 Tensor 类型
#        n = t.numpy()              →   n 是 ndarray 类型
#        t 本身呢？                  →   还在！t.sum() 照常能算
#
#  🔄 两个方向都能转：
#        PyTorch 张量  ──[.numpy()]──────────►  numpy 数组
#        PyTorch 张量  ◄──[torch.from_numpy()]──  numpy 数组
#        走一圈能回到原点，数据不变
#
#  🤔 为什么要有两个东西？能力不同：
#
#        能力              numpy 数组    PyTorch 张量
#        ─────────────────────────────────────────────
#        多维数组运算       ✅           ✅
#        GPU 加速          ❌           ✅
#        自动求导           ❌           ✅
#        神经网络层         ❌           ✅
#        和 pandas/画图     ✅ 原生      🟡 要转换
#
#     一句话：PyTorch 张量 ≈ numpy 数组 + GPU + 自动求导
#             深度学习用张量，数据处理/画图用 numpy
#
#  ⚠️ 容易混的地方（两个东西名字像，但不是一回事）：
#
#        t.numpy()          ← 【方法】把 t 转成 numpy 数组
#        np.delete(a, 0)    ← 【函数】删除数组里某个元素
#
#     而且 numpy 的删除【也不改原数组】：
#        a = np.array([10,20,30,40])
#        np.delete(a, 1)  →  [10, 30, 40]    （新数组）
#        a                →  [10, 20, 30, 40] （原数组没变）
#
#  💡 看这个规律，全是「变成什么」，不是「删什么」：
#        t.numpy()     →  变成 numpy 数组
#        t.cpu()       →  搬到 CPU
#        t.float()     →  变成 float
#        t.detach()    →  从计算图断开
#
# ────────────────────────────────────────────────────────────────────

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
print("""            梯度信息存在哪？

   叶子 x  ────  .grad 存这里（backward 之后才有）
     │
     │  y = x * 2
     ▼
   中间 y  ────  不存（is_leaf=False）
     │
     │  y.detach()  →  新张量 yd，标记 requires_grad=False
     │                  （y 本身没变）
     │
     │  yd.numpy()  →  ndarray，身上没这些字段
     │                  （数据还共享）
     ▼
  numpy arr ────  连"梯度"这个概念都没有
""")
