#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 62)
print(" 2.4 微积分 —— 练习")
print("=" * 62)
print("""
【教材】https://zh-v2.d2l.ai/chapter_preliminaries/calculus.html

【先说清楚：这节的重点不是"算导数"】
  你会求导，但深度学习里【没人手算导数】。
  这节真正要建立的是三个认识：

    1. 导数 = 函数在这一点【变化有多快】（几何上就是切线斜率）
    2. 数值求导靠"取极限"，h 越小越准（但太小会出错）
    3. 多变量时每个变量各有各的导数（偏导数），
       它们拼起来就是【梯度】

  第 3 条是 2.5 自动微分和第 3 章梯度下降的基础。

【教材的 16 个格子里有一大半是画图工具】
  use_svg_display / set_figsize / set_axes / plot
  那四个是 d2l 作者写的工具函数，书里标了 #@save
  （意思是"存到 d2l 包里以后复用"）。
  不用背，知道它们在干嘛就行。

【怎么用】光标放在某格里按 Shift+Enter。第 0 格必须先跑。
""")

import torch
import math

print("torch 版本:", torch.__version__)


#%% 实验 1：导数是什么 —— 先看"平均变化率"
print("--- 实验 1：从平均变化率到瞬时变化率 ---")

# 教材用的函数：f(x) = 3x^2 - 4x
def f(x):
    return 3 * x ** 2 - 4 * x

print("  f(x) = 3x² - 4x")
print()
print("  几个点上的函数值:")
for xv in [0, 1, 2, 3]:
    print(f"    f({xv}) = {f(xv)}")
print()

print("=" * 50)
print(" 第一步：什么是「平均变化率」")
print("=" * 50)
print()
print("  从 x=1 到 x=1.1，函数值变了多少？")
x0, x1 = 1.0, 1.1
print(f"    f({x1}) = {f(x1):.4f}")   #：.4f是保留四位小数
print(f"    f({x0})  = {f(x0):.4f}")
print(f"    变化量 = {f(x1) - f(x0):.4f}")
print(f"    x 的变化量 = {x1 - x0:.4f}")
print()
print(f"    平均变化率 = {f(x1)-f(x0):.4f} / {x1-x0:.4f} = {(f(x1)-f(x0))/(x1-x0):.4f}")
print()
print("  这就是「这一段区间的平均斜率」—— 中学学的 (f(b)-f(a))/(b-a)")
print()

print("=" * 50)
print(" 第二步：让区间越来越小")
print("=" * 50)
print()
print("  教材用的写法（把区间宽度记作 h）:")
print("      (f(x+h) - f(x)) / h")
print()
print("  h 越小，这个「平均变化率」越接近 x 处【真正的切线斜率】:")
print()
print(f"  {'h':>10}  {'(f(1+h)-f(1))/h':>18}")
print(f"  {'-'*32}")
for h in [1.0, 0.1, 0.01, 0.001, 0.0001]:
    slope = (f(1 + h) - f(1)) / h
    print(f"  {h:>10.5f}  {slope:>18.6f}")
print()
print("  ^^^ 越来越靠近 2.0 —— 这就是 f 在 x=1 处的导数")
print()


#%% 实验 2：教材那个 numerical_lim
print("--- 实验 2：教材的 numerical_lim ---")

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

print("  教材代码:")
print("      def numerical_lim(f, x, h):")
print("          return (f(x + h) - f(x)) / h")
print()
print("      h = 0.1")
print("      for i in range(5):")
print("          print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')")
print("          h *= 0.1")
print()
print("  实际输出:")
h = 0.1
for i in range(5):
    print(f"      h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}")
    h *= 0.1
print()
print("  ^^^ 逐次逼近 2.0")
print()

print("=" * 50)
print(" 验证：手算导数确实是 2")
print("=" * 50)
print("""
  f(x) = 3x² - 4x
  f'(x) = 6x - 4        （求导法则：3x² 的导数是 6x，-4x 的导数是 -4）

  f'(1) = 6*1 - 4 = 2   ✓ 和数值结果一致
""")
print()

# TODO 先猜：x=2 处的导数是多少？再取消注释验证
print("x=2 的数值导数:", numerical_lim(f, 2, 1e-6))
print("手算 f'(2) = 6*2-4 =", 6*2-4)


#%% 实验 3：h 到底该取多小 —— 这里有个反直觉的坑
print("--- 实验 3：h 越小越准吗？---")

def num_deriv(f, x, h):
    return (f(x + h) - f(x)) / h

true_val = 2.0     # f'(1) 的真实值
print(f"  真实导数 f'(1) = {true_val}")
print()
print(f"  {'h':>12}  {'数值结果':>10}  {'误差':>14}")
print(f"  {'-'*50}")
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-15]:
    v = num_deriv(f, 1.0, h)
    err = abs(v - true_val)
    #flag = ""
    if err > 0.01:
        flag = "  <- 误差明显变大！"
    print(f"  {h:>12.0e}  {v:>20.10f}  {err:>14.2e}{flag}")
print()
print("=" * 50)
print(" 反直觉的结论")
print("=" * 50)
print("""
  h 从 0.1 减到 1e-6，误差越来越小 —— 符合预期。

  但 h 再减小，误差反而【变大】了！为什么？

  因为计算机用【有限位数】表示小数（float64 约 16 位有效数字）：

      f(x+h) - f(x)
      └─────┬─────┘
        两个很接近的数相减

      h 很小时，f(x+h) 和 f(x) 前面很多位都一样，
      相减之后【有效数字被抵消掉一大半】，
      剩下的部分主要是舍入误差 —— 这叫【灾难性抵消】。

      f(x+h) 的值本身只有 ~16 位精度
      除以一个极小的 h（比如 1e-15）
      相当于把误差放大了 1e15 倍

  实用结论：
      数值求导时 h 一般取 1e-5 到 1e-7 之间，
      不是越小越好。
""")
print()

# TODO 自己试试：把 f 换成 sin，看 x=0 处的导数（真实值 1.0）
# 会不会有同样的规律？
def q(x):
    return math.sin(x)
def example(q,x,h):
    return (q(x + h) - q(x)) / h
sin_ture_valu = 1.0
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-15]:
    v1 = example(q, 0, h)
    err = abs(v1 - sin_ture_valu)
    flag = ""
    if err > 0.01:
        flag = "  <- 误差明显变大！"
    print(f"  {h:>12.0e}  {v1:>20.10f}  {err:>14.2e}{flag}")
print()




#%% 实验 4：看懂教材那堆画图工具函数
print("--- 实验 4：教材的画图工具（不用背，看懂即可）---")
print("""
  教材格子 6-12 定义了四个函数，都标了 #@save：

  ┌────────────────────┬────────────────────────────────────────┐
  │ use_svg_display()  │ 让 Jupyter 用 svg 格式显示图片          │
  │                    │ （比 png 清晰，放大不失真）             │
  ├────────────────────┼────────────────────────────────────────┤
  │ set_figsize(...)   │ 设置图片大小                            │
  ├────────────────────┼────────────────────────────────────────┤
  │ set_axes(...)      │ 设置坐标轴：标签、范围、刻度类型、图例    │
  ├────────────────────┼────────────────────────────────────────┤
  │ plot(...)          │ 画图的主函数                            │
  │                    │ 能自动处理"一条线"还是"多条线"           │
  └────────────────────┴────────────────────────────────────────┘

  #@save 是什么意思：
      d2l 作者用它标记"这个函数以后会被反复用到，
      存进 d2l 包里，后面章节直接 d2l.plot(...) 调用"。

  所以你在后面章节看到 d2l.plot(...) 就是调的这个。

  对我们的意义：
      你不需要手写这些工具。自己的项目里直接用 matplotlib 就行：

          import matplotlib.pyplot as plt
          plt.plot(x, y)
          plt.xlabel('x')
          plt.show()
""")
print()

print("=" * 50)
print(" 教材最后一格画的是什么")
print("=" * 50)
print("""
  x = np.arange(0, 3, 0.1)
  plot(x, [f(x), 2 * x - 3], 'x', 'f(x)',
       legend=['f(x)', 'Tangent line (x=1)'])

  画了两条线：
    1. f(x) = 3x² - 4x            <- 抛物线
    2. 2x - 3                      <- 切线

  第二条为什么是 2x - 3？
      切点: x=1, y = f(1) = -1
      斜率: f'(1) = 2
      切线方程: y - (-1) = 2(x - 1)
                y = 2x - 2 - 1 = 2x - 3    ✓

  所以那格是在【可视化"导数 = 切线斜率"】。
  你可以自己用 matplotlib 画一遍（我们不在交互窗口里画图，
  想看就写个 .py 脚本跑，或者直接看教材的输出）。
""")


#%% 实验 5：用 PyTorch 自动算导数（提前看看 2.5 会讲什么）
print("--- 实验 5：PyTorch 怎么算导数 ---")

print("  深度学习里没人手算导数，也不用数值近似。")
print("  PyTorch 有【自动微分】—— 你只要写前向计算，它自己算导数。")
print()

x = torch.tensor(1.0, requires_grad=True)     # 告诉 PyTorch：这个变量要算梯度
y = 3 * x ** 2 - 4 * x                        # 就是 f(x) = 3x² - 4x

print("  代码：")
print("      x = torch.tensor(1.0, requires_grad=True)")
print("      y = 3 * x ** 2 - 4 * x")
print()
print(f"  x = {x.item()}")
print(f"  y = f(x) = {y.item()}")
print()

y.backward()          # 反向传播，算导数
assert x.grad is not None        # 断言梯度已算出（同时消除 Pylance 的 None 警告）
print("  y.backward()  之后:")
print(f"      x.grad = {x.grad.item()}     <- 这就是 f'(1)")
print()
print("  和手算的 f'(1) = 6*1-4 = 2 一致 ✓")
print()

print("=" * 50)
print(" 对比三种求导方式")
print("=" * 50)
print("""
  ┌──────────────┬──────────────────────┬──────────────┬─────────────┐
  │ 方式          │ 怎么做                │ 精确度        │ 用在        │
  ├──────────────┼──────────────────────┼──────────────┼─────────────┤
  │ 手工推导      │ 用求导法则算           │ 精确          │ 考试、推导   │
  │ 数值近似      │ (f(x+h)-f(x))/h      │ 近似，有误差   │ 验证、调试   │
  │ 自动微分      │ y.backward()         │ 精确          │ 深度学习     │
  └──────────────┴──────────────────────┴──────────────┴─────────────┘

  为什么深度学习用自动微分：
      · 精确（不是近似）
      · 自动（不用手推）
      · 高效（一次反向传播算出所有参数的导数）
      这是 PyTorch 存在的核心价值之一。

  ⚠️ 注意：requires_grad=True 和 .backward() 是 2.5 节的主角，
          这里只是先看一眼。搞不清细节没关系，2.5 会展开。
""")
print()

# TODO 试试别的点
x2 = torch.tensor(2.0, requires_grad=True)
y2 = 3 * x2 ** 2 - 4 * x2
y2.backward()
assert x2.grad is not None
print("f'(2) =", x2.grad.item())     # 先猜是多少？
#此处item会报错，主要是返回类型判断为了None，但是输出正常
#加入：
#assert x2.grad is not None       断言梯度已算出（同时消除 Pylance 的 None 警告）

#%% 实验 6 ⭐：偏导数 —— 多变量时"分别求导"
print("--- 实验 6：偏导数 ---")

print("  教材在「偏导数」那节只给了文字，没给代码。补上。")
print()

# 二元函数：f(x, y) = x² + 2xy + y²
def g(x, y):
    return x ** 2 + 2 * x * y + y ** 2

print("  g(x, y) = x² + 2xy + y²")
print()

print("=" * 50)
print(" 偏导数是什么：把其它变量【当常数】，只对一个变量求导")
print("=" * 50)
print("""
  对 x 求偏导（把 y 看作常数）:
      ∂g/∂x = 2x + 2y

  对 y 求偏导（把 x 看作常数）:
      ∂g/∂y = 2x + 2y

  （这个例子两个偏导数碰巧一样，换个函数就不一样了）
""")
print()

print("  在点 (1, 2) 处:")
xv, yv = 1.0, 2.0
print(f"    g(1,2) = {g(xv, yv)}")
print(f"    ∂g/∂x = 2*1 + 2*2 = {2*xv + 2*yv}")
print(f"    ∂g/∂y = 2*1 + 2*2 = {2*xv + 2*yv}")
print()

print("  用数值法验证（对 x 求偏导 = 只动 x，y 不动）:")
h = 1e-6
dx_num = (g(xv + h, yv) - g(xv, yv)) / h
dy_num = (g(xv, yv + h) - g(xv, yv)) / h
print(f"    数值 ∂g/∂x ≈ {dx_num:.6f}    （理论值 {2*xv+2*yv}）")
print(f"    数值 ∂g/∂y ≈ {dy_num:.6f}    （理论值 {2*xv+2*yv}）")
print()

print("  用 PyTorch 自动微分:")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2 + 2*x*y + y**2
z.backward()
assert x.grad is not None and y.grad is not None
print(f"    x.grad = {x.grad.item():.6f}")
print(f"    y.grad = {y.grad.item():.6f}")
print()
print("  ^^^ 三种方法结果一致 ✓")
print()

print("=" * 50)
print(" 梯度 = 所有偏导数拼成的向量")
print("=" * 50)
print("""
  在点 (1, 2) 处：
      梯度 = (∂g/∂x, ∂g/∂y) = (6, 6)

  梯度的含义：
      指向函数【上升最快】的方向。
      所以想找最小值 -> 往梯度的【反方向】走。
      这就是梯度下降。

  这也是为什么 PyTorch 里参数都有 .grad ——
  训练就是反复读 .grad，然后往反方向挪一点点。
""")
print()


#%% 实验 7：换个函数，偏导数就不一样了
print("--- 实验 7：偏导数会各不相同 ---")

def h_func(x, y):
    return x ** 3 + 2 * x * y ** 2 + y

print("  h(x, y) = x³ + 2xy² + y")
print()
print("  手工求偏导:")
print("      ∂h/∂x = 3x² + 2y²      （把 y 当常数：x³ -> 3x², 2xy² -> 2y², y -> 0）")
print("      ∂h/∂y = 4xy + 1        （把 x 当常数：x³ -> 0, 2xy² -> 4xy, y -> 1）")
print()

for pt in [(1.0, 2.0), (2.0, 1.0), (0.0, 3.0)]:
    xv, yv = pt
    dx_theory = 3 * xv**2 + 2 * yv**2
    dy_theory = 4 * xv * yv + 1

    x = torch.tensor(xv, requires_grad=True)
    y = torch.tensor(yv, requires_grad=True)
    z = x**3 + 2*x*y**2 + y
    z.backward()
    assert x.grad is not None and y.grad is not None

    print(f"  点 {pt}:")
    print(f"    手算  ∂h/∂x = {dx_theory:>8.2f}    ∂h/∂y = {dy_theory:>8.2f}")
    print(f"    torch ∂h/∂x = {x.grad.item():>8.2f}    ∂h/∂y = {y.grad.item():>8.2f}")
    ok = "✓" if abs(dx_theory - x.grad.item()) < 1e-5 and abs(dy_theory - y.grad.item()) < 1e-5 else "✗"
    print(f"    一致吗: {ok}")
    print()

# TODO 自己造一个函数，手推偏导，再用 torch 验证
# 比如 f(x,y) = x*y + x^2
def g_func(x,y):
    return x*y + x**2
for i in [(1.0, 2.0), (2.0, 1.0), (0.0, 3.0)]:
    xv1 , yv1 = i
    dx_theory1 = yv1 + 2*xv1
    dy_theory1 = xv1

    x1 = torch.tensor(xv1,requires_grad=True)
    y1 = torch. tensor(yv1,requires_grad=True)
    z = x1*y1 + x1**2
    z.backward()
    assert x1.grad is not None and y1.grad is not None
    print(f"  点 {i}:")
    print(f"    手算  ∂h/∂x = {dx_theory1:>8.2f}    ∂h/∂y = {dy_theory1:>8.2f}")
    print(f"    torch ∂h/∂x = {x1.grad.item():>8.2f}    ∂h/∂y = {y1.grad.item():>8.2f}")
    ok = "✓" if abs(dx_theory1 - x1.grad.item()) < 1e-5 and abs(dy_theory1 - y1.grad.item()) < 1e-5 else "✗"
    print(f"    一致吗: {ok}")
    print()



#%% 实验 8：梯度下降 —— 导数的实际用途
'''你在山上，蒙着眼睛，想走到山谷最低点。
你能做的只有一件事：感受脚下的坡度。

坡度告诉你「哪个方向是上坡」，
那你就往【反方向】走，就是在下坡。
这就是梯度下降'''
print("--- 实验 8：梯度下降（导数用来干什么）---")

print("  导数最大的用途：找函数的最小值。")
print()
print("  方法：从一个随机点出发，反复做")
print("      x ← x - 学习率 × f'(x)")
print()
print("  直觉：")
print("      f'(x) > 0  -> 函数在上升 -> 往【左】走（减）")
print("      f'(x) < 0  -> 函数在下降 -> 往【右】走（加）")
print("      所以减号能自动朝最小值方向走")
print()

print("=" * 50)
print(" 用 f(x) = 3x² - 4x 演示")
print("=" * 50)
print("  f'(x) = 6x - 4")
print("  最小值在 f'(x) = 0 处 -> 6x - 4 = 0 -> x = 2/3 ≈ 0.6667")
print()

x = torch.tensor(3.0, requires_grad=True)     # 从 x=3 开始（离最优很远）
lr = 0.1                                       # 学习率（后面交给优化器）或者说步长
print(f"  起点 x = {x.item():.4f}   学习率 = {lr}")
print()
print(f"  {'轮次':>4}  {'x':>10}  {'f(x)':>12}  {'f\'(x)':>10}")
print(f"  {'-'*42}")
optimizer = torch.optim.SGD([x], lr=lr)     # 优化器：负责更新参数和清空梯度

for step in range(15):
    y = 3 * x**2 - 4 * x
    y.backward()

    if step % 3 == 0 or step == 14:
        grad = x.grad                                   # 取到局部变量
        if grad is not None:
            print(f"  {step:>4}  {x.item():>10.4f}  {y.item():>12.4f}  {grad.item():>10.4f}")

    optimizer.step()          # 更新参数：x = x - lr * x.grad
    optimizer.zero_grad()     # 清空梯度，准备下一轮
print()
print(f"  收敛到 x = {x.item():.6f}")
print(f"  理论最优 x = {2/3:.6f}")
print(f"  误差 = {abs(x.item() - 2/3):.2e}")
print()
print("=" * 50)
print(" 这就是机器学习的核心循环")
print("=" * 50)
print("""
  真实训练就是同一件事，只是：

      这里的 x           ->  模型的参数（几百万个）
      这里的 f(x)        ->  损失函数（预测有多差）
      这里的 f'(x)       ->  梯度（自动微分算出来的）
      这里的 lr           ->  学习率

  循环:
      1. 前向：算预测、算损失
      2. 反向：loss.backward()，得到所有参数的梯度
      3. 更新：param -= lr * param.grad
      4. 清空梯度：param.grad.zero_()
      5. 回到 1

  第 3.2 节「线性回归的从零开始实现」写的就是这五步。
""")
print()

# TODO 先猜：学习率改成 0.01 会怎样？改成 1.0 呢？
# 自己改 lr 试试，观察收敛速度。（提示：太大会"震荡"甚至"发散"）


#%% 踩坑记录（自己填）

# 格式： 现象 -> 原因
#
# 例：
#   h 取 1e-15 时数值导数误差很大 -> 浮点数的灾难性抵消
#
# 你的记录：
#
#
#


#%% 最后一格：自检
print("=" * 62)
print(" 如果上面每一格都能跑通，2.4 就过关了")
print("=" * 62)
print()
print("对照一下，这些你能说清楚吗：")
print("  1. 导数在几何上是什么？（提示：和切线有关）")
print("  2. (f(x+h)-f(x))/h 这个式子在算什么？")
print("  3. h 是不是越小越准？为什么？")
print("  4. 教材那四个画图函数是干嘛的？#@save 什么意思？")
print("  5. 切线方程 2x-3 是怎么从 f'(1)=2 推出来的？")
print("  6. 手工推导 / 数值近似 / 自动微分，三者区别？")
print("  7. 偏导数是什么？和普通导数有什么不同？")
print("  8. 梯度是什么？它指向哪个方向？")
print("  9. 梯度下降为什么用减号？（x ← x - lr * f'(x)）")
print(" 10. 写出机器学习训练循环的五步。")
print()
print("说不清楚的那条，回去重跑对应的实验。")
print("=" * 62)
