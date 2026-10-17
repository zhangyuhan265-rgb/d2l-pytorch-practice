#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 62)
print(" 2.6 概率 —— 练习")
print("=" * 62)
print("""
【教材】https://zh-v2.d2l.ai/chapter_preliminaries/probability.html

【先说清楚：这节在干什么】
  前面 2.1-2.5 处理的都是【确定的】东西：
     张量、矩阵、导数、梯度 —— 给定输入，输出是唯一确定的。

  但机器学习面对的是【不确定】：
     · 这张图是猫还是狗？99% 像猫 —— 那 1% 是什么？
     · 用户会不会买这本书？
     · 明天北京下雨吗？

  概率就是描述「不确定程度」的正式语言。

【这节其实就三件事】
  ① 大数定律：频率 → 概率
     掷 6 次骰子，1 可能一次都不出；
     掷 6000 次，1 出现的比例一定接近 1/6。

  ② 贝叶斯定理：证据 → 更新信念
     一个 99% 准确率的检测，阳性之后真的患病概率只有 13%。
     （这个反直觉的结论是本节的高潮）

  ③ 期望和方差：用一个数概括一个分布
     期望 = 平均值（分布的中心）
     方差 = 离中心有多散

【跟深度学习的关系（为什么要学）】
  · loss 本质上是「期望」：E[loss]，我们只能用小批量估计它
  · 训练时的随机打乱、dropout、数据增强 —— 全是随机
  · 生成模型（VAE/GAN/diffusion）直接在建概率分布
  · 第 3 章线性回归的「平方损失」，就是从概率推导出来的

【怎么用】光标放在某格里按 Shift+Enter。第 0 格必须先跑。
""")

import torch
from torch.distributions import multinomial

print("torch 版本:", torch.__version__)
print()
print("torch 自带的分布工具：")
print("  from torch.distributions import multinomial   ← 本节用这个")
print("  torch.distributions 里还有 Normal / Bernoulli / Categorical 等等")


#%% 实验 1：掷一次骰子 —— multinomial 到底在干嘛
print("--- 实验 1：multinomial 的第一个参数是什么 ---")
print()
#multinomial.Multinomial(total_count, probs)
#                          ↑            ↑
#                   总共试验几次      各结果的概率
fair_probs = torch.ones(6) / 6
print("  fair_probs（公平骰子，六个面各 1/6）:")
print("   ", fair_probs)
print(f"    合计 = {fair_probs.sum().item():.4f}")
print()

print("  Multinomial(1, fair_probs).sample()   ← n=1，掷一次")
one = multinomial.Multinomial(1, fair_probs).sample()
print("    结果 =", one)
print("    看出什么？是一个 6 个数的向量，只有一个位置是 1，其余全 0")
print("    这叫 one-hot 编码 —— 第几个位置是 1，就表示掷出了几点")
print()

print("  Multinomial(10, fair_probs).sample()  ← n=10，掷十次")
ten = multinomial.Multinomial(10, fair_probs).sample()
print("    结果 =", ten)
print("    这次是【计数】：每个位置的数字表示那一点出现了几次")
print(f"    合计 = {ten.sum().item():.0f}  ← 一定是 10")
print()

print("  ⭐ 关键理解：")
print("     Multinomial 的输入不是「掷几次」，而是「总共掷几次，把结果计数」")
print("     n=1  →  一个 one-hot（这次掷了什么）")
print("     n=10 →  十个计数（十次里各个面各出现几次）")
print()


#%% 实验 2：大数定律 —— 掷得越多，频率越接近 1/6
print("--- 实验 2：大数定律 ---")
print()

print("  理论概率：每个面 1/6 = 0.166667")
print()
print(f"  {'掷的次数':>10}  {'面1的频率':>12}  {'与 1/6 的差距':>16}")
print("  " + "-" * 44)

for n in [6, 60, 600, 6000, 60000]:
    counts = multinomial.Multinomial(n, fair_probs).sample()
    freq = (counts[0] / n).item()
    print(f"  {n:>10}  {freq:>12.5f}  {abs(freq - 1/6):>16.6f}")

print()
print("  看出来了吗：掷的次数越多，频率越靠近 0.166667")
print()

print("  但注意：差距【不是单调下降】的。")
print("  多掷几次，会看到有时 60 次比 600 次还准 —— 这叫波动，正常。")
print()

# ——— 留给你做 ———
print("=" * 50)
print(" ✏️ 自己试试：把上面的 [6, 60, 600, 6000, 60000] 换成")
print("     [1000] 重复跑 5 次，看 5 次的结果各是多少。")
print("     它们一样吗？为什么不一样？")
print("=" * 50)
print()

for i in range(5):
    counts = multinomial.Multinomial(1000, fair_probs).sample()
    freq = (counts[0] / 1000).item()
    print(f"  {1000:>10}  {freq:>12.5f}  {abs(freq - 1/6):>16.6f}")

#%% 实验 3：教材的招数 —— 500 组，每组 10 次，看累积
print("--- 实验 3：cumsum 那招（教材最后一个格子）---")
print()

print("  思路很聪明：")
print("     一次掷 5000 次，只能看到【一个】最终结果，看不到过程。")
print("     换成「掷 500 组，每组 10 次」，然后【累积】加总，")
print("     就能看到「从第 1 组到第 500 组」的收敛过程。")
print()

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
print("  counts.shape =", tuple(counts.shape), " ← 500 行 × 6 列")
print("    第 i 行 = 第 i 组（10 次）的计数")
print()

cum_counts = counts.cumsum(dim=0)
print("  cum_counts = counts.cumsum(dim=0)  ← 沿行方向累积加总")
print("    cum_counts.shape =", tuple(cum_counts.shape))
print()

estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)
print("  estimates = cum_counts / 各行总和")
print("    keepdims=True 的意思是：保持二维形状，这样才能逐行广播除法")
print()

print("  看几个关键行（第 1、10、100、500 组之后）:")
print()
print(f"  {'组数':>6}  {'总掷数':>8}  {'面1估计':>10}  {'面2估计':>10}  {'误差(面1)':>12}")
print("  " + "-" * 62)
for idx in [0, 9, 49, 99, 249, 499]:
    row = estimates[idx]
    total = (idx + 1) * 10
    err = abs(row[0].item() - 1/6)
    print(f"  {idx+1:>6}  {total:>8}  {row[0].item():>10.5f}  {row[1].item():>10.5f}  {err:>12.6f}")

print()
print("  ⭐ 组数越多，各面的估计值越贴近 0.166667")
print("     这就是大数定律的「可视化」版本。")
print()

# ——— 留给你做 ———
print("=" * 50)
print(" ✏️ 教材练习1：把 n 从 10 改成 1（每组只掷 1 次），")
print("     500 组之后收敛得比现在快还是慢？自己试。")
print("     提示：multinomial.Multinomial(1, fair_probs).sample((500,))")
print("=" * 50)
print()


#%% 实验 4：期望与方差 —— 手算 vs 采样
print("--- 实验 4：期望和方差 ---")
print()

side = torch.arange(1.0, 7.0)
print("  一个公平骰子的取值：", side.tolist())
print()

E_hand = side.mean()#算术平均数
print("  【期望】手算：(1+2+3+4+5+6)/6 =", E_hand.item())
print()

E2 = (side ** 2).mean()
print("  【E[X²]】手算：(1+4+9+16+25+36)/6 =", E2.item())
var_hand = E2 - E_hand ** 2#方差
print("  【方差】手算：E[X²] - E[X]² =", f"{E2.item():.5f} - {E_hand.item()**2:.5f} = {var_hand.item():.5f}")
print("  分数形式 = 35/12 =", 35/12)
print()

print("  【采样验证】掷很多次，看平均值和方差趋不趋近理论值")
print()
print(f"  {'掷的次数':>10}  {'样本均值':>12}  {'样本方差':>12}")
print("  " + "-" * 40)
for n in [100, 1000, 10000, 100000]:
    c = multinomial.Multinomial(n, fair_probs).sample()
    # 用计数还原出每个样本，再算均值方差
    samples = torch.repeat_interleave(side, c.long())
    print(f"  {n:>10}  {samples.mean().item():>12.4f}  {samples.var(unbiased=False).item():>12.4f}")

print()
print("  理论值                     3.5000        2.9167")
print()
print("  ⭐ 样本越多越接近。10000 次之后就只差百分位了。")
print()

print("  【一个常用公式】Var[X] = E[X²] - (E[X])²")
print("     为什么有这个公式？因为方差定义里要减去均值，")
print("     展开后 E[(X-μ)²] = E[X²] - 2μE[X] + μ² = E[X²] - μ²")
print()


#%% 实验 5：独立 vs 不独立
print("--- 实验 5：两个事件独立是什么意思 ---")
print()

print("  定义：A 和 B 独立 ⇔ P(A,B) = P(A)·P(B)")
print()
print("  【例子1】掷两次骰子 —— 独立")
print("     第一次掷出 6 的概率 = 1/6")
print("     第二次掷出 6 的概率 = 1/6")
print("     两次都 6 的概率    = 1/36  （= 1/6 × 1/6）")
print()

n1 = multinomial.Multinomial(1, fair_probs).sample()
n2 = multinomial.Multinomial(1, fair_probs).sample()
print(f"  实测一次：第一次 = {n1.argmax().item()+1} 点，第二次 = {n2.argmax().item()+1} 点")
print()

print("  用大量样本验证「两次都是 6」的概率：")
trials = 200000
first = multinomial.Multinomial(1, fair_probs).sample((trials,))
second = multinomial.Multinomial(1, fair_probs).sample((trials,))
both_six = ((first.argmax(dim=1) == 5) & (second.argmax(dim=1) == 5)).float().mean()
p_first_six = (first.argmax(dim=1) == 5).float().mean()
p_second_six = (second.argmax(dim=1) == 5).float().mean()
print(f"     P(A)     = {p_first_six.item():.5f}")
print(f"     P(B)     = {p_second_six.item():.5f}")
print(f"     P(A,B)   = {both_six.item():.5f}")
print(f"     P(A)·P(B)= {(p_first_six * p_second_six).item():.5f}")
print(f"     两者接近吗？ {abs(both_six.item() - (p_first_six*p_second_six).item()) < 0.003}")
print()

print("  【例子2】不独立的情况")
print("     从 1-6 里掷一次，事件 A = 「掷出偶数」，事件 B = 「掷出 6」")
print("     P(A) = 3/6 = 0.5")
print("     P(B) = 1/6")
print("     P(A 且 B) = P(掷出 6) = 1/6   ← 不是 0.5 × 1/6 = 1/12")
print("     因为「掷出 6」已经蕴含了「掷出偶数」，B 发生则 A 必发生")
print()
print("  ⭐ 这就是「不独立」：知道 B 发生了，A 的概率就从 0.5 变成 1")
print()


#%% 实验 6：贝叶斯定理 —— 一个反直觉的例子（HIV 检测）
print("--- 实验 6：99% 准确的检测，阳性之后真患病概率只有 13% ---")
print()

print("  场景：HIV 检测")
print("     H = 1 表示真的患病，H = 0 表示健康")
print("     D = 1 表示检测阳性，D = 0 表示阴性")
print()
print("  已知条件（教材给的）:")
print("     P(H=1)      = 0.0015   ← 人群中患病率（先验）")
print("     P(D=1|H=1)  = 1.00     ← 真患者一定测出阳性（不漏检）")
print("     P(D=1|H=0)  = 0.01     ← 健康人中 1% 误报（假阳性）")
print()

P_H1 = 0.0015
P_D1_given_H1 = 1.00
P_D1_given_H0 = 0.01
P_H0 = 1 - P_H1

print("  第一步：算出「不管得没得病，测出阳性的总概率」P(D=1)")
print("     用全概率公式：P(D=1) = P(D=1|H=1)P(H=1) + P(D=1|H=0)P(H=0)")
print()
term1 = P_D1_given_H1 * P_H1
term2 = P_D1_given_H0 * P_H0
P_D1 = term1 + term2
print(f"     = {P_D1_given_H1} × {P_H1} + {P_D1_given_H0} × {P_H0:.4f}")
print(f"     = {term1:.6f} + {term2:.6f}")
print(f"     = {P_D1:.6f}")
print()

print("  第二步：贝叶斯定理")
print("     P(H=1|D=1) = P(D=1|H=1)·P(H=1) / P(D=1)")
print()
P_H1_given_D1 = term1 / P_D1
print(f"     = {term1:.6f} / {P_D1:.6f}")
print(f"     = {P_H1_given_D1:.4f}")
print(f"     = {P_H1_given_D1*100:.2f}%")
print()
print("  教材的答案：0.1306  →  对上了 ✅")
print()

print("  ⭐ 为什么这么低？因为【健康人太多了】。")
print("     100000 人里：")
print(f"        真患病  : {100000*P_H1:.0f} 人，全部测出阳性")
print(f"        健康的  : {100000*P_H0:.0f} 人，其中 {100000*P_H0*P_D1_given_H0:.0f} 人误报")
print(f"        阳性总数: {100000*P_D1:.0f} 人")
print(f"        其中真病的比例 = {100000*P_H1:.0f}/{100000*P_D1:.0f} = {P_H1_given_D1*100:.1f}%")
print()


#%% 实验 7：第二次检测 —— 为什么能大幅提高信心
print("--- 实验 7：再做一次检测 ---")
print()

print("  第二个检测（没那么准）:")
print("     P(D2=1|H=1) = 0.98")
print("     P(D2=1|H=0) = 0.03")
print()
print("  假设两次检测在【给定 H 的条件下独立】：")
print("     P(D1=1, D2=1 | H) = P(D1=1|H) · P(D2=1|H)")
print()

P_D2_given_H1 = 0.98
P_D2_given_H0 = 0.03

both_given_H1 = P_D1_given_H1 * P_D2_given_H1
both_given_H0 = P_D1_given_H0 * P_D2_given_H0
print(f"     P(D1=1,D2=1|H=1) = {P_D1_given_H1} × {P_D2_given_H1} = {both_given_H1:.4f}")
print(f"     P(D1=1,D2=1|H=0) = {P_D1_given_H0} × {P_D2_given_H0} = {both_given_H0:.6f}")
print()

P_both = both_given_H0 * P_H0 + both_given_H1 * P_H1
print(f"     P(D1=1,D2=1) = {both_given_H0:.6f}×{P_H0:.4f} + {both_given_H1:.4f}×{P_H1}")
print(f"                  = {both_given_H0*P_H0:.8f} + {both_given_H1*P_H1:.6f}")
print(f"                  = {P_both:.8f}")
print()

P_H1_given_both = both_given_H1 * P_H1 / P_both
print(f"     P(H=1|D1=1,D2=1) = {both_given_H1*P_H1:.6f} / {P_both:.8f}")
print(f"                      = {P_H1_given_both:.4f}")
print(f"                      = {P_H1_given_both*100:.2f}%")
print()
print("  教材的答案：0.8307  →  对上了 ✅")
print()
print("  ⭐ 从 13% 跳到 83%。第二次检测虽然更不准，但【两次独立的阳性证据】，")
print("     比单靠一次更可信 —— 因为「两次都误报」的概率只有 0.01×0.03 = 0.03%。")
print()


#%% 实验 8：教材练习4 —— 为什么不连续做两次同一个检测？
print("--- 实验 8：连续做两次【第一个】检测，会怎样？---")
print()

P_H1 = 0.0015
P_H0 = 1 - P_H1
P_D1_given_H1 = 1.00
P_D1_given_H0 = 0.01
P_D2_given_H1 = 0.98
P_D2_given_H0 = 0.03
P_D1 = P_D1_given_H1 * P_H1 + P_D1_given_H0 * P_H0
P_H1_given_D1 = P_D1_given_H1 * P_H1 / P_D1

print("  先【假设两次完全独立】，算一遍：")
print()

h1_twice = P_D1_given_H1 * P_D1_given_H1
h0_twice = P_D1_given_H0 * P_D1_given_H0
p_twice = h0_twice * P_H0 + h1_twice * P_H1
P_H1_twice = h1_twice * P_H1 / p_twice

print(f"     P(D1=1,D1=1|H=0) = 0.01 × 0.01 = {h0_twice:.6f}")
print(f"     P(D1=1,D1=1|H=1) = 1.00 × 1.00 = {h1_twice:.4f}")
print(f"     P(两次阳性)      = {p_twice:.8f}")
print(f"     P(H=1|两次阳性)  = {P_H1_twice:.4f} = {P_H1_twice*100:.2f}%")
print()

print("  再算两个【不同】检测（实验7 的结果）：")
both_h1 = P_D1_given_H1 * P_D2_given_H1
both_h0 = P_D1_given_H0 * P_D2_given_H0
p_both = both_h0 * P_H0 + both_h1 * P_H1
P_H1_both = both_h1 * P_H1 / p_both
print(f"     P(H=1|两次阳性)  = {P_H1_both*100:.2f}%")
print()

print("  ⚠️ 注意矛盾：")
print(f"     做两次同一个检测（独立性假设下） → {P_H1_twice*100:.2f}%")
print(f"     做两个不同检测                → {P_H1_both*100:.2f}%")
print(f"     数字上「同一个」反而更好！那教材为什么推荐换一个？")
print()

print("  ⭐ 因为那个 93.76% 建立在【独立性假设】上，而现实中它不成立：")
print()
print("     · 同一台仪器、同一批试剂、同一次操作、同一管血样")
print("     · 如果第一次是误报（试剂污染 / 样本溶血 / 操作失误），")
print("       第二次极可能【也是误报】—— 同一个原因导致")
print("     · 所以两次结果高度相关，不是两份独立证据")
print()

print("  算一下极端情况：假设两次结果完全一样（相关系数 = 1）")
print("     那「两次都阳性」就等于「一次阳性」，没有新信息：")
print(f"     P(H=1|两次阳性) = {P_H1_given_D1*100:.2f}%    ← 一点没提高！")
print()

print("  ⭐ 教材练习4 的正确答案：")
print("     贝叶斯更新要求【条件独立】：P(D1,D2|H) = P(D1|H)·P(D2|H)")
print("     重复同一个检测破坏了这个前提。")
print("     换一个检测器，失败模式不同（不同原理、不同试剂、不同操作），")
print("     才能提供真正独立的新证据 —— 这才是它有价值的原因。")
print()


#%% 实验 9：小测验 —— 先猜再验证
print("=" * 62)
print(" 小测验（先在心里答，再改代码验证）")
print("=" * 62)
print("""
  Q1. 掷骰子 600 次，1 点出现的次数大约是几？

  Q2. 一个公平骰子的期望是 3.5，那掷出来的结果可能恰好等于 3.5 吗？

  Q3. 如果 P(A,B) = P(A)·P(B)，A 和 B 独立还是不独立？

  Q4. 一个检测的准确率 99%，为什么阳性之后真患病概率可能只有 13%？
      关键在哪里？

  Q5. 方差 Var[X] = E[X²] - (E[X])²，如果 E[X] = 0，
      方差等于什么？
""")
print()
print("  --- 逐条验证 ---")
print()

print("  Q1: 期望次数 = 600 × 1/6 = 100")
counts = multinomial.Multinomial(600, fair_probs).sample()
print(f"      实测这一次：{counts[0].item()} 次（会波动，不是每次都是 100）")
print()

print("  Q2: 期望是【平均意义】，单次结果不可能是 3.5")
print(f"      骰子只能出 1,2,3,4,5,6 —— 没有 3.5 这个面")
print(f"      3.5 是「掷无穷多次的平均值」")
print()

print("  Q3: 独立。这就是独立性的定义 P(A,B) = P(A)·P(B)")
print()

print("  Q4: 关键在【基准率】(base rate)。")
print(f"      患病率只有 {P_H1*100:.2f}%，健康人占绝大多数。")
print(f"      健康人基数大，1% 的误报也能产生大量假阳性，淹没真阳性。")
print()

print("  Q5: 若 E[X] = 0，则 Var[X] = E[X²] - 0 = E[X²]")
print(f"      也就是「平方的期望」")
print()


#%% 第 N 格：本节小结
print("=" * 62)
print(" 2.6 概率 —— 小结")
print("=" * 62)
print("""
【三句话】

  ① 大数定律
        频率 → 概率，样本越多越准。
        这是「为什么能用数据估计概率」的依据。

  ② 贝叶斯定理
        先验 × 似然 / 证据 = 后验
        一个反直觉的后果：准确率高的检测，阳性也可能不可信，
        因为【基准率】太小，误报会淹没真阳性。

  ③ 期望和方差
        期望 = 分布的中心（平均）
        方差 = 分布的宽度（离散程度）
        Var[X] = E[X²] - (E[X])²

【这一节的地位】

  它是第 3 章【线性回归】的地基：
     · 平方损失其实是「高斯噪声下最大似然」的结果
     · 训练时的「平均 loss」就是期望
     · 小批量采样、随机初始化 —— 全是这一节的直接应用

【预备知识到此结束】

  第二章全貌：
     2.1 张量操作    ✅
     2.2 数据预处理  ✅
     2.3 线性代数    ✅
     2.4 微积分      ✅
     2.5 自动微分    ✅
     2.6 概率        ✅ ← 你在这里

  下一站：第 3 章 线性神经网络
     3.1 线性回归       ← 用上了 2.3 矩阵 + 2.4 梯度 + 2.5 autograd + 2.6 概率
     3.2 从零实现
     3.3 简洁实现
     ...

  ⭐ 你会看到：前面六节的全部工具，在第 3 章第一节就全用上了。
""")
