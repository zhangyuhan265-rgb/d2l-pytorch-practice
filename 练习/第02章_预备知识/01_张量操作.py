#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 60)
print(" 2.1 数据操作（张量）—— 练习")
print("=" * 60)
print("""
【怎么用这个文件】
  它用 #%% 分成多个"单元格"：
    - 光标放在某一格里，按 Shift+Enter  → 只跑这一格
    - 每格上面会出现 Run Cell 小字，点它也行
    - 结果会显示在这个交互窗口里

【怎么练】
  1. 浏览器看教材: https://zh-v2.d2l.ai/chapter_preliminaries/ndarray.html
  2. 回到这里，把代码敲一遍（别复制！）
  3. 先猜结果，再运行
  4. 猜错了 → 在最后的【踩坑记录】里写下来

【注意】格子之间有依赖，顺序不能乱。
        第 0 格必须先跑，否则后面会 NameError。
""")

import torch

print("torch 版本 :", torch.__version__)
print("CUDA 可用  :", torch.cuda.is_available())


#%% 实验 1：创建张量
print("--- 实验 1：创建张量 ---")

x = torch.arange(12) #从0开始，到11结束
print("arange(12)   :", x)
print("  .shape     :", x.shape)  #torch.Size([12])
print("  .numel()   :", x.numel()) #元素数量

# TODO 先猜形状，再取消注释逐行试
print("zeros((2,3,4)) :", torch.zeros((2, 3, 4)).shape)
print("ones((2,3,4))  :", torch.ones((2, 3, 4)).shape)
print("randn(3,4)     :", torch.randn(3, 4).shape)


#%% 实验 2：reshape —— 猜猜原张量会不会变？
print("--- 实验 2：reshape ---")

X = x.reshape(3, 4)
print("X = x.reshape(3,4)")
print("  X.shape    :", X.shape)
print("  X =\n", X)

# TODO 关键问题：reshape 之后，原来的 x 变了吗？
#      先猜，再取消下面这行
print("  x.shape 还是 :", x.shape)


#%% 实验 3：reshape 的边界 —— 这里最容易踩坑
print("--- 实验 3：reshape 边界测试 ---")
print("逐个取消注释，看哪个报错、为什么")

# TODO 逐个取消注释试
print("reshape(4,3)  ->", x.reshape(4, 3).shape)
print("reshape(2,6)  ->", x.reshape(2, 6).shape)
#print("reshape(5,3)  ->", x.reshape(5, 3).shape)      # 这个呢？
print("reshape(-1,4) ->", x.reshape(-1, 4).shape)     # -1 是什么意思？---→-1的出现可以让代码自己运算，比如这里有个-1和4，就是说"每行 4 个，几行你自己算" → 12/4 = 3
print("reshape(3,-1) ->", x.reshape(3, -1).shape)     # 再猜一个


#%% 实验 4：运算与求和 —— axis 到底按哪个方向？
print("--- 实验 4：运算与求和 ---")

A = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("A =\n", A)
print("A.sum()      :", A.sum())

# TODO 先猜：axis=0 是按行加还是按列加？结果形状是什么？
print("A.sum(axis=0) :", A.sum(dim=0)) #每列相加
print("A.sum(axis=1) :", A.sum(dim=1)) #每行相加
#.sum(axis) = k 就是把第k维消灭掉

#%% 实验 5：广播机制 —— 本节重点
print("--- 实验 5：广播机制（本节重点）---")

a = torch.arange(3).reshape((3, 1))    # 形状 (3,1)
b = torch.arange(2).reshape((1, 2))    # 形状 (1,2)

print("a =", a.tolist(), "  a.shape =", tuple(a.shape))
print("b =", b.tolist(), "  b.shape =", tuple(b.shape))

# TODO 先猜：a + b 的结果是什么形状？里面每个数是多少？
#      拿张纸画出 a 和 b 展开后的样子，再取消下面这行！
print("a + b =\n", a + b)


#%% 实验 6：节省内存 —— 对你的 2GB 显存很重要
print("--- 实验 6：内存 ---")

Y = torch.ones(3, 4)
print("Y 的 id       :", id(Y))

Y = Y + A          # 新开内存
print("Y = Y + A 后  :", id(Y))     # id 变了吗？

Z = torch.ones(3, 4)
print("Z 的 id       :", id(Z))
Z += A             # 原地修改
print("Z += A 后     :", id(Z))     # id 变了吗？

# TODO 用自己的话解释：为什么 += 不新开内存？
#      为什么这在深度学习里重要？（提示：你的显存只有 2GB）


#%% 实验 7：搬到 GPU 上看看
print("--- 实验 7：GPU ---")

if torch.cuda.is_available():
    Xg = torch.ones(1000, 1000, device="cuda")
    Yg = Xg @ Xg
    torch.cuda.synchronize()
    print("GPU 上的张量 :", Xg.device)
    print("矩阵乘法结果 :", tuple(Yg.shape))
    print("峰值显存     : %.0f MB" % (torch.cuda.max_memory_allocated() / 1024**2))
    del Xg, Yg
    torch.cuda.empty_cache()
else:
    print("CUDA 不可用")


#%% 实验 8：章节练习题（教材末尾的题，至少做前两道）
print("--- 实验 8：章节练习 ---")
print("""
练习 1: 把本节里的 X == Y 改成 X < Y 或 X > Y，看看得到什么张量。
练习 2: 用三维张量替换广播机制里的两个张量，结果和预期一样吗？

在下面写你的答案：
""")

# 在这里写练习 1 的答案：
#练习1原题：
"""
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
print(X)
print(Y)
print(X < Y)
print(X > Y)
"""
# 在这里写练习 2 的答案：
a = torch.arange(6).reshape((3 ,2, 1))    # 形状 (3 ,2, 1)
b = torch.arange(4).reshape((1 ,2, 2))    # 形状 (1 ,2, 2)

print("a =", a.tolist(), "  a.shape =", tuple(a.shape))
print("b =", b.tolist(), "  b.shape =", tuple(b.shape))

print("a + b =\n", a + b)

#%% 踩坑记录（自己填）

#
# 你的记录：
#
#广播机制
# 1) 两个 shape 右对齐写出来
# 2) 从最右边往左，逐位比较
# 3) 每一位：相等 或 其中一个是 1   → 过关
# 4) 有一关不过 → 整体报错
# 5) 全过关 → 结果 shape = 每一位取【较大】的

# 例:    a:  3  2  1
#        b:  1  1  2
#        ----------
#    取大:  3  2  2   → (3,2,2)


#%% 最后一格：自检
print("=" * 60)
print(" 如果上面每一格都能跑通，2.1 就过关了")
print("=" * 60)
