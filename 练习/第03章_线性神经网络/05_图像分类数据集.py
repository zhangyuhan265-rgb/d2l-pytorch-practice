#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 64)
print(" 3.5 图像分类数据集 —— 练习（精简版 6 格）")
print("=" * 64)
print("""
【教材】https://zh-v2.d2l.ai/chapter-linear-networks/image-classification-dataset.html

【这一节在做什么】

  3.4 讲完了 softmax 和交叉熵，但一个真实数据都没有。
  3.5 补上数据 —— 而且只做三件事：

     ① 下载 Fashion-MNIST
     ② 看它长什么样
     ③ 包成 DataLoader

  ⭐ 这一节【不训练任何模型】。训练留到 3.6。

【Fashion-MNIST 是什么】

  · 10 类衣物，每张 28×28 灰度图 → 拉平是 784 个数
  · 6 万张训练 + 1 万张测试
  · 比手写数字 MNIST 难一点（衣服比数字难分）

  类别（索引 0-9）：
     0 T恤      1 裤子      2 套头衫    3 连衣裙    4 外套
     5 凉鞋     6 衬衫      7 运动鞋    8 包        9 短靴

【数据和 3.4 的接口】

  3.4 的 softmax 回归：784 维输入 → 10 个 logits → 10 个概率
                              ↑            ↑
                        一张图的像素    正好 10 类

  ⭐ 所以 3.4 学的交叉熵，就是为这个 10 类任务准备的。

【数据在哪】

  ~/projects/06_深度学习/data/FashionMNIST/raw/   （已下载，约 82MB）
  第一次跑会检查文件，已存在就不重复下载。

【怎么用】光标放在某格里按 Shift+Enter。第 0 格必须先跑。
""")

import os
import time
from typing import Any

import torch
from torch.utils import data
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

torch.manual_seed(42)
print("torch 版本:", torch.__version__)
print("数据根目录:", os.path.expanduser("~/projects/06_深度学习/data"))


#%% 实验 1：加载数据集 —— ToTensor 到底做了什么
print("--- 实验 1：加载 Fashion-MNIST ---")
print()

root = os.path.expanduser("~/projects/06_深度学习/data")

print("  先看不加变换时，一张图是什么类型：")
raw = datasets.FashionMNIST(root=root, train=True, download=True)
img_raw, label_raw = raw[0]
print(f"     类型  = {type(img_raw).__name__}")
print(f"     尺寸  = {img_raw.size}      （宽 × 高）")
print(f"     模式  = {img_raw.mode}      （L = 灰度）")
print(f"     标签  = {label_raw}           （{type(label_raw).__name__}）")
print()

print("  加上 transforms.ToTensor() 之后：")
trans = transforms.ToTensor()
mnist_train = datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
mnist_test = datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)

img, label = mnist_train[0]
print(f"     类型  = {type(img).__name__}")
print(f"     shape = {tuple(img.shape)}      （通道 × 高 × 宽）")
print(f"     dtype = {img.dtype}")
print(f"     值域  = [{img.min().item():.1f}, {img.max().item():.1f}]")
print(f"     标签  = {label}")
print()
print("  ⭐ ToTensor 做了两件事：")
print("     ① PIL 图像 → torch.Tensor，形状变成 (C, H, W)")
print("     ② 像素值 0~255 的 uint8 → 0.0~1.0 的 float32（除以 255）")
print()

print(f"  训练集 {len(mnist_train)} 张，测试集 {len(mnist_test)} 张")
print(f"  单张图片元素数：{img.numel()} = 1 × 28 × 28")
print("  ⭐ 这个 784 就是 3.1 造数据时 features 的列数（那里是 2）。")


#%% 实验 2：10 个类别长什么样
print("--- 实验 2：类别与样本 ---")
print()

# 教材用的英文名（画图时避免中文字体缺失变成方块）
text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat",
               "sandal", "shirt", "sneaker", "bag", "ankle boot"]
cn_labels = ["T恤", "裤子", "套头衫", "连衣裙", "外套",
             "凉鞋", "衬衫", "运动鞋", "包", "短靴"]

print(f"  {'索引':<6} {'英文':<12} {'中文'}")
print("  " + "-" * 34)
for i, (en, cn) in enumerate(zip(text_labels, cn_labels)):
    print(f"  {i:<6} {en:<12} {cn}")
print()

print("  统计训练集里每一类有多少张：")


def get_fashion_mnist_labels(labels):
    """把标签索引转成英文名（教材同款函数）"""
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.2):
    """画一个 num_rows × num_cols 的图片网格"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes_obj = plt.subplots(num_rows, num_cols, figsize=figsize)
    # plt.subplots 在 1×1 时返回单个 Axes，其余返回 ndarray，
    # 类型检查器只看到 Axes | ndarray 的联合类型，所以这里标成 Any 再 flatten
    axes: Any = axes_obj
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i].squeeze().numpy(), cmap="gray")
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)
    plt.tight_layout()
    plt.show()


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
print(f"     取一个 batch：X.shape = {tuple(X.shape)}, y.shape = {tuple(y.shape)}")
print(f"     ⭐ 18 张图，每张 1×28×28")
print()
print("  画出来看（前 2 行 × 9 列）：")
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# ——— 留给你做 ———
print("=" * 52)
print(" ✏️ 自己验：每一类到底有多少张？")
print("    提示：遍历 mnist_train 的 targets 统计，")
print("          看 10 类是不是各 6000 张（会不会不均衡？）。")
print("=" * 52)
print()
# ────────────────────────────────────────────────────────────
# 【iter / next / DataLoader 三个词各管一件事】
#
#   mnist_train (60000 张)
#         ↓  DataLoader(batch_size=18)    「按什么规则切批」
#        dl      ← 可迭代对象，但自己没游标
#         ↓  iter(dl)                     「给我一个游标」
#        it      ← 迭代器：内部记着走到第几批
#         ↓  next(it)                     「取下一批」
#      (X, y)    X(18,1,28,28)  y(18,)    第 1 批
#
# ① DataLoader(...) = 造一个「分批读取器」
#      type(dl)     = DataLoader
#      有 __iter__  → True    能 for 循环
#      有 __next__  → False   ⚠️ 不能直接 next()
#      next(dl)     → TypeError: DataLoader object is not an iterator
#    ⭐ 它知道「怎么切」，但不记「走到哪了」。
#
# ② iter(dl) = 要一个「带游标的迭代器」
#      type(it)     = _SingleProcessDataLoaderIter
#      有 __next__  → True    现在能 next 了
#
# ③ next(it) = 从游标取下一批
#      next(it)     → 第 1 批   y = [9, 0, 0, 3, ...]
#      next(it)     → 第 2 批   y = [6, 4, 3, 1, ...]   游标往前走
#      取完了       → 抛 StopIteration
#
# ⭐ 所以 next(iter(DataLoader(...))) = 「只看第一批」
#    等价写法：  for X, y in dl: break
# ⚠️ iter(dl) 每次都新建游标，不是复用 —— 再 iter 又从第 1 批开始
# ────────────────────────────────────────────────────────────

#%% 实验 3：包成 DataLoader —— 复习 3.3 的 shuffle / batch_size
print("--- 实验 3：DataLoader ---")
print()

batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

print(f"  train_iter = DataLoader(mnist_train, batch_size={batch_size}, shuffle=True)")
print()
print(f"  一个 epoch 有多少个 batch：")
n_batches = len(train_iter)
print(f"     60000 / {batch_size} = {60000/batch_size:.2f} → 向上取整 = {n_batches} 批")
print(f"     ⚠️ 不能被整除，所以最后一批只有 {60000 - (n_batches-1)*batch_size} 张")
print()

first_X, first_y = next(iter(train_iter))
print(f"  取第一批：X.shape = {tuple(first_X.shape)}, y.shape = {tuple(first_y.shape)}")
print(f"  ⭐ 和 3.3 的 (10, 2) 一个道理，只是这里一张图 = 784 个数")
print()

print("  读一遍完整 epoch 要多久（含 shuffle 和 ToTensor 的开销）：")
t0 = time.perf_counter()
for _ in train_iter:
    pass
t1 = time.perf_counter()
print(f"     耗时 {t1 - t0:.2f} 秒 / {n_batches} 批")
print()
print("  ⭐ 这个时间会加到每个 epoch 的训练时间里 ——")
print("     所以 3.6 训练时要留意数据加载是不是瓶颈。")

# ——— 留给你做 ———
print("=" * 52)
print(" ✏️ 自己验：batch_size 从 256 改成 8，读一遍要多久？")
print("    变快了还是变慢了？为什么？")
print("    （提示：批数变多，但每批更小；总数据量不变）")
print("=" * 52)
print()


#%% 实验 4：封装成函数 —— 教材最后那个 load_data_fashion_mnist
print("--- 实验 4：封装 load_data_fashion_mnist ---")
print()


def load_data_fashion_mnist(batch_size, resize=None):
    """下载 Fashion-MNIST 并返回 train_iter / test_iter（教材同款）"""
    # 必须标 list[Any]：只写 [transforms.ToTensor()] 的话，
    # 类型检查器会把元素类型推断成 ToTensor，后面 insert 一个 Resize 就报错。
    trans_list: list[Any] = [transforms.ToTensor()]
    if resize:
        trans_list.insert(0, transforms.Resize(resize))
    trans_chain = transforms.Compose(trans_list)
    mnist_tr = datasets.FashionMNIST(root=root, train=True,
                                     transform=trans_chain, download=True)
    mnist_te = datasets.FashionMNIST(root=root, train=False,
                                     transform=trans_chain, download=True)
    return (data.DataLoader(mnist_tr, batch_size, shuffle=True),
            data.DataLoader(mnist_te, batch_size, shuffle=False))


print("  def load_data_fashion_mnist(batch_size, resize=None):")
print("      trans_list = [transforms.ToTensor()]")
print("      if resize:")
print("          trans_list.insert(0, transforms.Resize(resize))")
print("      trans_chain = transforms.Compose(trans_list)")
print("      ...")
print()
print("  ⭐ 三个细节：")
print("     ① train 用 shuffle=True，test 用 shuffle=False（测试不需要打乱）")
print("     ② resize 用 transforms.Compose 插到 ToTensor 前面")
print("     ③ 返回两个 DataLoader，3.6 直接拿来用")
print()

train_iter, test_iter = load_data_fashion_mnist(32)
X_tr, y_tr = next(iter(train_iter))
X_te, y_te = next(iter(test_iter))
print("  load_data_fashion_mnist(32)：")
print(f"     train 一批 = X{tuple(X_tr.shape)}  y{tuple(y_tr.shape)}")
print(f"     test  一批 = X{tuple(X_te.shape)}  y{tuple(y_te.shape)}")
print("     ⭐ y 是 (32,) 一维，不是 (32, 1) —— 交叉熵要的就是整数索引")
print()

print("  加 resize=64 试试：")
train_iter64, _ = load_data_fashion_mnist(32, resize=64)
X64, y64 = next(iter(train_iter64))
print(f"     train 一批 = X{tuple(X64.shape)}  y{tuple(y64.shape)}")
print(f"     ⭐ 28×28 → 64×64，输入维度从 784 变成 4096")
print()


#%% 第 N 格：本节小结
print("=" * 64)
print(" 3.5 图像分类数据集 —— 小结")
print("=" * 64)
print("""
【三件事，一句话各一句】

  ① 下载：datasets.FashionMNIST(root, train, transform, download)
          6 万训练 + 1 万测试，28×28 灰度
  ② 看：  ToTensor 把 PIL(0~255) 变成 tensor(0.0~1.0)，
          形状 (C, H, W) = (1, 28, 28)
  ③ 包：  DataLoader(dataset, batch_size, shuffle)
          train 打乱、test 不打乱

【关键数字】

  单张图      (1, 28, 28) → 拉平 784 个数
  10 类       0-9，对应 T恤/裤子/套头衫/连衣裙/外套/凉鞋/衬衫/运动鞋/包/短靴
  训练集      60000 张，batch=256 → 235 批（最后一批 96 张）
  测试集      10000 张

【和 3.4 的接口】

  一张图 784 维 ──→ Linear(784, 10) ──→ 10 个 logits ──→ softmax ──→ 交叉熵
        ↑                                                              ↓
    3.5 提供的数据                                              3.4 讲的损失

  ⭐ 3.4 + 3.5 拼起来，就是 3.6 要写的东西。

【三个容易踩的点】

  ⚠️ 1. ToTensor 会同时改【类型】和【数值范围】（uint8 0~255 → float32 0~1）
       忘了它，后面算出来的东西全是错的。

  ⚠️ 2. 60000 / 256 除不尽 → 最后一批只有 96 张。
       做统计时（比如算准确率）要按实际样本数算，别直接乘 batch_size。

  ⚠️ 3. test_iter 不要 shuffle。
       测试只求结果，不需要随机顺序。

【下一节：3.6 softmax 回归的从零实现】

  3.6 会把这一节的数据 + 3.4 的公式拼成一个能训练的模型：
     · 手写 softmax（要减最大值）
     · 手写交叉熵
     · 手写准确率
     · 一个完整的训练循环

  ⭐ 3.2 手写过线性回归的训练循环，3.6 是同一个骨架，
     只换掉「模型」和「损失」两个零件。

【去 3.6 之前，确认这三件事】

  ✓ 知道 mnist_train[0] 返回的 shape 是 (1, 28, 28)
  ✓ 知道 784 是怎么来的，以及 Linear(784, 10) 的 weight 形状是 (10, 784)
  ✓ 知道为什么 nn.CrossEntropyLoss 吃的是 logits 不是概率

  三条都清楚，3.6 会顺很多。
""")

# %%
