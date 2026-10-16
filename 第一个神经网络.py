"""
第一个神经网络 —— MNIST 手写数字识别

这是深度学习最经典的入门例子，跑通它你就理解了整个流程。

运行方式：
    conda activate dl
    python 第一个神经网络.py

你会看到：
    1. 自动下载 MNIST 数据集（约 11 MB）
    2. 训练 5 轮
    3. 每轮打印损失和准确率
    4. 最后显示测试集准确率（应该在 97% 以上）

理解重点：
    - 数据从哪来
    - 模型长什么样
    - 训练循环在干什么
    - 准确率怎么算出来的
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
print("实际用的解释器:", sys.executable)

# ============================================================
#  一、准备工作
# ============================================================

def check_device():
    """检查用 CPU 还是 GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("使用 CPU（GPU 不可用）")
    print()
    return device


def load_data(batch_size=64):
    """加载 MNIST 数据集

    MNIST = 70000 张 28x28 的手写数字图片，标注了 0-9
    """
    # 把图片转成张量，并归一化到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST 的均值和标准差
    ])

    # 下载训练集（第一次运行会下载）
    print("加载数据集...")
    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform)

    # DataLoader 负责"分批"和"打乱"
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    print(f"训练集: {len(train_set)} 张")
    print(f"测试集: {len(test_set)} 张")
    print(f"批次大小: {batch_size}，训练集共 {len(train_loader)} 个批次")
    print()

    return train_loader, test_loader


# ============================================================
#  二、定义模型
# ============================================================

class Net(nn.Module):
    """一个简单的卷积神经网络

    结构：卷积 → 池化 → 卷积 → 池化 → 全连接 → 输出
    """

    def __init__(self):
        super().__init__()
        # 第一个卷积层：1 个输入通道（灰度图）→ 32 个特征图
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积层：32 → 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 全连接层：64*7*7 是两次池化后的尺寸
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)        # 10 个数字类别

    def forward(self, x):
        # 第一层：卷积 → ReLU → 池化
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)                # 28x28 → 14x14

        # 第二层：卷积 → ReLU → 池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)                # 14x14 → 7x7

        # 展平成一维
        x = x.view(x.size(0), -1)

        # 全连接
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x                              # 输出 10 个分数


# ============================================================
#  三、训练和测试
# ============================================================

def train(model, device, train_loader, optimizer, epoch):
    """训练一轮"""
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()                 # 1. 清空梯度
        output = model(data)                  # 2. 前向传播
        loss = F.cross_entropy(output, target) # 3. 算损失
        loss.backward()                       # 4. 反向传播
        optimizer.step()                      # 5. 更新参数

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

        # 每 200 批打印一次进度
        if batch_idx % 200 == 0:
            print(f"  批次 {batch_idx:>4}/{len(train_loader)}  "
                  f"损失 {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    acc = 100.0 * correct / len(train_loader.dataset)
    print(f"  训练完成: 平均损失 {avg_loss:.4f}  准确率 {acc:.2f}%")
    return avg_loss, acc


def test(model, device, test_loader):
    """在测试集上评估"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():                     # 评估时不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(f"  测试集: 平均损失 {test_loss:.4f}  准确率 {acc:.2f}%")
    return test_loss, acc


# ============================================================
#  四、主流程
# ============================================================

def main():
    print("=" * 62)
    print("  MNIST 手写数字识别 —— 第一个神经网络")
    print("=" * 62)
    print()

    device = check_device()
    train_loader, test_loader = load_data(batch_size=64)

    # 建模型
    model = Net().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")
    print()

    # 优化器：负责更新参数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练 5 轮
    print("=" * 62)
    print("  开始训练")
    print("=" * 62)
    print()

    for epoch in range(1, 6):
        print(f"--- 第 {epoch} 轮 ---")
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        print()

    # 保存模型
    torch.save(model.state_dict(), "mnist_model.pth")
    print("模型已保存到 mnist_model.pth")
    print()

    print("=" * 62)
    print("  训练完成！")
    print("=" * 62)
    print()
    print("  你应该看到测试集准确率在 97% 以上")
    print("  如果达到了，说明你的深度学习环境完全正常")

    if torch.cuda.is_available():
        print()
        print(f"  本次训练用的是 GPU: {torch.cuda.get_device_name(0)}")
        print(f"  峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
