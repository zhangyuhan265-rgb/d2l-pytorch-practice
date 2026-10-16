#%% 第 0 格：说明 + 导入（每次开新会话先跑这格）
print("=" * 62)
print(" 2.2 数据预处理 —— 练习")
print("=" * 62)
print("""
【教材】https://zh-v2.d2l.ai/chapter_preliminaries/pandas.html

【本节三件事】
  1. 读取数据集        read_csv
  2. 处理缺失值        fillna / get_dummies
  3. 转换为张量        torch.tensor

【怎么用这个文件】
  它用 #%% 分成多个单元格：
    - 光标放在某一格里，按 Shift+Enter 只跑这一格
    - 必须先跑本格（第 0 格），否则后面会 NameError

【本节有两个真坑】
  - 坑1：fillna 默认不修改原对象（实验 5）
  - 坑2：混合类型转 numpy 会变 object，torch 拒收（实验 8）
""")

import os
import numpy as np
import pandas as pd
import torch

print("pandas 版本:", pd.__version__)
print("numpy  版本:", np.__version__)
print("torch  版本:", torch.__version__)


#%% 实验 1：造一个自己的数据文件
print("--- 实验 1：造数据 ---")

# 教材用房价数据。我们换个场景：学生成绩
# 故意留几个空洞（NA）来模拟真实数据的缺失值
os.makedirs("data", exist_ok=True)
csv_path = os.path.join("data", "student.csv")

with open(csv_path, "w", encoding="utf-8") as f:
    f.write("数学,英语,班级,专业,总分\n")
    f.write("85,90,1班,计算机,175\n")
    f.write("NA,78,1班,计算机,155\n")       # 数学缺失
    f.write("92,NA,2班,数学,180\n")         # 英语缺失
    f.write("78,85,NA,计算机,163\n")        # 班级缺失
    f.write("NA,NA,2班,数学,140\n")         # 两个都缺失
    f.write("88,91,1班,NA,179\n")           # 专业缺失

print("已写入:", os.path.abspath(csv_path))
print()
print("原始文件内容:")
print("-" * 40)
with open(csv_path, encoding="utf-8") as f:
    for line in f:
        print("  " + line.rstrip())
print("-" * 40)
print()
print("数一数：有 6 个位置是 NA（缺失值）")


#%% 实验 2：读进 DataFrame
print("--- 实验 2：read_csv ---")

data = pd.read_csv(csv_path)
print("data 的样子:")
print(data)
print()
print("形状     :", data.shape, "  (行, 列)")
print("列名     :", list(data.columns))
print()
print("每列类型 :")
for col in data.columns:
    print(f"    {col:<6} {str(data[col].dtype):<10}")
print()

# TODO 先猜，再取消注释
# print("data.head(3) 只看前 3 行:")
# print(data.head(3))
# print()
# print("data.info() 会告诉你每列有多少非空值:")
# data.info()


#%% 实验 3：看看到底缺了多少
print("--- 实验 3：缺失值统计 ---")

print("isna() 的结果（True 表示缺失）:")
print(data.isna())
print()
print("每列缺失个数:")
print(data.isna().sum())
print()

# 关键：isna() 返回的是【布尔 DataFrame】
# 和 2.1 学的布尔张量是一回事，只是换了 pandas 的壳
print("isna() 返回的类型:", type(data.isna()).__name__)
print("它的形状        :", data.isna().shape)
print("布尔值可以当 1/0 求和 —— 和 2.1 里 (X==Y).sum() 一个道理")
print()

# TODO 先猜下面会输出什么，再取消注释
# print("总共缺了几个格子:", data.isna().sum().sum())
# print("有没有任何缺失  :", data.isna().any().any())
# print("哪些行有缺失    :", data.isna().any(axis=1).tolist())


#%% 实验 4：iloc 切片 —— 分离特征和目标
print("--- 实验 4：iloc 切片 ---")

print("data 的列:", list(data.columns))
print("           下标:  0=数学  1=英语  2=班级  3=专业  |  4=总分")
print()

inputs = data.iloc[:, 0:4]     # 前 4 列：特征
outputs = data.iloc[:, 4]      # 第 4 列：目标
print("inputs 形状 :", inputs.shape, "  <- 二维 (6行4列)")
print("outputs 形状:", outputs.shape, "  <- 一维 (6个)")

print("=" * 50)
print(" 知识点：iloc 和 loc 的区别")
print("=" * 50)
print("""
  iloc  ->  i = integer  ->  用【数字下标】   data.iloc[:, 0:2]
  loc   ->  loc = label  ->  用【列名】       data.loc[:, '数学':'英语']

  口诀: iloc 是"第几列", loc 是"叫什么"
""")
print("  data.iloc[:, 0]     ->", data.iloc[:, 0].tolist())
print("  data.loc[:, '数学'] ->", data.loc[:, "数学"].tolist())
print("  上面两个结果一样")
print()
print("  loc 的切片是【闭区间】，iloc 是【左闭右开】")
print("    data.iloc[:, 0:2]          -> 取下标 0,1 （2 列）")
print("    data.loc[:, '数学':'英语'] -> 取数学、英语 （也是 2 列）")
print()

# TODO 先猜：loc 从'数学'到'专业'是几列？
# print("data.loc[:, '数学':'专业'] 列数 =", data.loc[:, '数学':'专业'].shape[1])
# print("data.iloc[:, 0:3]          列数 =", data.iloc[:, 0:3].shape[1])


#%% 实验 5：填充数值列的缺失值
print("--- 实验 5：fillna 填均值 ---")

print("填充前 inputs:")
print(inputs)
print()
print("数值列的均值:")
print(inputs.mean(numeric_only=True))
print()

inputs2 = inputs.fillna(inputs.mean(numeric_only=True))
print("填充后 inputs2:")
print(inputs2)
print()

print("=" * 50)
print(" 坑 1：fillna 默认返回【新对象】，不改原来那个")
print("=" * 50)
print()
print("  原始 inputs 的数学列:", inputs["数学"].tolist())
print("  inputs2   的数学列  :", inputs2["数学"].tolist())
print()
print("  ^^^ inputs 还是带 NaN 的！因为 fillna 返回新对象")
print()
print("  如果写 inputs.fillna(...) 但不赋值给任何变量 -> 白写了！")
print("  教材写的是 inputs = inputs.fillna(inputs.mean())")
print("  那个 '=' 赋值不能省。这是新手最常犯的错之一。")
print()

# ============================================================
# TODO 实验：inplace=True 的那些坑
# ============================================================
#
# 坑 A: 用一个值填所有列 -> 会在 str 列上报错
#       tmp = inputs.copy()
#       tmp.fillna(0, inplace=True)
#       -> TypeError: Invalid value '0' for dtype 'str'
#          因为 班级/专业 是字符串列，不能填整数 0
#
# 坑 B: 先选列再 inplace -> ChainedAssignmentError
#       tmp[['数学','英语']].fillna(0, inplace=True)
#       -> 选出来的子集是副本，对副本 inplace 改不到原来那个
#
# 正确写法 1: 按列指定不同的填充值
#       tmp = inputs.copy()
#       tmp.fillna({'数学': 0, '英语': 0, '班级': '空', '专业': '空'}, inplace=True)
#       print("tmp 的数学列:", tmp['数学'].tolist())
#
# 正确写法 2: 按列类型分别填（更通用，推荐）
#       tmp = inputs.copy()
#       num_cols = tmp.select_dtypes(include='number').columns
#       cat_cols = tmp.select_dtypes(exclude='number').columns
#       tmp[num_cols] = tmp[num_cols].fillna(0)
#       tmp[cat_cols] = tmp[cat_cols].fillna('空')
#       print(tmp)
#
# 顺便：inplace=True 的返回值是 None，
#       所以不能写 df = df.fillna(..., inplace=True)
#       pandas 现在推荐直接赋值: df = df.fillna({...})
#       inplace 未来版本会被移除。


#%% 实验 6：类别列的缺失值 —— 当成一个类别
print("--- 实验 6：类别列怎么办 ---")

print("班级列的值:", inputs2["班级"].tolist())
print("专业列的值:", inputs2["专业"].tolist())
print()
print("问题：班级和专业是【文字】，算不了均值")
print("      而且它们本身就是【类别】，不是数量")
print()

inputs3 = inputs2.copy()
inputs3["班级"] = inputs3["班级"].fillna("未知")
inputs3["专业"] = inputs3["专业"].fillna("未知")
print("把 NaN 填成'未知'之后:")
print(inputs3)
print()
print("现在还有 NaN 吗:", inputs3.isna().any().any())
print()

print("=" * 50)
print(" 工具：value_counts() 看每个类别有多少个")
print("=" * 50)
print()
print("  班级分布:")
print(inputs3["班级"].value_counts())
print()
print("  专业分布:")
print(inputs3["专业"].value_counts())
print()

# TODO 先猜：unique() 和 nunique() 分别返回什么？
print("班级的所有取值:", inputs3['班级'].unique()) #去重后的取值列表
print("班级有几种取值:", inputs3['班级'].nunique()) #有几种取值


#%% 实验 7：独热编码 —— 文字变数字
print("--- 实验 7：get_dummies 独热编码 ---")

print("编码前（文字没法喂给神经网络）:")
print(inputs3[["班级", "专业"]])
print()

encoded = pd.get_dummies(inputs3, dtype=float)
print("编码后（变成一堆 0.0/1.0 列）:")
print(encoded)
print()
print("列数变化:", inputs3.shape[1], "->", encoded.shape[1])
print("列名:", list(encoded.columns))
print()

print("=" * 50)
print(" 独热编码是什么")
print("=" * 50)
print("""
  把"一个有很多取值的列"拆成"每个取值一列"，用 0/1 表示。

  班级 = ['1班', '2班', '未知']
    ->
  班级_1班   班级_2班   班级_未知
      1.0        0.0         0.0      <- 原来是 1班
      0.0        1.0         0.0      <- 原来是 2班
      0.0        0.0         1.0      <- 原来是 未知

  为什么必须这样：
    神经网络只认数字。
    但不能用 1/2/3 给班级编号 —— 那会暗示"2班 > 1班"，
    而班级之间没有大小关系。独热编码避免了这种假的大小关系。
""")
print()

print("=" * 50)
print(" dtype=float 是关键（否则会踩坑 2）")
print("=" * 50)
print()
bug = pd.get_dummies(inputs3)
print("  不加 dtype=float 时:")
print("     生成的 0/1 列是 bool 类型:", bug["班级_1班"].dtype)
print("     转 numpy 后整体类型  :", bug.values.dtype, " <- object！")
print()
good = pd.get_dummies(inputs3, dtype=float)
print("  加了 dtype=float 之后:")
print("     生成的列是 float 类型 :", good["班级_1班"].dtype)
print("     转 numpy 后整体类型   :", good.values.dtype, " <- 可以喂给 torch")
print()
print("  原理：DataFrame 里混了 float 列和 bool 列，")
print("        转 numpy 时只能统一成 object（万能类型），")
print("        而 torch 不接受 object。")
print()
print("  这就是教材那版代码在你的 pandas 3.0 上会报错的原因。")
print()


#%% 实验 8：转成张量 —— 这里有个真坑
print("--- 实验 8：转成 torch 张量 ---")

print("encoded.values 的类型    :", type(encoded.values).__name__)
print("encoded.values 的形状    :", encoded.values.shape)
print("encoded.values 的元素类型:", encoded.values.dtype)
print()

X = torch.tensor(encoded.values, dtype=torch.float32)
y = torch.tensor(outputs.values, dtype=torch.float32)
print("X 的形状:", tuple(X.shape), "  dtype:", X.dtype)
print("y 的形状:", tuple(y.shape), "  dtype:", y.dtype)
print()
print("X =")
print(X)
print()
print("y =", y)
print()

print("=" * 50)
print(" 坑 2 完整演示：object 类型 torch 拒收")
print("=" * 50)
print()
bad_df = pd.get_dummies(inputs3)      # 故意不加 dtype=float
print("  bad_df.values.dtype =", bad_df.values.dtype)
print()
print("  试三种写法，看哪个能成：")
print()
try:
    torch.tensor(bad_df.values)
    print("  [1] torch.tensor(bad_df.values)                      成功")
except TypeError as e:
    print("  [1] torch.tensor(bad_df.values)                      失败")
    print("      ", str(e)[:75])
print()
try:
    torch.tensor(bad_df.values, dtype=torch.float32)
    print("  [2] torch.tensor(bad_df.values, dtype=torch.float32) 成功")
except TypeError as e:
    print("  [2] torch.tensor(bad_df.values, dtype=torch.float32) 失败")
    print("      ", str(e)[:75])
    print("      ^^^ 指定 dtype 也救不了！因为 torch 先检查 numpy 数组本身")
print()
try:
    torch.tensor(bad_df.values.astype(np.float32))
    print("  [3] torch.tensor(bad_df.values.astype(np.float32))   成功")
except TypeError as e:
    print("  [3] ...astype(np.float32)                            失败")
print()
print("  结论：必须【先转 numpy 类型】，再交给 torch")
print("        两个办法：")
print("          A. pd.get_dummies(df, dtype=float)     <- 推荐，一步到位")
print("          B. df.values.astype(np.float32)        <- 转张量时补一刀")
print()


#%% 实验 9：完整流程串一遍
print("--- 实验 9：从头到尾跑一遍完整流程 ---")
print()

def load_and_prepare(csv_file):
    """把 CSV 变成可以直接喂给模型的 X, y"""
    df = pd.read_csv(csv_file)

    # 1. 分离特征和目标
    feats = df.iloc[:, 0:4].copy()
    target = df.iloc[:, 4]

    # 2. 自动分辨数值列和类别列
    num_cols = feats.select_dtypes(include="number").columns
    cat_cols = feats.select_dtypes(exclude="number").columns

    # 3. 数值列填均值，类别列填"未知"
    feats[num_cols] = feats[num_cols].fillna(feats[num_cols].mean())
    feats[cat_cols] = feats[cat_cols].fillna("未知")

    # 4. 类别列独热编码（dtype=float 避免 object 问题）
    feats = pd.get_dummies(feats, dtype=float)

    # 5. 转张量
    X = torch.tensor(feats.values, dtype=torch.float32)
    y = torch.tensor(target.values, dtype=torch.float32)
    return X, y

X2, y2 = load_and_prepare(csv_path)
print("完整流程结果:")
print("  X.shape =", tuple(X2.shape), "  dtype =", X2.dtype)
print("  y.shape =", tuple(y2.shape), "  dtype =", y2.dtype)
print()
print("X =")
print(X2)
print()
print("y =", y2)
print()

print("=" * 50)
print(" 这两个工具比一列一列手动处理强")
print("=" * 50)
print("""
  select_dtypes(include='number')  -> 自动挑出数值列
  select_dtypes(exclude='number')  -> 自动挑出类别列

  好处：列名变了、加了新列，代码都不用改。
""")
print()


#%% 实验 10：章节练习题（教材末尾的题）
print("--- 实验 10：章节练习 ---")
print("""
教材 2.2 末尾有两道题：

练习 1: 删除缺失值最多的列。
练习 2: 将预处理后的数据集转换为张量格式。（实验 8 已做过）

练习 1 的思路提示（先别看，自己试）：
  - data.isna().sum() 看每列缺几个
  - .idxmax() 找出最大的那个列名
  - data.drop(columns=[...]) 删掉

在下面写你的答案：
""")

# ===== 在这里写练习 1 的答案 =====


most_missing_col = data.isna().sum().idxmax()
print("缺失最多的列:", most_missing_col, " 缺", data.isna().sum().max(), "个")
data_dropped = data.drop(columns=[most_missing_col])
print("删除后的列:", list(data_dropped.columns))
print("新形状:", data_dropped.shape)

New_X = torch.tensor(encoded.values,dtype=torch.float32)
New_Y = torch.tensor(outputs.values,dtype=torch.float32)
print(New_X)
print(New_Y)


#%% 踩坑记录（自己填）

# 格式： 现象 -> 原因
#
# 例：
#   调了 fillna 但数据没变 -> fillna 默认返回新对象，必须赋值回去
#
# 你的记录：
#
#
#


#%% 最后一格：自检
print("=" * 62)
print(" 如果上面每一格都能跑通，2.2 就过关了")
print("=" * 62)
print()
print("对照一下，这些你能说清楚吗：")
print("  1. read_csv 读进来的是什么类型？")
print("  2. iloc 和 loc 的区别？loc 的切片是开区间还是闭区间？")
print("  3. 数值列的缺失值为什么用均值填？")
print("  4. 类别列的缺失值为什么不能填均值？")
print("  5. 独热编码把 4 列变成了几列？为什么？")
print("  6. fillna 之后不赋值会怎样？")
print("  7. 为什么 get_dummies 要加 dtype=float？")
print()
print("说不清楚的那条，回去重跑对应的实验。")
print("=" * 62)
