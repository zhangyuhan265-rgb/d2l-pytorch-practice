#%%
import os
from matplotlib import axis
import numpy as np
import pandas as pd
import torch



A = torch.arange(40).reshape(2,5,4)
A
A.shape
A.sum()

# %%
B = torch.arange(6,dtype=torch.float32)
B.sum()

# %%
A = torch.arange(40).reshape(2,5,4)
A_sum_axis0 = A.sum(dim=0)
A_sum_axis0
A_sum_axis0.shape

# %%
A_sum_dim1 = A.sum(dim=1)
A_sum_dim1
A_sum_dim1.shape

# %%
A_sum_dim2 = A.sum(dim=2)
A_sum_dim2
A_sum_dim2.shape

# %%
A.sum(dim=[0,1])
A.sum(dim=[0,1]).shape
# %%
A.sum(dim=[0,2])
#A.sum(dim=[0,2]).shape
# %%
import numpy as np
import matplotlib.pyplot as plt        # ← 惯例用 plt，不用 plot

def f(x):
    return 3 * x ** 2 - 4 * x

x = np.arange(0, 3, 0.1)

plt.plot(x, f(x), label='f(x)')                     # 第一条线
plt.plot(x, 2 * x - 3, 'r--', label='切线 (x=1)')    # 第二条：r=红，--=虚线
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()                            # 显示图例（文字取自 label）
plt.grid(True, alpha=0.3)
plt.show()

# %%
