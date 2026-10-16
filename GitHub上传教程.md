# 把 06_深度学习 传到 GitHub —— 手把手教程

> 仓库根目录 = `~/projects/06_深度学习/`
> 也就是说，打开仓库首页看到的就是「教材导航.md」「周报-10月第一周.md」「练习/」这些。
>
> 桌面上的 `Git.md` 是**公司版**（Gitee + gitflow + Pull Request + rebase），
> 跟你这个个人仓库的场景不一样，别照那个做。以本文为准。

---

## 〇、已经替你做完的部分

下面这些不用你动手了：

| 项目 | 状态 |
|---|---|
| `.gitignore`（过滤规则） | ✅ 已写好，`~/projects/06_深度学习/.gitignore` |
| 删除 3 个 `.bak` 备份文件 | ✅ 已删 |
| 删除 `.ruff_cache/`、`__pycache__/` | ✅ 已删 |
| `git init` + `git add`（预演） | ✅ 已做，可以随时重来 |
| 确认上传内容 | ✅ 13 个文件、120 KB |

**当前状态**：仓库已初始化、文件已暂存，**但还没提交、还没连 GitHub**。

**你需要做的**：下面第一 ~ 四节。大概 10 分钟。

---

## 一、先在网页上建一个空仓库

1. 打开 https://github.com/new
2. 填三个地方：

   **Repository name**：`d2l-pytorch-practice`
   （名字随便起，建议用英文、小写、用 `-` 连接。别用中文名，虽然支持但容易出乱子）

   **Description**（可选）：`《动手学深度学习》PyTorch 版 逐章练习代码`

   **Visibility**：选 **Private** 🔒
   （师兄要看你再加 Collaborator，见第七节）

3. ⚠️ **最容易踩的坑**：下面那三个勾选框

   ```
   ☐ Add a README file          ← 不要勾！
   ☐ Add .gitignore             ← 不要勾！
   ☐ Choose a license           ← 不要勾！
   ```

   **一个都别勾。** 建一个完全空的仓库。
   如果你勾了 README，GitHub 会先帮你提交一次，你本地推送时就会撞上
   `! [rejected] main -> main (fetch first)` 这个错，还得先 pull 才能推，多一堆麻烦。

4. 点绿色 **Create repository**

5. 建完会跳到一个页面，上面有一堆命令。**别照抄那个页面上的命令**，
   因为上面写着 `git branch -M main` 之类的，而你本地已经 init 过了。
   以本文第二节为准。

---

## 二、配好 Git 身份（一次性）

打开 WSL 终端（桌面上的 **Ubuntu** 图标，或在 VS Code 里按 `` Ctrl+` ``）。

**先检查现在是什么状态**：

```bash
git config --global --list
```

**如果 `user.name` / `user.email` 是空的，或者 email 是 `yohan@localhost` 这种假的**，
就设成你 GitHub 的：

```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub注册邮箱"
```

> ⚠️ 这两个值会写进每一条提交记录，**公开可见、永久可查**。
> 所以别填真名、别填不想暴露的邮箱。
>
> **更专业的做法**：用 GitHub 的匿名邮箱。到
> https://github.com/settings/emails → 勾上 *Keep my email addresses private*，
> 页面会给你一个 `12345678+用户名@users.noreply.github.com`，用这个。

顺手把几个好用的配置设上（可选，但建议）：

```bash
git config --global init.defaultBranch main    # 新仓库默认叫 main 而不是 master
git config --global pull.rebase false          # 将来 pull 时用 merge，不折腾 rebase
git config --global fetch.prune true           # 远端删掉的分支，本地也清掉
git config --global core.editor nano           # 需要写提交说明时用 nano，比 vim 好上手
git config --global push.default simple        # 只推当前分支
git config --global color.ui auto              # 终端里彩色显示
```

---

## 三、配 SSH 密钥（一次性，也是最容易卡住的一步）

**为什么需要**：推代码时 GitHub 得知道「你是你」。
两种方式——HTTPS 要输账号密码（现在还得用 token，麻烦），
SSH 配一次以后永久免密。**用 SSH。**

### 3.1 生成密钥

```bash
ls ~/.ssh
```

如果提示 `No such file or directory`，先建目录：

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
```

然后生成（**把邮箱换成你的**）：

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
```

接下来会连着问三个问题，**全部直接按回车**：

```
Enter file in which to save the key (/home/yohan/.ssh/id_ed25519):      ← 回车
Enter passphrase (empty for no passphrase):                             ← 回车
Enter same passphrase again:                                            ← 回车
```

> **passphrase 是什么**：给私钥文件再加一层密码。
> 更安全，但每次推代码都要输一遍。个人学习用，直接回车留空就行。
> （想安全又不想输密码，可以用 `ssh-agent`，那是后话。）

成功的话会打印一个「随机艺术图」，类似：

```
+--[ED25519 256]--+
|       .o.       |
|      .oo+       |
|     . +o+.      |
+----[SHA256]-----+
```

### 3.2 把公钥复制到剪贴板

```bash
cat ~/.ssh/id_ed25519.pub
```

会输出一行，形如：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI...很长一串... 你的GitHub邮箱
```

**把整行完整选中复制**（从 `ssh-ed25519` 一直到最后的邮箱，中间不能断行）。

> 🔑 **`.pub` 结尾的是公钥，可以随便给人。**
> 没有 `.pub` 的 `id_ed25519` 是**私钥，绝对不能外传、不能上传、不能发群里。**
> 这个 `.gitignore` 挡不住它，只能靠你自己记住。

### 3.3 贴到 GitHub

1. 打开 https://github.com/settings/keys
2. 点 **New SSH key**
3. **Title**：填个好认的名字，比如 `WSL-Ubuntu-笔记本`
   （将来换了电脑要删旧 key，有名字才知道删哪个）
4. **Key type**：保持 `Authentication Key`
5. **Key**：粘贴刚才复制的那一整行
6. 点 **Add SSH key**

### 3.4 测试

```bash
ssh -T git@github.com
```

**第一次会问**：

```
The authenticity of host 'github.com (20.205.243.166)' can't be established.
ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

输入 `yes` 回车。（这是在确认「对面真的是 GitHub 吗」，那串指纹是 GitHub 官方的，可以放心。）

**看到这个就成功了一半**：

```
Hi 你的用户名! You've successfully authenticated, but GitHub does not provide shell access.
```

> 最后那句 `does not provide shell access` **不是错误**。
> GitHub 本来就不给你登录服务器，它只回这么一句证明「认出来了，你是 你的用户名」。

---

## 四、提交并推送

回到 `06_深度学习` 目录：

```bash
cd ~/projects/06_深度学习
```

### 4.1 提交

```bash
git commit -m "feat: 初始化《动手学深度学习》练习代码"
```

> 提交信息（commit message）的惯例：
> `feat:` 新功能 / `fix:` 修 bug / `docs:` 改文档 / `refactor:` 重构 / `chore:` 杂事
> 不是强制的，但师兄一看就知道你在干嘛。中文英文都行。

### 4.2 连上远端

**`<你的用户名>` 和 `<仓库名>` 要换成你自己的**：

```bash
git remote add origin git@github.com:<你的用户名>/d2l-pytorch-practice.git
```

检查连对没有：

```bash
git remote -v
```

应该输出两行（fetch 和 push），地址是 `git@github.com:...` 开头的。

> ⚠️ 如果这里写错了，改法是：
> `git remote set-url origin git@github.com:正确的地址.git`

### 4.3 保证分支叫 main

```bash
git branch -M main
```

`-M` = 强制改名。现在 Git 默认就是 `main`，所以这个命令大概率没输出，**没输出就是对的**。

### 4.4 推！

```bash
git push -u origin main
```

`-u` 是 `--set-upstream`，作用是**记住**「本地 main ↔ 远端 origin/main 的对应关系」。
**只有第一次推需要加 `-u`**，以后直接 `git push` 就行。

成功的话会看到：

```
Enumerating objects: 20, done.
...
To github.com:你的用户名/d2l-pytorch-practice.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
```

### 4.5 去看一眼

刷新 https://github.com/你的用户名/d2l-pytorch-practice

应该能看到 13 个文件。**点开 `练习/` 看中文路径有没有乱码**——
GitHub 网页端对中文文件名支持没问题，但确认一下安心。

**顺便确认 `_教材/` 和 `data/` 没上去。** 没上去就对了。

---

## 五、以后的日常流程

改完代码，三条命令：

```bash
cd ~/projects/06_深度学习
git add .
git commit -m "feat: 完成 2.5 自动微分练习"
git push
```

**就这三条。** 大部分时间你只会用到这个。

想先看看改了啥再提交：

```bash
git status          # 哪些文件改了（红色=没暂存，绿色=已暂存）
git diff            # 具体改了什么内容
git log --oneline   # 提交历史，一行一条
```

---

## 六、用 VS Code 图形界面（不想敲命令就用这个）

VS Code 左侧竖排图标里那个**分叉图形**（源代码管理，`` Ctrl+Shift+G ``）：

1. **更改** 列表里是改动的文件
2. 文件右边点 **`+`** = 相当于 `git add`（暂存）
3. 上方输入框写提交信息，点 **✓ 提交**
4. 点 **`...`** 菜单 → **推送**，或直接点状态栏左下角的 **⟳**（同步更改）

**好处**：改动的文件能并排 diff 对比，比 `git diff` 在终端里看舒服得多。
**坏处**：第一次配 remote 还是得敲命令。

> 有时候 VS Code 上会弹「是否要定期运行 `git fetch`？」→ 点 **是**，
> 它只是帮你定期检查远端有没有新东西，没坏处。

---

## 七、让师兄能看

Private 仓库别人默认看不到。两种给法：

### 方式一：加 Collaborator（推荐）

https://github.com/你的用户名/d2l-pytorch-practice/settings/access
→ **Add people** → 填师兄的 GitHub 用户名或邮箱 → 选 **Write** 权限

师兄会收到邀请邮件，接受后就能直接看、直接提 issue。
（给 **Read** 权限也行，他就只能看不能改。）

### 方式二：改成公开

Settings → 拉到最下面 **Danger Zone** → **Change visibility** → **Make public**

**先想清楚**：公开之后所有人都能看到，包括你的提交邮箱、提交时间、
以及**历史里所有内容**（哪怕后来删了，历史里还在）。
你这个仓库全是练习代码，公开问题不大，但知道一下总没坏处。

---

## 八、常见报错对照表

### `Permission denied (publickey)`

SSH 没配好。按顺序排查：

```bash
ssh -T git@github.com          # 能认出你吗？
ssh-add -l                     # 密钥加进 agent 了吗？
cat ~/.ssh/id_ed25519.pub      # 公钥内容，跟 GitHub 网页上贴的对得上吗？
```

常见原因：
- 公钥**复制时断了行**或漏了结尾的邮箱 → 回 GitHub 删掉重新贴
- 贴的是私钥（没有 `.pub` 的那个）→ 换成 `.pub`
- remote 地址写成了 HTTPS（`https://github.com/...`）→
  `git remote set-url origin git@github.com:用户名/仓库.git`

### `src refspec main does not match any`

本地还没有 `main` 分支——**多半是还没提交过**。

```bash
git log --oneline      # 有输出吗？
```

如果报 `does not have any commits yet`，说明只 `add` 了没 `commit`，补上：

```bash
git commit -m "feat: 初始提交"
```

### `! [rejected] main -> main (fetch first)`

远端有你本地没有的东西。**99% 是因为建仓库时勾了 README**。两种解法：

```bash
# 解法 A：把远端的东西拉下来合并（会保留那个 README）
git pull --no-rebase origin main
git push -u origin main

# 解法 B：远端那个 README 不要了，直接覆盖（只在确定远端没别的东西时用）
git push -u origin main --force
```

### `fatal: not a git repository`

你不在 `06_深度学习` 目录里，或者这个目录还没 `git init`。

```bash
cd ~/projects/06_深度学习
git status
```

### `Updates were rejected because the remote contains work...`

同 `fetch first`，是同一个问题的另一种说法。

### 想把 `mnist_model.pth` 之类的大文件传上去

**别传。** 已经用 `.gitignore` 挡住了。GitHub 单文件超过 **100 MB** 会直接拒收，
而且大文件进了提交历史就**很难彻底删掉**（要用 `git filter-repo` 重写历史）。
模型文件能重新训练出来，数据集能重新下载，都不该进仓库。

### 万一真的手滑把不该传的传上去了

```bash
# 1. 从 Git 的追踪里移除（保留本地文件）
git rm -r --cached venv/
git rm -r --cached data/MNIST/

# 2. 把它加进 .gitignore（关键！否则下次 add 又进来了）
echo "venv/" >> .gitignore
echo "data/MNIST/" >> .gitignore

# 3. 提交
git add .gitignore
git commit -m "chore: 移除不该追踪的文件"
git push
```

⚠️ 注意：**这样只是「以后不追踪」，历史里那次提交还在。** 如果传的是密码/密钥，
必须用 `git filter-repo` 重写历史，并且**立刻去把那个密钥作废**（重新生成）。

---

## 九、速查：这次要用到的全部命令

```bash
# ——— 一次性配置 ———
git config --global user.name "用户名"
git config --global user.email "GitHub邮箱"
ssh-keygen -t ed25519 -C "GitHub邮箱"       # 三个问题全回车
cat ~/.ssh/id_ed25519.pub                    # 复制这行到 GitHub
ssh -T git@github.com                        # 测通了再做下一步

# ——— 首次推送（在 ~/projects/06_深度学习 里）———
git commit -m "feat: 初始化《动手学深度学习》练习代码"
git remote add origin git@github.com:<用户名>/d2l-pytorch-practice.git
git branch -M main
git push -u origin main

# ——— 以后每天 ———
git add .
git commit -m "说明这次改了什么"
git push
```

---

## 十、这次到底传了什么（存档备查）

**传上去的 13 个文件（120 KB）**：

```
.gitignore
.vscode/settings.json
教材导航.md
周报-10月第一周.md
第一个神经网络.py
练习/README.md
练习/草稿纸.py
练习/_杂项/验证测试.ipynb
练习/第02章_预备知识/01_张量操作.py
练习/第02章_预备知识/02_数据预处理.py
练习/第02章_预备知识/03_线性代数.py
练习/第02章_预备知识/04_微积分.py
练习/第02章_预备知识/data/student.csv
```

**被 `.gitignore` 挡住的**：

| 挡掉的 | 体积 | 为什么 |
|---|---|---|
| `_教材/` | 60 MB | d2l 官方教材，别人能自己下载 |
| `data/` | 64 MB | MNIST 数据集，跑代码会自动下回来 |
| `mnist_model.pth` | 1.7 MB | 训练产物，能重新训练出来 |
| `__pycache__/`、`.ruff_cache/` | — | 运行缓存，每台机器不一样 |
| `*.bak` | — | 调试时留下的备份 |

**126 MB → 120 KB，省了 99.9%。**

---

*最后更新：2025-10-16*
