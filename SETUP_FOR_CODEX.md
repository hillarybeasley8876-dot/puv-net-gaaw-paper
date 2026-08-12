# SETUP_FOR_CODEX — 你本机 PowerShell 复制粘贴就能跑

> **重要**：这一步是给接手者（你 / Codex）看的。**我**不会替你执行任何 git 命令。
> 全部在 `E:\AE-CC托管\puv-net` 下进行。
>
> 本仓根目录：`E:\AE-CC托管\puv-net`（子项目，未配 remote，需要你自己在 GitHub 创空仓）

---

## 0. 前置（你那边已完成的）

- [x] 本机 git config 已设（`user.name=CEO`, `user.email=ceo@nightledger.game`）
- [x] `.gitignore` 已改好（屏蔽大件，反向放行判据文件）—— 在仓根 `E:\AE-CC托管\puv-net\.gitignore`
- [x] `INDEX.md` 已写（给接手者看的入口）

---

## 1. 在 GitHub 创一个空仓

去 https://github.com/new ，**Initialize 什么都不要勾**（不要 README / .gitignore / license），命名例如 `puv-net-gaaw-paper` 或你喜欢的名字。创完把 URL 复制下来，类似：

```
https://github.com/<your-owner>/<repo-name>.git
```

---

## 2. 复制粘贴执行（PowerShell）

打开 PowerShell，**逐段**执行（每段按 Enter 一次，等完成再下一段）。把 `<URL>` 换成你创的 URL。

### 2.1 初始化子项目仓
```powershell
cd 'E:\AE-CC托管\puv-net'
git init
git checkout -b main
```

### 2.2 验证 .gitignore 真的生效了（看会跟踪什么）
```powershell
git status --short | Measure-Object -Line
```
应该远小于 1813（脚本里估算的「无 .gitignore 下的总数」）。如果**等于或接近 1813**，说明 `.gitignore` 没生效，先停下检查。

**更严格：检查具体敏感文件是否被屏蔽**
```powershell
git check-ignore backups/*.zip 'data/PU1K_extract/PU1K/test/input_1024/gt_4096/02691156.37f2f187a1582704a29fef5d2b2f3d7.xyz' .env.local
```
全部回显路径 = 全部被忽略 = 安全。

### 2.3 看会进仓的文件
```powershell
git add -A --dry-run | Select-Object -First 50
```
这 50 个文件先扫一眼，确认没有：
- `.env` / `*.key` / `*.pem` / `secrets.yaml`
- `data/PU1K_extract/...` 里的 .xyz
- `backups/*.zip`
- `runs/<name>/best.pt` 之类的大件

如果出现以上任何一项，**停**，告诉我我改 `.gitignore`。

### 2.4 一次性 add
```powershell
git add -A
git status --short | Measure-Object -Line
```

### 2.5 commit
```powershell
git commit -m "puv-net paper archive: GAAW baseline + 6 ablations + paper draft

- 9 paper runs (3 on 5090, 6 on 3090) with calibrated cv_nn, CD, HD, NUC
- 150-epoch training traces + selection + summary_stats
- GAAW chapter 4 (formulation + rho curve evidence)
- 3.5.5 pre-registration executed; baseline revised to B1 in ch4
- STYLE_GUIDE §2.9 / §2.9.1 / FORMAT_TONGJI / INDEX"
```

### 2.6 推上去
HTTPS 方式（**最通用**，把 `<TOKEN>` 换成 GitHub PAT）：
```powershell
git remote add origin 'https://<TOKEN>@github.com/<your-owner>/<repo-name>.git'
git push -u origin main
```

SSH 方式（**如果你 GitHub 已加 SSH 公钥**）：
```powershell
git remote add origin 'git@github.com:<your-owner>/<repo-name>.git'
git push -u origin main
```

推完把 GitHub 页面 URL 复制下来，**告诉我**，我会把链接写进 `EVIDENCE_LEDGER.md` 第 423 行的"论文仓库链接"占位。

---

## 3. 我做不了、你必须自己做的 4 件事

1. **GitHub 创空仓**（我不能注册或创建账号）
2. **配 push 凭据**（PAT 或 SSH key；不能让我接触凭据）
3. **执行 git init / commit / push**（你说了自己走 CLI）
4. **验证推上去没漏不该推的东西**（在 GitHub 页面点开看文件列表）

---

## 4. 推完后给我一句话

「OK 推了，链接是 https://github.com/xxx/yyy」
我就把 `EVIDENCE_LEDGER.md` 第 423 行那行占位填上，备份 cron 加一项 "git-archive-ok"。
