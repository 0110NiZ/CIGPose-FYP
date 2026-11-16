# 在 WSL 提交 Module A

## 目录

实际 Git 仓库：`/home/nzd/projects/cig-module-a-n/code`。
分支：`feature/module-a`。远端：`https://github.com/0110NiZ/CIGPose-FYP.git`。
外层 `cig-module-a-n` 是工作目录，不能直接在外层运行提交命令。
日常直接修改这里的 `Module_A`，无需再与 Windows 仓库来回复制。

## 本次已准备好的提交

代码、A2 小报告及相关文档已暂存，尚未提交和推送：

```bash
cd ~/projects/cig-module-a-n/code
git diff --cached --stat
git commit -m "Add A2 ranking experiment and WSL upload guide"
git push origin feature/module-a
```

## 以后修改后上传

```bash
cd ~/projects/cig-module-a-n/code
git status
git diff -- Module_A
git add -- Module_A experiments/module_a README.md .gitignore docs/fyp/git_usage.md
git diff --cached --stat
git commit -m "Describe this change"
git push origin feature/module-a
```

公共任务在本机还有独立改动，继续归 `public-tasks` 管理。不要用 `git add .` 把所有公共任务改动一起混入 A 的提交。
新增其他文档时，明确 `git add 文档路径`。上传失败时不要 force push，先查看报错。

## 新实验报告

把新实验的 `history.json`、`settings.json`、`result.txt` 复制到
`experiments/module_a/实验名称/`，并写 `README.md` 解释配置、最佳轮次和对照结果。
模型权重、缓存、标签和大预测文件留在外层工作目录；不要使用 `git add -f` 绕过忽略规则。

## 登录

当前仓库使用 WSL 已登录的 GitHub CLI 提供凭据，已检查仓库写入权限。
如以后登录失效，在 Ubuntu 中执行 `gh auth login`，按提示完成 GitHub 登录，再重试 push。
不需要把密码或 token 写进代码、README 或 remote URL。
