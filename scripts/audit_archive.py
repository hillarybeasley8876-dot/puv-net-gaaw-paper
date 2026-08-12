# -*- coding: utf-8 -*-
"""实验存档完整性审计 —— 论文溯源的最后一道闸。

为什么需要这个脚本
------------------
现有 selfcheck_* 都是「单点校验」(引用编号 / 图有效性 / 判据一致性)。
但写论文时真正会咬人的是**跨 run 的存档完整性**:

1. 某个 run 的 clouds/ 少了几个 epoch -> 定性演进图缺帧, 事后无法重建
2. summary_stats.json 缺失 -> 主表报数没有出口, 只能手算 (违反"图必须由落盘数据重绘")
3. env.json 里 gpu_name 不同的 run 被并列进同一张表 -> 跨机器混排 (红线)
4. history.json 的 epoch 数 < config 里的 epochs -> run 中途崩了但没人注意
5. figures/*.png 有但同名 .data.json 没有 -> 图的数字无法溯源

这些问题共同特征: **单个 run 看不出来, 只有横向比对才暴露**。

用法
----
    python scripts/audit_archive.py                # 审计全部 run
    python scripts/audit_archive.py --paper-only   # 只审计论文引用的 run
    python scripts/audit_archive.py --json         # 输出机器可读报告
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RUNS = ROOT / "runs"

# ---------------------------------------------------------------------------
# 必存清单: 来自 docs/ARTIFACT_POLICY.md §3。改这里等于改规范, 必须同步改文档。
# ---------------------------------------------------------------------------
REQUIRED_FILES = [
    "config.yaml",          # §3 配置快照
    "env.json",             # §3 环境指纹
    "history.json",         # §3 逐 epoch 标量
    "metrics.json",         # §3 验证指标
    "summary_stats.json",   # 主表报数出口 (平台区均值±σ)
    "selection.json",       # best.pt 的选点审计轨迹
    "ckpt/best.pt",
    "ckpt/last.pt",
]

REQUIRED_FIGURES = ["F_loss.png", "F_metric.png"]

# ---------------------------------------------------------------------------
# 历史豁免: 这些 run 跑在对应特性上线之前, 缺失属既成事实, 不是当下的缺陷。
# 要点: 豁免必须写明"为什么可以豁免"+"替代证据在哪", 否则就是掩盖问题。
# ---------------------------------------------------------------------------
LEGACY_EXEMPT = {
    "B001_reproduce": {
        "summary_stats.json":
            "B-001 跑在平台区报数机制上线前; 替代证据 = "
            "runs/B001_reproduce/convergence_sensitivity.json "
            "(scripts/check_convergence_sensitivity.py 事后多窗口重算)",
        "selection.json":
            "同上; 替代证据 = selection_replay.json "
            "(scripts/replay_selection_b001.py 事后按同一 selector 重放)",
    },
}

# 非 git 仓库时 git_commit 必然为 null。这不是产物缺陷, 而是工程现状。
# 判定方式: 检测项目根有无 .git, 有才要求 git_commit。
IS_GIT_REPO = (ROOT / ".git").exists()

# 论文正式引用的 run (非 smoke / 非 probe)。这个名单决定"哪些 run 必须无瑕疵"。
# 维护纪律：run 一旦跑满目标 epoch 并进入论文口径，必须当场加进本名单。
# 2026-08-12 踩坑：C1/D1/AC_combo 已完工但漏登记，审计器报"论文 run 全部通过"
# 却根本没审这三个 —— 名单漏登记 = 审计假绿，比审计失败更危险。
PAPER_RUNS = [
    "B001_reproduce",
    "B002_baseline150",
    "ABL_A1_cd_balance",
    "ABL_A2_cd_boost_bwd",
    "ABL_C1_uniform",
    "ABL_D1_scale_qk",
    "ABL_AC_combo",
    # 5090 上的 B 组 (回传后才会出现在本地)
    "B002_baseline150_5090",
    "ABL_B1_adv_fixed",
    "ABL_B2_adv_adaptive",
    # 5090 SEED 队列 (兑现 3.5.5 的 2SE 门槛; 回传后出现在本地)
    "SEED_baseline_s20260812",
    "SEED_B2_s20260812",
    "SEED_baseline_s20260813",
    "SEED_B2_s20260813",
]


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": str(e)}


def audit_run(d: Path) -> dict:
    """审计单个 run 目录, 返回 {ok, issues[], warns[], meta{}}。

    issues = 必须修的硬缺口 (阻塞论文)
    warns  = 已知/可解释的缺口 (历史豁免、正在跑、非 git 仓库), 不阻塞
    """
    issues: list[str] = []
    warns: list[str] = []
    meta: dict = {"name": d.name}
    exempt = LEGACY_EXEMPT.get(d.name, {})

    # --- 0. 先判断这个 run 是不是"还在跑" ---
    # 依据: history.json 的 mtime 距今很近 (< 10 min) 说明训练进程还在写。
    # 这一步必须最先做, 因为"还在跑"会让后面几乎所有完整性检查都误报。
    import time as _t
    hist_p0 = d / "history.json"
    running = False
    if hist_p0.exists():
        age_min = (_t.time() - hist_p0.stat().st_mtime) / 60.0
        meta["history_age_min"] = round(age_min, 1)
        running = age_min < 10.0
    meta["running"] = running

    # --- 1. 必存文件 ---
    for rel in REQUIRED_FILES:
        if not (d / rel).exists():
            if rel in exempt:
                warns.append("豁免缺失 %s —— %s" % (rel, exempt[rel]))
            elif running:
                warns.append("暂缺 %s (run 仍在跑, 收尾阶段才写)" % rel)
            else:
                issues.append("缺文件: %s" % rel)
        elif (d / rel).stat().st_size == 0:
            issues.append("空文件: %s" % rel)

    # --- 2. config 里声明的 epochs 与 history 实际条数是否一致 ---
    cfg_p, hist_p = d / "config.yaml", d / "history.json"
    n_declared = n_actual = None
    if cfg_p.exists():
        import yaml
        try:
            cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
            n_declared = int(cfg.get("epochs", 0))
            meta["epochs_declared"] = n_declared
            meta["batch_size"] = cfg.get("batch_size")
            meta["seed"] = cfg.get("seed")
            # loader 段是基础设施配置, 不同机器可不同, 但要记录下来供审阅
            meta["loader"] = cfg.get("loader") or {"num_workers": 0}
        except Exception as e:
            issues.append("config.yaml 解析失败: %s" % e)
    if hist_p.exists():
        h = load_json(hist_p)
        if isinstance(h, list):
            n_actual = len(h)
            meta["epochs_actual"] = n_actual
            if h and isinstance(h[-1], dict):
                meta["last_sec"] = h[-1].get("sec")
                meta["gpu_peak_gb"] = h[-1].get("gpu_peak_gb")
        else:
            issues.append("history.json 不是 list")
    if n_declared and n_actual is not None and n_actual < n_declared:
        if running:
            warns.append("进度 %d/%d (仍在跑, %.1f 分钟前刚写过)"
                         % (n_actual, n_declared, meta.get("history_age_min", 0)))
        else:
            issues.append("★ 训练未跑满: history %d 条 < config epochs %d "
                          "且 %.0f 分钟无写入 (run 崩过?)"
                          % (n_actual, n_declared, meta.get("history_age_min", 0)))

    # --- 3. 环境指纹: 记录 GPU 型号, 供跨机器混排检查 ---
    env_p = d / "env.json"
    if env_p.exists():
        e = load_json(env_p)
        meta["gpu_name"] = e.get("gpu_name")
        meta["torch"] = e.get("torch")
        meta["cuda"] = e.get("cuda_version")
        meta["git_commit"] = (e.get("git_commit") or "")[:8] or None
        if not e.get("git_commit"):
            # 非 git 仓库时必然为 null —— 属工程现状, 不算产物缺陷。
            # 但要留一条 warn, 因为它确实降低了复现性 (无法锁定代码版本)。
            if IS_GIT_REPO:
                issues.append("env.json 无 git_commit, 但项目是 git 仓库 "
                              "—— 采集失败, 需修 env_fingerprint()")
            else:
                warns.append("无 git_commit (项目非 git 仓库; 复现性依赖 "
                             "config.yaml 快照 + docs/EXPERIMENT_LOG.md)")

    # --- 4. 图与 .data.json 配对 (ARTIFACT_POLICY 铁律: 图必须可溯源到数字) ---
    # 注意: 所有 figures 都在训练收尾时一次性生成, 所以 run 仍在跑时必然为空,
    # 这不是缺陷。
    fig_d = d / "figures"
    if not fig_d.exists():
        (warns if running else issues).append(
            "缺 figures/ 目录" + ("(run 仍在跑, 收尾才出图)" if running else ""))
    else:
        pngs = sorted(fig_d.glob("*.png"))
        meta["n_figures"] = len(pngs)
        for png in pngs:
            dj = png.with_suffix(".data.json")
            if not dj.exists():
                # 这条永远是硬错误: 图存在却无数据源, 直接违反"图必须由落盘数据重绘"
                issues.append("图无数据源: %s 缺同名 .data.json" % png.name)
        for need in REQUIRED_FIGURES:
            if not (fig_d / need).exists():
                (warns if running else issues).append(
                    "缺必备图: figures/%s%s"
                    % (need, " (run 仍在跑)" if running else ""))

    # --- 5. clouds 完整性: 逐个核对 manifest 声明的文件真实存在 ---
    # 这一项即使 run 在跑也要查: manifest 是每 dump_every 就追加的,
    # 已声明的文件必须已存在, 缺了就是真丢了 (事后无法重建)。
    cl_d = d / "clouds"
    man_p = d / "clouds_manifest.json"
    if cl_d.exists():
        npzs = list(cl_d.glob("*.npz"))
        meta["n_clouds"] = len(npzs)
        if man_p.exists():
            man = load_json(man_p)
            if isinstance(man, list):
                missing = 0
                for rec in man:
                    for s in rec.get("samples", []):
                        if not (cl_d / s["file"]).exists():
                            missing += 1
                if missing:
                    issues.append("★ clouds 缺 %d 个 manifest 声明的文件 "
                                  "(定性演进图会缺帧, 事后无法重建)" % missing)
                meta["n_cloud_epochs"] = len(man)
        else:
            (warns if running else issues).append("缺 clouds_manifest.json")
    else:
        (warns if running else issues).append(
            "缺 clouds/ 目录 (定性图唯一来源)")

    # --- 6. summary_stats 是否真含平台区数字 (主表出口不能是空壳) ---
    ss_p = d / "summary_stats.json"
    if ss_p.exists():
        ss = load_json(ss_p)
        plateau = (ss or {}).get("plateau") or {}
        got = [k for k in ("cd", "hd", "nuc")
               if (plateau.get(k) or {}).get("plateau_mean") is not None]
        meta["plateau_metrics"] = got
        if len(got) < 3:
            issues.append("summary_stats.json 平台区只有 %s, 缺 %s"
                          % (got or "无",
                             [k for k in ("cd", "hd", "nuc") if k not in got]))

    return {"ok": not issues, "issues": issues, "warns": warns, "meta": meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-only", action="store_true",
                    help="只审计 PAPER_RUNS 名单里的 run")
    ap.add_argument("--json", action="store_true", help="输出 JSON 报告")
    a = ap.parse_args()

    if not RUNS.exists():
        print("★ runs/ 不存在")
        return 1

    names = sorted(p.name for p in RUNS.iterdir() if p.is_dir())
    if a.paper_only:
        names = [n for n in names if n in PAPER_RUNS]
    # 只审计真的有 history.json 的目录 (排除 probe / design 等非训练产物)
    names = [n for n in names if (RUNS / n / "history.json").exists()]

    results = {n: audit_run(RUNS / n) for n in names}

    if a.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0 if all(r["ok"] for r in results.values()) else 1

    print("=" * 78)
    print("实验存档完整性审计   (规范: docs/ARTIFACT_POLICY.md §3)")
    print("=" * 78)

    n_ok = 0
    for n in names:
        r = results[n]
        m = r["meta"]
        is_paper = n in PAPER_RUNS
        tag = "[论文]" if is_paper else "[辅助]"
        ep = ("%s/%s" % (m.get("epochs_actual", "?"), m.get("epochs_declared", "?")))
        gpu = (m.get("gpu_name") or "?").replace("NVIDIA GeForce ", "")
        run_mark = " ◀跑" if m.get("running") else ""
        head = ("%s %-26s ep=%-9s gpu=%-12s%s"
                % (tag, n, ep, gpu, run_mark))
        if r["ok"]:
            n_ok += 1
            print("  OK   %s" % head)
        else:
            print("  FAIL %s" % head)
        for it in r["issues"]:
            print("         ✗ %s" % it)
        for w in r["warns"]:
            print("         · %s" % w)

    # --- 跨 run 检查: 同一指标口径下的机器混排 (你定的红线) ---
    print("\n" + "=" * 78)
    print("跨机器混排检查 (红线: 同一张表的数字不得跨机并列)")
    print("=" * 78)
    by_gpu: dict[str, list[str]] = defaultdict(list)
    for n in names:
        if n in PAPER_RUNS:
            by_gpu[(results[n]["meta"].get("gpu_name") or "未知")].append(n)
    if len(by_gpu) > 1:
        print("  论文 run 分布在 %d 种 GPU 上 —— 必须分表报数:" % len(by_gpu))
        for g, ns in by_gpu.items():
            print("    %-28s : %s" % (g.replace("NVIDIA GeForce ", ""),
                                      ", ".join(ns)))
        print("  提示: 这不是错误, 但主表必须按 GPU 分组; 若发现同表并列即为违规。")
    elif by_gpu:
        g = next(iter(by_gpu))
        print("  全部论文 run 同 GPU (%s), 无混排风险。"
              % g.replace("NVIDIA GeForce ", ""))
    else:
        print("  尚无论文 run 落盘。")

    # --- 跨 run 检查: batch_size / seed 是否一致 (消融的可比性前提) ---
    print("\n" + "=" * 78)
    print("消融可比性检查 (batch_size / seed 必须一致, 否则单一变量失效)")
    print("=" * 78)
    bs = {n: results[n]["meta"].get("batch_size") for n in names if n in PAPER_RUNS}
    sd = {n: results[n]["meta"].get("seed") for n in names if n in PAPER_RUNS}
    for label, dd in (("batch_size", bs), ("seed", sd)):
        vals = set(v for v in dd.values() if v is not None)
        if len(vals) <= 1:
            print("  OK   %-11s 全部一致: %s" % (label, vals or "n/a"))
        else:
            print("  FAIL %-11s 出现多个值 %s —— 消融不可比!" % (label, vals))
            for n, v in dd.items():
                print("         %-26s = %s" % (n, v))

    print("\n" + "=" * 78)
    print("结果: %d/%d run 存档完整" % (n_ok, len(names)))
    print("=" * 78)
    bad = [n for n in names if not results[n]["ok"] and n in PAPER_RUNS]
    if bad:
        print("★ 论文 run 存在缺口, 必须修复: %s" % ", ".join(bad))
        return 1
    print("论文 run 全部通过。辅助/smoke run 的缺口不阻塞。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
