# -*- coding: utf-8 -*-
"""把训练所需的代码与数据部署到 5090 远端, 然后起 B 组队列。

设计纪律:
  - 只传训练必需的东西: puvnet 包 + scripts/train_pu.py + configs/*.yaml + train h5。
    不传 runs/ (几 GB 产物)、不传 docs/、不传 .git/。
  - 大文件断点续传: 先比对远端大小, 一致则跳过 (重跑本脚本不会重复传 1 GB)。
  - 远端用 PUVNET_ROOT=/root/puv-net 覆盖数据根, 不改任何代码逻辑。
  - 每一步都验证结果, 不假设成功。

用法:
    python scripts/deploy_5090.py            # 只部署 + 验证, 不起训练
    python scripts/deploy_5090.py --launch   # 部署后起 B 组队列 (nohup 后台)
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import posixpath
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import paramiko  # noqa: E402
import yaml  # noqa: E402

from puvnet.cloud.compshare import CompShareClient  # noqa: E402

TARGET_ID = "cpod-1tq6i2ltk5mj"
REGION, ZONE = "cn-sh2", "cn-sh2-01"
SSH_HOST = "cpod-1tq6i2ltk5mj-s1.podtcp.compshare.cn"
SSH_PORT = 28870
SSH_USER = "root"

RWORK = "/root/puv-net"
RVENV = "/root/puvnet-venv/bin/python"
OUT_DIR = ROOT / "runs" / "probe_cpod"

# 训练必需的代码文件 (相对仓库根)
CODE_GLOBS = [
    "puvnet/**/*.py",
    "scripts/train_pu.py",
    "configs/*.yaml",
]
# 大数据文件: (本地相对路径, 远端相对路径)
DATA_FILES = [
    ("data/PU1K_extract/PU1K/train/"
     "pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5",
     "data/PU1K_extract/PU1K/train/"
     "pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5"),
]
# B 组队列: (远端 run 名, config 名)
B_QUEUE = [
    ("B002_baseline150_5090", "b002_baseline150"),
    ("ABL_B1_adv_fixed", "abl_B1_adv_fixed"),
    ("ABL_B2_adv_adaptive", "abl_B2_adv_adaptive"),
]

# SEED 队列: 兑现 3.5.5 预注册的 2SE 门槛（需跨 seed SE，非跨 epoch σ）。
# 三组 × 2 seed。config 由 scripts/make_seed_configs.py 从已完成 run 的
# config.yaml 原样复制 + 只改 seed/out_dir 生成（负例表 7/7 PASS）。
# 注意: 这些 run 全在 5090 上跑，其对照基准只能是 5090 的同 seed 组，
#       不得与 3090 的 ABL_C1_uniform / B002_baseline150 并表。
# 只放 baseline 与 B2：这两组的单 seed 原始数据在 5090 上产出，
# 跨 seed SE 必须同机取。C1 原生在 3090（cv_nn 主指标达标组），
# 其 seed 复现交给本机队列跑，同机可比且不占 5090 计费时间。
SEED_QUEUE = [
    ("SEED_baseline_s20260812", "SEED_baseline_s20260812"),
    ("SEED_B2_s20260812", "SEED_B2_s20260812"),
    ("SEED_baseline_s20260813", "SEED_baseline_s20260813"),
    ("SEED_B2_s20260813", "SEED_B2_s20260813"),
]
QUEUES = {"B": B_QUEUE, "SEED": SEED_QUEUE}


def get_password() -> str:
    cli = CompShareClient()
    cli.region, cli.zone = REGION, ZONE
    r = cli.call("DescribeCompShareInstance", Limit=20)
    if r.get("RetCode") != 0:
        raise RuntimeError("Describe RetCode=%s" % r.get("RetCode"))
    for h in r.get("UHostSet") or []:
        if h.get("UHostId") == TARGET_ID:
            if h.get("State") != "Running":
                raise RuntimeError("实例状态 %s, 非 Running" % h.get("State"))
            raw = h.get("Password") or ""
            try:
                return base64.b64decode(raw).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError):
                return raw
    raise RuntimeError("未找到实例")


def sh(cli, cmd, timeout=180, quiet=False):
    _i, o, e = cli.exec_command(cmd, timeout=timeout)
    so = o.read().decode("utf-8", errors="replace").rstrip()
    se = e.read().decode("utf-8", errors="replace").rstrip()
    rc = o.channel.recv_exit_status()
    if not quiet:
        if so:
            print(so[:1500])
        if se:
            print("  [stderr] %s" % se[:600])
    return so, se, rc


def rmkdir(sftp, rpath):
    parts = rpath.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def rsize(sftp, rpath):
    try:
        return sftp.stat(rpath).st_size
    except IOError:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="部署后起队列")
    ap.add_argument("--queue", default="B", choices=sorted(QUEUES),
                    help="要起哪个队列: B(默认) 或 SEED")
    a = ap.parse_args()

    pwd = get_password()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("连接 %s:%s ..." % (SSH_HOST, SSH_PORT))
    try:
        cli.connect(hostname=SSH_HOST, port=SSH_PORT, username=SSH_USER,
                    password=pwd, timeout=40, banner_timeout=40,
                    auth_timeout=40, look_for_keys=False, allow_agent=False)
    except Exception as exc:  # noqa: BLE001
        print("[FAIL] SSH: %s: %s" % (type(exc).__name__, exc))
        return 1
    finally:
        del pwd
    print("  已连接\n")

    sftp = cli.open_sftp()
    report = {}

    # ---- 1. 传代码 ----
    print("=" * 72)
    print("[1/4] 传代码")
    print("=" * 72)
    files = []
    for g in CODE_GLOBS:
        files.extend(sorted(ROOT.glob(g)))
    files = [f for f in files if f.is_file() and "__pycache__" not in str(f)]
    n_up = 0
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        rp = posixpath.join(RWORK, rel)
        rmkdir(sftp, posixpath.dirname(rp))
        lsz = f.stat().st_size
        if rsize(sftp, rp) == lsz:
            continue
        sftp.put(str(f), rp)
        n_up += 1
    print("  代码文件 %d 个, 本次上传 %d 个" % (len(files), n_up))
    report["code_files"] = len(files)
    report["code_uploaded"] = n_up

    # ---- 2. 传数据 (断点判断: 大小一致则跳过) ----
    print("\n" + "=" * 72)
    print("[2/4] 传数据")
    print("=" * 72)
    for lrel, rrel in DATA_FILES:
        lp = ROOT / lrel
        rp = posixpath.join(RWORK, rrel)
        lsz = lp.stat().st_size
        cur = rsize(sftp, rp)
        if cur == lsz:
            print("  已存在且大小一致, 跳过: %s (%.1f MB)"
                  % (rrel, lsz / 1e6))
            continue
        rmkdir(sftp, posixpath.dirname(rp))
        print("  上传 %s (%.1f MB), 远端现有 %s ..."
              % (rrel, lsz / 1e6, cur if cur >= 0 else "无"))
        t0 = time.time()
        last = [0.0]

        def cb(done, total, _t0=t0, _last=last):
            now = time.time()
            if now - _last[0] < 5:
                return
            _last[0] = now
            pct = 100.0 * done / total if total else 0
            mbps = done / 1e6 / max(now - _t0, 1e-6)
            print("    %.1f%%  %.0f/%.0f MB  %.1f MB/s"
                  % (pct, done / 1e6, total / 1e6, mbps), flush=True)

        sftp.put(str(lp), rp, callback=cb)
        got = rsize(sftp, rp)
        ok = got == lsz
        print("    完成 %.1f s, 远端 %d bytes, 大小一致=%s"
              % (time.time() - t0, got, ok))
        if not ok:
            print("  ★ 大小不一致, 中止")
            return 1

    # ---- 3. 远端校验 ----
    print("\n" + "=" * 72)
    print("[3/4] 远端校验")
    print("=" * 72)
    env = "PUVNET_ROOT=%s PYTHONPATH=%s PYTHONIOENCODING=utf-8 PYTHONUTF8=1" % (
        RWORK, RWORK)
    so, _, _ = sh(cli, "ls %s/configs/ | tr '\\n' ' '" % RWORK)
    so2, _, _ = sh(
        cli,
        "cd %s && %s %s -c \""
        "from puvnet.data.pu_dataset import ROOT, PU1K_TRAIN_H5;"
        "print('ROOT', ROOT);"
        "print('h5_exists', PU1K_TRAIN_H5.exists());"
        "import h5py;f=h5py.File(PU1K_TRAIN_H5,'r');"
        "print('keys', list(f.keys()));"
        "print('shape', f['poisson_256'].shape, f['poisson_1024'].shape)\""
        % (RWORK, env, RVENV))
    report["remote_check"] = so2
    if "h5_exists True" not in so2:
        print("★ 远端数据校验失败, 不起训练")
        return 1

    # ---- 4. 起队列 ----
    QID = a.queue                      # 队列标识, 贯穿 log 名与脚本名
    QUEUE = QUEUES[QID]
    QLOG = "queue_%s.log" % QID
    QSH = "run_%s_queue.sh" % QID
    print("\n" + "=" * 72)
    print("[4/4] %s 组队列  (%d 个 run)" % (QID, len(QUEUE)))
    print("=" * 72)
    report["queue"] = {"id": QID, "runs": [n for n, _ in QUEUE]}
    if not a.launch:
        print("  (未加 --launch, 只部署不起训练)")
        print("  将要跑: %s" % ", ".join(n for n, _ in QUEUE))
    else:
        # 远端 out_dir 必须显式覆盖: config 里的 out_dir 是本机名 (如
        # runs/B002_baseline150), 直接用会与本机 3090 产物同名, 回传时撞车,
        # 且违反"同一张表的数字不得跨机并列"这条红线。故远端统一加 _5090 后缀。
        #
        # 远端 loader 段也注入: 5090 是 14 核, 开 8 workers + pin_memory +
        # persistent, B002 当前 57.5 s/ep → 估 30-40 s/ep (待 B1 实测校准)。
        # 本机 yaml 不写这段, 是为了让 4 核的 3090 不被这条基础设施配置污染
        # 跨 run 行为; 同时生成脚本 make_ablation_configs.py 不会因为基线带
        # loader 段而把 8 个派生 yaml 全部污染 (它们只在本机 3090 上跑)。
        LOADER_5090 = {
            "num_workers": 8,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
        }
        for rname, cfgname in QUEUE:
            src = posixpath.join(RWORK, "configs", "%s.yaml" % cfgname)
            dst_name = "remote_%s.yaml" % rname
            dst = posixpath.join(RWORK, "configs", dst_name)
            so, _, rc = sh(cli, "cat %s" % src, quiet=True)
            if rc != 0 or not so.strip():
                print("  ★ 读不到 %s, 中止" % src)
                return 1
            cfg_obj = yaml.safe_load(so)
            cfg_obj["out_dir"] = "runs/%s" % rname
            cfg_obj["loader"] = dict(LOADER_5090)   # 注入, 覆盖任何残留
            with sftp.open(dst, "w") as fh:
                fh.write(yaml.safe_dump(cfg_obj, allow_unicode=True,
                                        sort_keys=False))
            print("  写远端 config %s  (out_dir=runs/%s, loader.num_workers=%d)"
                  % (dst_name, rname, LOADER_5090["num_workers"]))

        # 生成远端串行队列脚本: 逐个跑, 每个跑完写状态, 最后写 ALL_DONE
        lines = ["#!/bin/bash", "set -u", "cd %s" % RWORK,
                 "export %s" % env.replace(" ", " export ")]
        for rname, _cfg in QUEUE:
            lines += [
                "echo \"[$(date +%%H:%%M:%%S)] START %s\" >> %s/%s"
                % (rname, RWORK, QLOG),
                "%s scripts/train_pu.py --config configs/remote_%s.yaml "
                ">> %s/train_%s.log 2>&1" % (RVENV, rname, RWORK, rname),
                "echo \"[$(date +%%H:%%M:%%S)] END %s rc=$?\" >> %s/%s"
                % (rname, RWORK, QLOG),
            ]
        lines.append("echo ALL_DONE >> %s/%s" % (RWORK, QLOG))
        script = "\n".join(lines) + "\n"
        with sftp.open(posixpath.join(RWORK, QSH), "w") as fh:
            fh.write(script)
        sh(cli, "chmod +x %s/%s" % (RWORK, QSH))
        # 幂等: 用 pgrep 判断会**匹配到自己**(paramiko 把命令包成
        # `bash -c pgrep -f 'run_B_queue.sh'`, 该字符串本身含关键字),
        # 2026-08-11 连续两次误判为"已在跑"而不启动。
        # 改为: pgrep 结果剔除 bash -c 包装行, 只认真正的 train_pu.py 进程。
        so, _, _ = sh(cli, "pgrep -af 'train_pu.py' | grep -v 'bash -c' "
                           "| grep -v pgrep || true", quiet=True)
        if so.strip():
            print("  远端已有训练在跑:\n    %s" % so.replace("\n", "\n    "))
            print("  不重复启动。")
        else:
            # 启动后台任务时**不能读 stdout**: nohup ... & 之后通道不会关闭,
            # o.read() 会一直等到 timeout (2026-08-11 踩到 PipeTimeout)。
            # 故 fire-and-forget, 用后续独立查询确认结果。
            cli.exec_command(
                "cd %s && rm -f %s && nohup setsid bash "
                "%s > queue_%s_outer.log 2>&1 < /dev/null &"
                % (RWORK, QLOG, QSH, QID))
            print("  已下发启动命令, 等 30s 确认 ...")
            time.sleep(30)
            so, _, _ = sh(cli, "pgrep -af 'train_pu.py' | grep -v 'bash -c' "
                               "| grep -v pgrep || true")
            print("  启动后训练进程: %s" % (so.strip() or "★ 空, 启动失败"))
            if not so.strip():
                sh(cli, "echo '--- outer ---'; tail -20 %s/queue_%s_outer.log "
                        "2>/dev/null; echo '--- train ---'; tail -20 %s/train_%s.log "
                        "2>/dev/null" % (RWORK, QID, RWORK, QUEUE[0][0]))
                return 1
        sh(cli, "echo '--- %s ---'; cat %s/%s 2>/dev/null; "
                "echo '--- train log tail ---'; "
                "tail -6 %s/train_%s.log 2>/dev/null"
                % (QLOG, RWORK, QLOG, RWORK, QUEUE[0][0]))

    sftp.close()
    cli.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 按队列分文件, 否则 SEED 部署会覆盖 B 组的部署记录（存档不可互相踩）
    rp = OUT_DIR / ("deploy_5090.json" if a.queue == "B"
                    else "deploy_5090_%s.json" % a.queue)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print("\n[存档] %s" % rp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
