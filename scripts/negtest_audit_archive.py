# -*- coding: utf-8 -*-
"""audit_archive.py 的负例测试 —— 确认审计器不是"永远绿"。

为什么必须有这个
----------------
判据类脚本最危险的失效模式是**假绿**: 逻辑写错导致什么都检不出来,
而人看到一片 OK 就以为存档没问题。唯一的防线是**故意造缺陷, 看它抓不抓**。

做法: 在临时目录里复制一个真实通过的 run, 逐项破坏, 断言审计器必须 FAIL。
不动 runs/ 下的真实产物 (只读复制)。
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# 注意: audit_archive 顶层会把 sys.stdout 重新包装成 UTF-8 TextIOWrapper。
# 直接 import 会让本脚本原来的 stdout 被 GC 关闭 -> "I/O operation on closed file"。
# 故先 import, 再把 stdout 换成 audit_archive 装好的那个 (它同样是 UTF-8)。
import audit_archive as A  # noqa: E402
sys.stdout = A.sys.stdout

SRC = ROOT / "runs" / "ABL_A1_cd_balance"   # 已知完整通过的 run


def fresh(tmp: Path) -> Path:
    """复制一份干净 run 到 tmp, 返回其路径。"""
    dst = tmp / "CASE"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst)
    # 把 mtime 改老, 避免被判成"仍在跑"而豁免掉所有检查 ——
    # 这一步很关键: 不做的话负例会因为 running 豁免而假绿。
    import os
    old = 1_600_000_000  # 2020 年
    for p in dst.rglob("*"):
        if p.is_file():
            os.utime(p, (old, old))
    return dst


def expect_fail(name: str, d: Path, keyword: str) -> bool:
    r = A.audit_run(d)
    hit = any(keyword in i for i in r["issues"])
    ok = (not r["ok"]) and hit
    print("  %-38s %s" % (name, "PASS (已拦截)" if ok else "★ FAIL 未拦截!"))
    if not ok:
        print("       issues = %s" % r["issues"])
        print("       warns  = %s" % r["warns"])
    return ok


def main() -> int:
    if not SRC.exists():
        print("★ 基准 run 不存在: %s" % SRC)
        return 1

    print("=" * 74)
    print("audit_archive 负例测试 (基准: %s)" % SRC.name)
    print("=" * 74)

    results = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 先确认基准本身能过 (否则后面全部负例都没意义)
        base = fresh(tmp)
        rb = A.audit_run(base)
        base_ok = rb["ok"]
        print("  %-38s %s" % ("[前提] 基准 run 应通过",
                              "PASS" if base_ok else "★ FAIL"))
        if not base_ok:
            print("       issues = %s" % rb["issues"])
            return 1
        results.append(base_ok)

        # 负例 1: 删 summary_stats.json (主表报数出口)
        d = fresh(tmp)
        (d / "summary_stats.json").unlink()
        results.append(expect_fail("删 summary_stats.json", d, "summary_stats.json"))

        # 负例 2: 删一张必备图
        d = fresh(tmp)
        (d / "figures" / "F_loss.png").unlink()
        results.append(expect_fail("删 figures/F_loss.png", d, "F_loss.png"))

        # 负例 3: 图存在但删掉同名 .data.json (违反"图必须可溯源")
        d = fresh(tmp)
        (d / "figures" / "F_metric.data.json").unlink()
        results.append(expect_fail("删 F_metric.data.json", d, "图无数据源"))

        # 负例 4: 删一个 manifest 声明的点云 (定性图缺帧, 不可重建)
        d = fresh(tmp)
        man = json.loads((d / "clouds_manifest.json").read_text(encoding="utf-8"))
        victim = man[0]["samples"][0]["file"]
        (d / "clouds" / victim).unlink()
        results.append(expect_fail("删 clouds/%s" % victim, d, "clouds 缺"))

        # 负例 5: history 截断 (模拟训练中途崩)
        d = fresh(tmp)
        h = json.loads((d / "history.json").read_text(encoding="utf-8"))
        (d / "history.json").write_text(
            json.dumps(h[:50], ensure_ascii=False), encoding="utf-8")
        import os
        os.utime(d / "history.json", (1_600_000_000, 1_600_000_000))
        results.append(expect_fail("history 截断到 50 条", d, "训练未跑满"))

        # 负例 6: summary_stats 平台区置空 (空壳文件, 存在但没数字)
        d = fresh(tmp)
        ss = json.loads((d / "summary_stats.json").read_text(encoding="utf-8"))
        ss["plateau"] = {"cd": {}, "hd": {}, "nuc": {}}
        (d / "summary_stats.json").write_text(
            json.dumps(ss, ensure_ascii=False), encoding="utf-8")
        results.append(expect_fail("summary_stats 平台区置空", d, "平台区"))

        # 负例 7: 删 ckpt/best.pt
        d = fresh(tmp)
        (d / "ckpt" / "best.pt").unlink()
        results.append(expect_fail("删 ckpt/best.pt", d, "ckpt/best.pt"))

    print("=" * 74)
    n_pass = sum(1 for x in results if x)
    print("负例测试: %d/%d 通过" % (n_pass, len(results)))
    print("=" * 74)
    if n_pass != len(results):
        print("★ 审计器存在假绿风险, 必须修复后才能信任其 OK 结论。")
        return 1
    print("审计器能真实拦截各类存档缺陷, 其 OK 结论可信。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
