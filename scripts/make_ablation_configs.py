# -*- coding: utf-8 -*-
"""生成 8 组消融配置 —— 从 B-002 派生，保证除目标项外逐字相同。

为什么用脚本生成而不是手写 8 个 yaml：
  手写必然出现"某组漏改一项"或"某组多改一项"的隐性差异，
  而这种差异会被完全归因到改进项本身，是消融实验最常见的致命错误。
  脚本生成 = 单一变量由代码保证，且 diff 可审计。

8 组设计（每组只动一处，D 组除外见下）：
  A1  改进A：双向CD加权（均衡化）      w_cd_fwd=1.0790 / w_cd_bwd=0.9210
  A2  改进A反向对照（强化bwd）         w_cd_fwd=0.9210 / w_cd_bwd=1.0790
  B1  改进B：固定权重对抗              w_adv>0, adaptive_adv=false
  B2  改进B：自适应对抗(M1)            w_adv>0, adaptive_adv=true
  C1  改进C：uniform 项                w_uniform>0
  D1  改进D：scale_qk                  model.scale_qk=true
  AC  组合：A1 + C1                    检验是否叠加
  BD  组合：B2 + D1                    检验是否叠加

为什么 A2（反向对照）必须跑：
  若只跑 A1 且它变好了，无法排除"任何权重扰动都能变好"（正则化式伪效应）。
  A2 是与 A1 关于 1.0 对称的反方向扰动。判读：
    A1 好 / A2 差  -> 方向性真实，改进 A 成立
    A1 好 / A2 也好 -> ★伪效应（扰动本身有益），改进 A 的机理解释不成立
    A1 差 / A2 好  -> 方向判断反了，应改为强化 bwd

为什么 B 分 B1/B2 两组：
  改进 B 的卖点是"自适应"，但若不与"固定权重对抗"对比，
  就无法区分收益来自"引入对抗"还是"自适应机制"。B1 是必要的中间对照。

输出：configs/abl_*.yaml + runs/ablation_design/ablation_matrix.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"
OUT = ROOT / "runs" / "ablation_design"

BASE = CFG / "b002_baseline150.yaml"

# 改进 A 的权重来自 scripts/calibrate_improve_a.py 实测定标（等和约束 w_f+w_b=2）
A_FWD, A_BWD = 1.078971, 0.921029

# 对抗权重取值依据：
#   B-001 实测 adaptive_adv_weight = 8.265（见 EXPERIMENT_LOG「M1 梯度比」）。
#   即"对抗梯度达到 CD 梯度 10% "所需的权重约 8.27。
#   B1 固定权重取该实测值，使 B1/B2 起点可比 —— 否则 B1 取一个随手的 1.0
#   会让对抗项梯度只有 CD 的 1/82.65，B1 等于几乎没开对抗，对照失去意义。
ADV_W_FIXED = 8.27
ADV_TARGET_RATIO = 0.1

# uniform 权重：由 scripts/calibrate_uniform.py 梯度范数比实测定标。
#   实测（CPU，4 batch 真实 PU1K patch，均值）：
#     未加权时 |g_uniform| / |g_CD| = 52.29  ← uniform 梯度天然比 CD 大 52 倍
#   令该比值 = 0.1（与 adv_target_ratio 一致，保持辅助项强度可比）
#     => w_uniform = 0.1 / 52.29 = 0.00191
#
# ⚠️ 历史教训：初版取 w_uniform=1.0（凭直觉），冒烟实测 cd=0.070 vs baseline
#    0.013，差 5.4 倍 —— uniform 梯度是 CD 的 52 倍，优化被均匀性完全主导。
#    若不定标就跑满 150 epoch，会得出"改进 C 导致精度崩溃"的错误结论，
#    而真相只是权重大了 523 倍。
#    定标方法用**梯度范数比**而非 loss 值比 —— 后者已被 B-001 证实会严重误判
#    （w_adv=1.0 时对抗梯度仅为 CD 的 1/82.65，等于没开对抗）。
UNIFORM_W = 0.00191

SPECS = {
    "A1_cd_balance": {
        "desc": "改进A 双向CD加权-均衡化(w_fwd>w_bwd)",
        "set": {"loss.w_cd_fwd": A_FWD, "loss.w_cd_bwd": A_BWD},
        "tests": "bwd 长期占优是否应被纠正",
    },
    "A2_cd_boost_bwd": {
        "desc": "改进A 反向对照-强化bwd(与A1对称)",
        "set": {"loss.w_cd_fwd": A_BWD, "loss.w_cd_bwd": A_FWD},
        "tests": "排除『任何权重扰动都变好』的伪效应",
    },
    "B1_adv_fixed": {
        "desc": "改进B 固定权重对抗(中间对照)",
        "set": {"loss.w_adv": ADV_W_FIXED, "loss.adaptive_adv": False},
        "tests": "引入对抗本身是否有收益",
    },
    "B2_adv_adaptive": {
        "desc": "改进B 梯度自适应对抗(M1)",
        "set": {"loss.w_adv": ADV_W_FIXED, "loss.adaptive_adv": True,
                "loss.adv_target_ratio": ADV_TARGET_RATIO},
        "tests": "自适应机制相对固定权重是否额外有收益",
    },
    "C1_uniform": {
        "desc": "改进C uniform 项",
        "set": {"loss.w_uniform": UNIFORM_W},
        "tests": "显式均匀性约束能否改善 NUC",
    },
    "D1_scale_qk": {
        "desc": "改进D SC-MSA 加 1/sqrt(d) 缩放",
        "set": {"model.scale_qk": True},
        "tests": "原文省略的注意力缩放是否影响稳定性/精度",
    },
    "AC_combo": {
        "desc": "组合 A1+C1",
        "set": {"loss.w_cd_fwd": A_FWD, "loss.w_cd_bwd": A_BWD,
                "loss.w_uniform": UNIFORM_W},
        "tests": "A 与 C 的收益是否可叠加",
    },
    "BD_combo": {
        "desc": "组合 B2+D1",
        "set": {"loss.w_adv": ADV_W_FIXED, "loss.adaptive_adv": True,
                "loss.adv_target_ratio": ADV_TARGET_RATIO,
                "model.scale_qk": True},
        "tests": "对抗与注意力缩放的收益是否可叠加",
    },
}

HEADER = """\
# =============================================================================
# 消融组 {name}
# {desc}
# =============================================================================
# ⚠️ 本文件由 scripts/make_ablation_configs.py 自动生成，请勿手改。
#    要改请改生成脚本后重新生成，否则「除目标项外与 baseline 逐字相同」
#    这一保证会失效。
#
# 派生自 : configs/b002_baseline150.yaml (B-002 baseline @150ep)
# 检验   : {tests}
# 改动项 :
{diff_lines}
#
# 显著性判据（跑前定死，禁止事后调整）—— 来自 runs/ablation_design/power_analysis.json
#   平台区均值相对 B-002 变化须超过：CD 0.66% / HD 2.18% / NUC 0.93%
#   （= 2 x 平台区均值标准误，n=75）。未超过则判「无显著差异」，
#   不得在论文中声称有改进。
#
# 接受准则（用户 2026-08-11 定案：「各个维度指标最好都要有提升，幅度不用大」）
#   ACCEPT_FULL  : CD/HD/NUC 三项改善均 > 门槛           -> 可作为主表改进项
#   ACCEPT_PART  : 部分项 > 门槛，其余项在 ±门槛内(持平)  -> 附录报告，须写明持平项
#   REJECT_TRADE : 任一项劣化 > 门槛                     -> 判 trade-off，不得声称改进
#   REJECT_NULL  : 三项均在 ±门槛内                      -> 判无效，如实报告
#   注：CD/HD 越小越好，NUC 越小越好；改善 = (base - exp) / base。
# =============================================================================
"""


def set_dotted(d: dict, key: str, val) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur[p]
    if parts[-1] not in cur:
        raise KeyError(f"目标键不存在于 baseline：{key} —— 防止静默新增无效项")
    cur[parts[-1]] = val


def main() -> int:
    base_txt = BASE.read_text(encoding="utf-8")
    base = yaml.safe_load(base_txt)
    print("=" * 72)
    print(f"从 {BASE.name} 派生 {len(SPECS)} 组消融配置")
    print("=" * 72)

    matrix = {}
    for name, spec in SPECS.items():
        cfg = yaml.safe_load(base_txt)          # 每组独立深拷贝
        diffs = []
        noop = []
        for k, v in spec["set"].items():
            old = base
            for p in k.split("."):
                old = old[p]
            set_dotted(cfg, k, v)
            if old == v:
                # 空改动：值与 baseline 相同。保留写入（使配置自解释），
                # 但不计入"改动项"，否则会虚报单一变量数、误导消融解读。
                noop.append((k, v))
            else:
                diffs.append((k, old, v))
        cfg["out_dir"] = f"runs/ABL_{name}"

        # 校验：除声明项 + out_dir 外，不得有任何其他差异
        flat_base, flat_new = {}, {}

        def flat(d, prefix, into):
            for k, v in d.items():
                if isinstance(v, dict):
                    flat(v, f"{prefix}{k}.", into)
                else:
                    into[f"{prefix}{k}"] = v
        flat(base, "", flat_base)
        flat(cfg, "", flat_new)
        allowed = set(spec["set"].keys()) | {"out_dir"}
        unexpected = {k for k in set(flat_base) | set(flat_new)
                      if flat_base.get(k) != flat_new.get(k)} - allowed
        if unexpected:
            print(f"[FAIL] {name} 出现未声明的差异：{unexpected}")
            return 1

        diff_lines = "\n".join(
            f"#   {k}: {o!r} -> {v!r}" for k, o, v in diffs)
        if noop:
            diff_lines += "\n#   （以下显式声明但与 baseline 同值，非改动项）\n"
            diff_lines += "\n".join(f"#   {k} = {v!r}" for k, v in noop)
        head = HEADER.format(name=name, desc=spec["desc"],
                             tests=spec["tests"], diff_lines=diff_lines)
        body = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False,
                              default_flow_style=False)
        dst = CFG / f"abl_{name}.yaml"
        dst.write_text(head + "\n" + body, encoding="utf-8")

        matrix[name] = {"config": dst.name, "out_dir": cfg["out_dir"],
                        "desc": spec["desc"], "tests": spec["tests"],
                        "n_real_changes": len(diffs),
                        "changes": {k: {"from": o, "to": v}
                                    for k, o, v in diffs},
                        "explicit_but_unchanged": {k: v for k, v in noop}}
        print(f"[OK] {dst.name:<28} 真实改动 {len(diffs)} 项"
              + (f" (+{len(noop)} 项显式同值)" if noop else "")
              + f"  {spec['desc']}")
        for k, o, v in diffs:
            print(f"       {k}: {o} -> {v}")
        for k, v in noop:
            print(f"       (同值) {k} = {v}")

    # 需要判别器的组必须确认 D 会被建起来
    need_d = [n for n, m in matrix.items()
              if any("w_adv" in k for k in m["changes"])]
    print(f"\n[提示] 需建判别器的组（+255,426 参数、显存与耗时上升）: {need_d}")

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "ablation_matrix.json"
    dst.write_text(json.dumps(
        {"base": BASE.name, "n_groups": len(matrix),
         "improve_a_weights": {"w_cd_fwd": A_FWD, "w_cd_bwd": A_BWD,
                               "source": "scripts/calibrate_improve_a.py"},
         "adv_w_fixed": ADV_W_FIXED,
         "adv_w_fixed_source": ("B-001 实测 adaptive_adv_weight=8.265，"
                                "使 B1/B2 起点可比"),
         "uniform_w": UNIFORM_W,
         "uniform_w_source": ("scripts/calibrate_uniform.py 梯度范数比实测定标："
                              "未加权 |g_uni|/|g_CD| = 52.29，取目标比 0.1 "
                              "=> w = 0.00191。初版凭直觉取 1.0 已被冒烟证伪"),
         "significance_thresholds_pct": {"cd": 0.66, "hd": 2.18, "nuc": 0.93},
         "acceptance_rule": {
             "decided_by": "user 2026-08-11",
             "user_words": "各个维度指标最好都要有提升，不需要很多，数据有提升就行",
             "improvement_formula": "(base_mean - exp_mean) / base_mean, 三项均越小越好",
             "verdicts": {
                 "ACCEPT_FULL": "CD/HD/NUC 三项改善均 > 门槛 -> 主表改进项",
                 "ACCEPT_PART": "部分项 > 门槛，其余在 ±门槛内 -> 附录，须写明持平项",
                 "REJECT_TRADE": "任一项劣化 > 门槛 -> 判 trade-off，不得声称改进",
                 "REJECT_NULL": "三项均在 ±门槛内 -> 判无效，如实报告"},
             "note": ("B-001 三项最优点分散在 ep41/ep48/ep98，说明 CD 与 NUC 存在"
                      "拉扯，三项同时超门槛非自动结果；C1(uniform) 针对 NUC、"
                      "A1(双向CD均衡) 针对 CD/HD，故 AC_combo 是最可能 ACCEPT_FULL 的组")},
         "groups": matrix}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[存档] {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
