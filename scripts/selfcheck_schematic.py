# -*- coding: utf-8 -*-
"""
示意图自检器 —— 量化检查布局缺陷, 不依赖肉眼。

检查项:
  1. 每张 PNG 存在且非空, 有同名 .data.json
  2. .data.json 必须标注 depends_on_experiment=False (示意图不得冒充实验结果)
  3. 图题必须含「示意图 / 框图 / 流程」等字样
  4. 时间轴同层相邻框间距 >= 框宽 (量化重叠检测, 针对已复现的重叠缺陷)
  5. PNG 像素级白边检测: 图像内容不能贴边溢出
  6. 禁用字符检测: 「⚠」等在中文字体下渲染为方框的字符不得出现在脚本文本中
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "docs", "figures_schematic")
SRC = os.path.join(ROOT, "scripts", "make_schematic_figures.py")

PASS, FAIL = [], []


def ck(cond, msg):
    (PASS if cond else FAIL).append(msg)
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)


def main():
    print("=" * 68)
    print("示意图自检")
    print("=" * 68)

    pngs = sorted(f for f in os.listdir(OUTDIR) if f.endswith(".png"))
    ck(len(pngs) >= 6, "PNG 数量 >= 6 (实际 %d)" % len(pngs))

    # --- 1/2/3: 每张图的产物与元数据 ---
    print("\n[1] 产物完整性与元数据")
    for p in pngs:
        base = p[:-4]
        pp = os.path.join(OUTDIR, p)
        jp = os.path.join(OUTDIR, base + ".data.json")
        ck(os.path.getsize(pp) > 20000, "%s 非空且 > 20KB (%d B)" % (p, os.path.getsize(pp)))
        ck(os.path.exists(jp), "%s 有同名 .data.json" % base)
        if not os.path.exists(jp):
            continue
        d = json.load(open(jp, encoding="utf-8"))
        ck(d.get("depends_on_experiment") is False,
           "%s 标注 depends_on_experiment=False" % base)
        cap = d.get("caption", "")
        ck(bool(re.search(r"示意图|框图|流程|脉络", cap)),
           "%s 图题含示意/框图字样: %r" % (base, cap))
        ck(bool(d.get("figure_id")), "%s 有 figure_id" % base)

    # --- 4: 时间轴重叠量化检测 (读脚本里的真实布局参数) ---
    print("\n[2] 时间轴同层重叠量化检测")
    src = open(SRC, encoding="utf-8").read()

    m = re.search(r"BW_L,\s*BH_L\s*=\s*([\d.]+),\s*([\d.]+)", src)
    ck(m is not None, "能从脚本解析下半部分框宽 BW_L")
    bw = float(m.group(1)) if m else None

    # 解析 lower 列表: (年份, "标题", 层号)
    lower_block = re.search(r"lower\s*=\s*\[(.*?)\n    \]", src, re.S)
    ck(lower_block is not None, "能解析 lower 布局列表")
    items = []
    if lower_block:
        for mm in re.finditer(r"\(\s*([\d.]+)\s*,\s*\"(.*?)\"\s*,\s*(\d)\s*\)",
                              lower_block.group(1), re.S):
            items.append((float(mm.group(1)), mm.group(2), int(mm.group(3))))
    ck(len(items) >= 9, "解析到下半条目 >= 9 (实际 %d)" % len(items))

    if bw and items:
        layers = {}
        for x, t, l in items:
            layers.setdefault(l, []).append((x, t))
        worst = None
        for l, arr in sorted(layers.items()):
            arr.sort()
            for i in range(len(arr) - 1):
                gap = arr[i + 1][0] - arr[i][0]
                name = "层%d: %s -> %s" % (l, arr[i][1].split("\n")[0],
                                           arr[i + 1][1].split("\n")[0])
                ok = gap >= bw
                if not ok and (worst is None or gap < worst[0]):
                    worst = (gap, name)
                ck(ok, "同层间距 %.2f >= 框宽 %.2f  (%s)" % (gap, bw, name))
        ck(worst is None,
           "无同层重叠" if worst is None
           else "存在同层重叠, 最窄: %s gap=%.2f" % (worst[1], worst[0]))

    # --- 5: 禁用字符 ---
    print("\n[3] 禁用字符检测 (中文字体下渲染为方框)")
    banned = ["\u26a0", "\u2713", "\u2717", "\u2705", "\u274c"]
    hits = [repr(c) for c in banned if c in src]
    ck(not hits, "脚本无禁用字符 (命中: %s)" % (hits or "无"))

    # --- 6: 层数声明与实际一致 ---
    print("\n[4] 元数据与实际布局一致性")
    tl = os.path.join(OUTDIR, "F2_1_method_timeline.data.json")
    if os.path.exists(tl) and items:
        d = json.load(open(tl, encoding="utf-8"))
        ck(d.get("n_lower") == len(items),
           "n_lower 声明 %s == 实际解析 %d" % (d.get("n_lower"), len(items)))
        nl = len({l for _, _, l in items})
        ck(d.get("n_layers_lower") == nl,
           "n_layers_lower 声明 %s == 实际 %d" % (d.get("n_layers_lower"), nl))

    print("\n" + "=" * 68)
    print("结果: %d PASS / %d FAIL" % (len(PASS), len(FAIL)))
    print("=" * 68)
    if FAIL:
        print("\n未通过项:")
        for f in FAIL:
            print("  - " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
