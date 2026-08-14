# -*- coding: utf-8 -*-
"""
全文数字回溯审计器 —— 针对「引用不存在的证据文件 + 编造派生数字」的直接防线。

事故背景：外部协作方产出的 docx 引用了两个本地并不存在的证据文件
（_split_audit_and_holdout_subset.json / _thesis_results_summary.json），
并据此给出了一整章无法回溯的数字。人眼逐段核对成本极高，故固化为脚本。

三项检查：
  C1  证据文件存在性：正文中出现的每个 .json / .py / .png 路径必须真实存在
  C2  数字可回溯性：正文中每个「实测数值」必须能在权威汇总或落盘 json 中找到
  C3  跨机器红线：同一表格/段落内不得并列不同 GPU 组的绝对值

用法：
  python scripts/audit_thesis_numbers.py                # 审全部章节
  python scripts/audit_thesis_numbers.py ch4_method.md  # 审单章
退出码：0 全过 / 1 有 issue / 2 无法执行
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH_DIR = os.path.join(ROOT, 'docs', 'chapters')
RESULTS = os.path.join(ROOT, 'docs', '_thesis_results.json')

# 数字容差：正文可能保留较少小数位，按相对容差匹配
REL_TOL = 5e-3

# 权威数字池允许的 runs 子目录 —— 与 audit_archive.py 的 PAPER_RUNS 口径一致
# 的 9 个论文级 run，外加两个非训练类存档（指标标定、ρ 轨迹反解）。
# 严禁把 smoke / _pipeline_check / SEED_* / R001_local_smoke / B001_reproduce
# 放进来：它们不进论文，却会给纯数值匹配提供大量可撞靶子。
POOL_RUNS = {
    # 3090 组（uniform / structure）
    'B002_baseline150', 'ABL_A1_cd_balance', 'ABL_A2_cd_boost_bwd',
    'ABL_C1_uniform', 'ABL_AC_combo', 'ABL_D1_scale_qk',
    # 5090 组（adversarial）
    'B002_baseline150_5090', 'ABL_B1_adv_fixed', 'ABL_B2_adv_adaptive',
    # 非训练类存档
    'E000_metric_calibration', 'rho_trace_B2.json',
}
# 免检数字（章节号、图表号、公式号、年份、参数量等结构性数字）
SKIP_PATTERNS = [
    r'^\d{1,2}$',            # 小整数：章节号/项目编号
    r'^\d{4}$',              # 年份
    r'^1[.,]?152[,.]?803$',  # 网络参数量（已由 audit_archive 核验）
    r'^150$', r'^200$', r'^100$',  # epoch 数 / 样本数 / 百分比基数
]


def load_authoritative():
    """收集权威数字池：来自 _thesis_results.json 与各 run 落盘 json。"""
    pool = {}   # value -> [来源描述]

    def add(v, src):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            pool.setdefault(round(float(v), 12), []).append(src)

    def walk(o, path):
        # per_sample[] 是逐样本明细（9 run × 200 样本 × 11 字段 ≈ 2 万个值）。
        # 正文只引用其汇总量，从不直接引用单样本值；把它们纳入池会把池撑到
        # 4.7 万个靶子，使「数字在池里」退化为几乎必然命中的弱证据
        # （实测：量级被篡改的 5.469e-2 撞上 per_sample[45].nn_gt_cv 而假绿）。
        if isinstance(o, list) and path.endswith('per_sample'):
            return
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f'{path}.{k}')
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f'{path}[{i}]')
        else:
            add(o, path)

    if os.path.exists(RESULTS):
        walk(json.load(open(RESULTS, encoding='utf-8')), '_thesis_results')
    # runs 下的 json 需递归：判据产物散落在 runs/<run>/ 与 runs/<run>/ckpt/ 等子目录。
    # 顶层 glob 会漏掉 runs/E000_metric_calibration/result.json 这类标定存档。
    #
    # 但池必须限定在「论文口径 run + 标定存档」内：smoke / pipeline_check /
    # SEED_* 等非论文 run 会把池从数百膨胀到数千，凭空提供大量可撞靶子，
    # 使「数字在池里」退化为弱证据（实测：篡改量级后的 5.469e-2 撞上
    # runs/smoke/history.json 的某个中间 epoch 值而假绿）。
    for p in glob.glob(os.path.join(ROOT, 'runs', '**', '*.json'),
                       recursive=True):
        rel = os.path.relpath(p, os.path.join(ROOT, 'runs')) \
            .replace(os.sep, '/')
        top = rel.split('/')[0]
        if top not in POOL_RUNS:
            continue
        try:
            walk(json.load(open(p, encoding='utf-8')), 'runs/' + rel)
        except (json.JSONDecodeError, OSError):
            continue
    for name in ('_cv_nn_measure.json', '_ch3_diag.json', '_ch3_stats.json',
                 '_ch3_diag_SMOKE.json', '_ch3_shapes.json',
                 '_arch_probe.json', '_promise_audit.json',
                 '_paired_improvement_B2_vs_B1.json',
                 # §6.5.3 定性小节的汇总层证据。为什么必须单独一个文件：
                 # 该节引用的「67133 劣化量」「最严重变差样本 66013」等量
                 # 源自 per_sample，而 walk() 对 per_sample 短路排除，
                 # 故这些量无法从 _cv_nn_measure.json 进入池。
                 # 由 scripts/build_qualitative_evidence.py 生成。
                 '_qualitative_panel_67133.json'):
        p = os.path.join(ROOT, 'docs', name)
        if os.path.exists(p):
            walk(json.load(open(p, encoding='utf-8')), 'docs/' + name)
    # 插图 .meta.json 属正当存档来源：图注与正文引用的逐 panel 统计量
    # （如图 5-1 的逐样本 cv_nn、图 5-2 的逐 epoch 尖峰统计）只落在这里，
    # 不在 _thesis_results.json 的汇总层。不纳入池会把正确引用误报为 C2 假红。
    # 仅收 paper_assets_TRIAL 下的图 meta，不收任意 json。
    for p in glob.glob(os.path.join(ROOT, 'paper_assets_TRIAL',
                                    '**', '*.meta.json'), recursive=True):
        rel = os.path.relpath(p, ROOT).replace(os.sep, '/')
        try:
            walk(json.load(open(p, encoding='utf-8')), rel)
        except (json.JSONDecodeError, OSError):
            continue
    return pool


def sig_digits(raw):
    """从正文原始写法推断有效数字位数。

    定精度容差的依据：正文写 `0.239709`（6 位小数）时，作者主张的精度就是
    6 位；此时若仍用统一的 5e-3 相对容差，篡改成 `0.238709`（偏差 0.42%）
    会落在容差内、撞上**正确字段**而假绿 —— 实测确认过这一漏洞。
    反之正文写 `-6.43%` 只主张 3 位有效数字，必须容忍池中的 -6.4331...。
    故容差必须跟随正文自身声明的精度，而非全局常量。
    """
    s = str(raw).strip().lstrip('+-')
    s = s.replace(',', '')
    if 'e' in s.lower():                      # 科学计数法取尾数部分
        s = re.split(r'[eE]', s)[0]
    if '.' in s:
        intpart, frac = s.split('.', 1)
        intpart = intpart.lstrip('0')
        n = len(intpart) + len(frac) if intpart else len(frac.lstrip('0'))
    else:
        n = len(s.strip('0')) or 1
    return max(n, 1)


MEANINGFUL_KW = ('pct', 'rel', 'ratio', 'share', 'cv', 'n_se',
                 'mean', 'median', 'std', 'se_sample', 'span',
                 'spearman', 'nuc', 'w_auto', 'rho')


def _meaningful_srcs(srcs):
    """筛出「看起来是真实测量量」的来源，剔除纯 epoch / 数组下标命中。

    池中含 0..200 几乎全部整数（实测 158 个），绝大多数是 epoch 号与数组
    下标。百分比通道若允许命中它们，任何两位小数百分比只要整数部分落在
    0..200 就能白拿一个"命中"——实测 `79.00%` 撞上
    `rho_trace_B2.rows[79].epoch`、`53.00%` 撞上 `rows[53].epoch` 而假绿，
    等于该百分比从未被真正核对。

    判定按字段语义白名单（去掉 [下标] 后取末段），而非黑名单匹配 `.epoch`
    尾部——后者会被 `F_metric.data.json.epochs[79]` 这类路径绕过。
    """
    keep = []
    for s in srcs:
        base = re.sub(r'\[\d+\]', '', s.lower())
        tail = base.rsplit('.', 2)[-1]
        if any(k in tail for k in MEANINGFUL_KW):
            keep.append(s)
    return keep


def match_pool(val, pool, raw=None, allow_abs=True, reject_index=False):
    """在权威池中寻找容差内的匹配。

    容差由 raw 的有效数字位数决定：n 位有效数字 → 相对容差 10^-n * 5。
    未提供 raw 时退回全局 REL_TOL（保持旧调用点行为）。
    allow_abs=False 时禁用异号(绝对值)兜底匹配，供百分比通道使用。
    reject_index=True 时只接受语义字段命中，拒绝纯 epoch/下标命中。
    """
    if val == 0:
        return pool.get(0.0)
    tol = REL_TOL
    if raw is not None:
        tol = min(REL_TOL, 5.0 * 10 ** (-sig_digits(raw)))
    # 两轮匹配：先要求同号命中，只有全池无同号命中时才退让到绝对值命中。
    # 理由：绝对值分支本意是容忍正文写 -6.43% 而池中存 -6.4312...，
    # 但它同时会让「-8.43% 撞上某 run 的 n_se=8.468」这类**语义无关**的
    # 异号数字假绿（实测确认）。百分比与标准误倍数不是同一种量，
    # 不能仅因数位相近就互认。
    fallback = None
    for v, srcs in pool.items():
        if v == 0:
            continue
        use = srcs
        if reject_index:
            use = _meaningful_srcs(srcs)
            if not use:
                continue
        rel = abs(val - v) / max(abs(v), 1e-30)
        if rel <= tol and (val > 0) == (v > 0):
            return use
        if allow_abs and fallback is None and \
                abs(abs(val) - abs(v)) / max(abs(v), 1e-30) <= tol:
            fallback = use
    return fallback


def match_derived(val, pool_vals):
    """
    识别「由池内两个数派生」的合法量：比值 a/b 与百分比 a/b*100。

    正文中大量数字是派生量（如参数量比值 1152803/969900=1.1886、
    判别器占比 255426/1152803=22.16%）。若不识别，审计器会把合法派生
    报成硬失败，产生假红并淹没真问题。
    """
    if val == 0:
        return None
    for a in pool_vals:
        if a == 0:
            continue
        for b in pool_vals:
            if b == 0 or a is b:
                continue
            r = a / b
            if abs(val - r) / max(abs(r), 1e-30) <= REL_TOL:
                return ('ratio', a, b)
            r100 = r * 100.0
            if abs(val - r100) / max(abs(r100), 1e-30) <= REL_TOL:
                return ('pct', a, b)
    return None


def audit_file(path, pool):
    text = open(path, encoding='utf-8').read()
    name = os.path.basename(path)
    issues, warns, ok = [], [], 0

    # ---- C1 证据文件存在性 ----
    # 只检查「看起来是路径」的引用（含 / 分隔符）。裸文件名如 `pu_gan.py`
    # 在正文中作为模块简称使用，不构成路径承诺，不予检查。
    # 另设白名单：尚未生成的规划产物目录，登记为 warn 而非 issue。
    PLANNED_PREFIX = ('figures_schematic/',)
    for m in re.finditer(r'`([^`]+\.(?:json|py|png|yaml|yml|pt))`', text):
        rel = m.group(1).strip()
        if rel.startswith(('http', '<', '{')) or '*' in rel or '<' in rel:
            continue
        if '/' not in rel:
            continue                      # 裸文件名 = 正文简称，不检查
        cand = os.path.join(ROOT, rel.replace('/', os.sep))
        if os.path.exists(cand):
            ok += 1
        elif rel.startswith(PLANNED_PREFIX):
            warns.append(f'[C1-planned] 规划中尚未生成: {rel}')
        else:
            issues.append(f'[C1] 引用的文件不存在: {rel}')

    # ---- C2 数字可回溯性 ----
    # 只审「看起来是实测量」的数字：含小数点且 >=4 位有效数字，或带 % 的两位小数
    # 科学计数法陷阱：正文写 `$5.469 \times 10^{-4}$` 时，若只取尾数 5.469
    # 去查池，既会把正确数字报成假红，更会让尾数碰巧撞上池中同名数值而假绿。
    # 故必须先把紧跟其后的 \times 10^{-N} / e-N 还原成真实量级再回查。
    SCI_SUFFIX = re.compile(
        r'\s*(?:\\times|×)\s*10\^\{?\s*(-?\d+)\s*\}?|\s*[eE]\s*(-?\d+)')
    for m in re.finditer(r'(?<![\w.])(-?\d+\.\d{3,})(?![\w])', text):
        raw = m.group(1)
        val = float(raw)
        if any(re.match(p, raw) for p in SKIP_PATTERNS):
            continue
        # 向后看指数后缀（允许中间夹一个 $ 或空格）
        tail = text[m.end():m.end() + 24].lstrip('$ ')
        sm = SCI_SUFFIX.match(tail)
        shown = raw
        if sm:
            expo = int(sm.group(1) if sm.group(1) is not None else sm.group(2))
            val = val * (10.0 ** expo)
            shown = f'{raw}e{expo}'
        srcs = match_pool(val, pool, raw=raw)
        if srcs is None:
            ctx = text[max(0, m.start() - 45):m.end() + 25].replace('\n', ' ')
            issues.append(f'[C2] 数字无法回溯: {shown}   ...{ctx}...')
        else:
            ok += 1

    # 百分比通道必须同时接受裸 `%` 与 LaTeX 转义 `\%`。
    # 实测教训：ch5 全章百分比均写作 `$\mathbf{-6.43\%}$`，原正则 `\s*%`
    # 对 `\%` 一个都匹配不上 → 该通道在整章上静默失效、零命中，
    # 却因不产生 issue 而表现为「PASS」。这是典型的假绿（检查项没跑）。
    #
    # 分级依据（实测教训，篡改表 P3/P6/P7）：把主结果表里的
    # `-6.43%` 改成 `-8.43%`、`+148.42%` 改成 `+48.42%`，此通道**确实
    # 报了 warn**，但因本章另有 1 条合法派生量 warn，篡改后仍显示
    # `PASS warns=2` —— 篡改藏在 warn 堆里，与合法告警无法区分。
    # 故按位置分级：
    #   · 表格行内（`|` 分隔单元格）的百分比 = 主结果，必须能回溯 → issue
    #   · 正文散文中的百分比 = 可能是派生量（如两个存档值之差）→ warn
    for m in re.finditer(r'(-?\d+\.\d{1,2})\s*\\?%', text):
        raw = m.group(1)
        val = float(raw)
        srcs = match_pool(val, pool, raw=raw, allow_abs=False,
                          reject_index=True)
        if srcs is None:
            ctx = text[max(0, m.start() - 45):m.end() + 25].replace('\n', ' ')
            # 判定所处行是否为表格行
            ls = text.rfind('\n', 0, m.start()) + 1
            le = text.find('\n', m.end())
            line = text[ls:le if le != -1 else len(text)]
            in_table = line.lstrip().startswith('|') and line.count('|') >= 3
            # 表内百分比可能是**合法派生量**（如参数量占比 = 分量/总量），
            # 这类值不直接落在池中，但可由池内两数整除得出。
            # 实测教训：ch3 表内 4 个参数量占比（2.64/10.44/41.48/44.32）
            # 全部形如 `<分量>/1152803`，已逐条验算正确；若一律判 issue
            # 就是把正确的稿子判红（假红），会诱导「改稿迁就审计器」。
            # 故表内值先试派生匹配，仅在派生也匹配不上时才升级为 issue。
            derived = match_derived(val, pool) if in_table else None
            if in_table and derived is None:
                issues.append(
                    f'[C2%] 表内百分比无法回溯: {raw}%   ...{ctx}...')
            elif in_table:
                warns.append(
                    f'[C2%] 表内百分比为派生值 {raw}% = {derived}   '
                    f'...{ctx}...')
            else:
                warns.append(
                    f'[C2%] 百分比未在池中直接命中: {raw}%   ...{ctx}...')
        else:
            ok += 1

    # ---- C2b 配对计数一致性 ----
    # C2 通道只查「含 3 位以上小数」的数字，**裸整数计数不被覆盖**
    # （实测：把改善样本数 158 篡改成 168 无人拦阻）。
    #
    # 第一版 C2b 用 `str(imp) not in text` 做存在性判断，仍然拦不住篡改：
    #   ① 存在性 ≠ 一致性。把 158 改成 168 后，「158 未出现」才报错，
    #      而正文别处（如 `142`/`4.56%`/`148.42%`）本来就含 `42`，
    #      于是变差数篡改 42→32 时 `'42' in text` 依旧为真 → 静默放行。
    #   ② 它从不读正文里**实际写的那个数**，因此无法发现「写了个错的」。
    # 正确做法是**反向**做：从正文里把数字抽出来，逐个跟存档比。
    #
    # 抽取靶点限定在两种确定写法，避免加宽正则把章节号/epoch 卷进来：
    #   A. 表 5-4 的表格行：`| <指标> | <改善数> | <变差数> | <占比>% | ...`
    #   B. 行内括注：`（<改善数> 对 <变差数>）`
    # 两者都要求**同一处**同时出现两个计数，故一处篡改必然暴露。
    paired_p = os.path.join(ROOT, 'docs',
                            '_paired_improvement_B2_vs_B1.json')
    if os.path.exists(paired_p):
        try:
            pd = json.load(open(paired_p, encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            pd = None
        if pd:
            n_tot = pd['n_paired']
            # 正文指标名 → 存档 key。表格首列用 LaTeX，正文用中文/缩写。
            def _key_of(label):
                if 'cv' in label:
                    return 'cv_nn'
                if 'CD' in label:
                    return 'cd'
                if 'HD' in label:
                    return 'hd'
                return None

            # ---- A. 表格行通道 ----
            # 单元格可能带 `**加粗**`，故先剥星号再取整数。
            row_re = re.compile(
                r'^\|([^|\n]*?)\|\s*\**(\d{1,3})\**\s*\|'
                r'\s*\**(\d{1,3})\**\s*\|\s*\**(\d+\.\d{1,2})\**\s*\\?%',
                re.M)
            seen_rows = 0
            for m in row_re.finditer(text):
                label, s_imp, s_wor, s_pct = m.groups()
                mk = _key_of(label)
                if mk is None or mk not in pd['metrics']:
                    continue
                mv = pd['metrics'][mk]
                seen_rows += 1
                imp, wor = int(s_imp), int(s_wor)
                if imp != mv['n_improved']:
                    issues.append(
                        f'[C2b] 表行 {mk} 改善样本数 正文={imp} '
                        f'存档={mv["n_improved"]}')
                if wor != mv['n_worse']:
                    issues.append(
                        f'[C2b] 表行 {mk} 变差样本数 正文={wor} '
                        f'存档={mv["n_worse"]}')
                if abs(float(s_pct) - mv['pct_improved']) > 5e-3:
                    issues.append(
                        f'[C2b] 表行 {mk} 改善占比 正文={s_pct} '
                        f'存档={mv["pct_improved"]}')
                # 计数自洽：改善+变差+持平 必须等于配对样本总数
                if imp + wor != n_tot - mv['n_tie']:
                    issues.append(
                        f'[C2b] 表行 {mk} 计数不自洽: {imp}+{wor} != '
                        f'{n_tot - mv["n_tie"]}')
                # 占比自洽：占比必须由改善数算得，防「改数不改占比」
                if abs(imp / n_tot * 100.0 - float(s_pct)) > 5e-3:
                    issues.append(
                        f'[C2b] 表行 {mk} 占比与计数不自洽: {imp}/{n_tot} '
                        f'= {imp / n_tot * 100.0:.2f} vs 正文 {s_pct}')
                ok += 1

            # ---- B. 行内括注通道 `（106 对 94）` ----
            for m in re.finditer(r'（(\d{1,3})\s*对\s*(\d{1,3})）', text):
                imp, wor = int(m.group(1)), int(m.group(2))
                hit = [k for k, v in pd['metrics'].items()
                       if v['n_improved'] == imp and v['n_worse'] == wor]
                if not hit:
                    # 和的自洽性可用于区分「真错」与「非配对语境」
                    if imp + wor == n_tot:
                        issues.append(
                            f'[C2b] 括注计数对 ({imp}, {wor}) 与存档任一'
                            f'指标均不符')
                    continue
                ok += 1

            # 受检数量守卫：**有配对数据表却零命中**才是正则失效。
            # 防「正则失效 → 零命中 → 因不产生 issue 而假绿」。
            #
            # 触发条件必须是「有表」而不是「提到关键词」。实测（ch4/ch7）：
            # 原条件写作「正文含『改善占比』且 seen_rows == 0」，把两类
            # **本就不该有表**的合法写法判成假红：
            #   · ch4 §4.x 预先规定「配对占比是描述性事实、不设门槛」
            #     —— 这是方法论声明，摆数据表反而违反「跑前定死判据」；
            #   · ch7 结论章引述已在 ch6 核验过的 158/79.00%
            #     —— 结论章按体例不重复摆表。
            # 这与 C2c 早期同款错误同源：守卫条件应为「有表但没检完」，
            # 而非「提到就必须有表」。
            #
            # 「有表」的判定用表头列名，而不是复用 row_re——否则
            # row_re 一旦失效，判定同时失效，守卫等于自我豁免（假绿）。
            has_paired_table = re.search(
                r'^\|[^|\n]*\|\s*改善样本数\s*\|', text, re.M) is not None
            if has_paired_table and seen_rows == 0:
                issues.append(
                    '[C2b] 存在配对改善表（表头含「改善样本数」）'
                    '但表行通道零命中 → 疑似正则失效（假绿风险）')

    # ---- C2c 裁定完整性（SE 倍数 + 裁定词）----
    # 为什么必须单开一个通道：`8.48` / `0.32` / `0.35` 这三个数**承载了
    # 全文的显著性结论**，但它们是 2 位小数、不带 `%`，因此
    #   · C2 通道要求 3 位以上小数 → 不覆盖
    #   · C2% 通道要求 `%` 号     → 不覆盖
    # 实测（篡改表 P4/P5）：把 `8.48` 改成 `12.48`、把 CD 的 `0.32` 改成
    # `2.32`（即把「未达门槛」伪装成「过门槛」），**审计器全程零反应**。
    # 这是本审计器此前最危险的缺口：改动收益最大、被发现概率为零。
    #
    # 校验三件事，缺一不可：
    #   ① 倍数与存档 n_se 一致；
    #   ② 裁定词与存档 verdict 一致；
    #   ③ 裁定词与倍数**自洽**（按存档门槛重算），防「改倍数不改裁定」
    #      或「改裁定不改倍数」这类单侧篡改。
    tr_p = os.path.join(ROOT, 'docs', '_thesis_results.json')
    if os.path.exists(tr_p):
        try:
            tr = json.load(open(tr_p, encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            tr = None
        if tr and 'main_comparison' in tr:
            mult = tr['criteria']['accept_multiplier_se']
            mc = tr['main_comparison']['metrics']

            def _key_of2(label):
                if 'cv' in label:
                    return 'cv_nn'
                if 'CD' in label:
                    return 'cd'
                if 'HD' in label:
                    return 'hd'
                return None

            # 表 5-3 形态：| 指标 | … | <SE倍数> | <裁定> |
            # 倍数与裁定是最后两个单元格，故从行尾锚定。
            #
            # 单元格可能被三种装饰包裹，必须全部剥掉才能取到数字：
            #   `0.32` / `**0.32**` / `$\mathbf{8.48}$`
            # 实测教训（篡改表 P4 漏网）：只写 `\**` 只能剥星号，
            # cv_nn 行写作 `$\mathbf{8.48}$` → 该行整行不匹配，于是
            # **恰恰是唯一 ACCEPT 的那一行没被检查**，而 CD/HD 两行
            # （裸数字）正常命中 → 表面上通道「有命中」，实则漏掉了
            # 最关键的主结论行。这是「部分命中」伪装成「通道有效」。
            def _unwrap(cell):
                """剥掉 markdown 加粗与 LaTeX \\mathbf{} 装饰。"""
                s = cell.strip()
                s = re.sub(r'^\*\*(.*)\*\*$', r'\1', s.strip()).strip()
                s = re.sub(r'^\$(.*)\$$', r'\1', s).strip()
                s = re.sub(r'^\\mathbf\{(.*)\}$', r'\1', s).strip()
                s = re.sub(r'^\*\*(.*)\*\*$', r'\1', s).strip()
                return s

            seen_verdict = 0
            for line in text.splitlines():
                ls = line.strip()
                if not ls.startswith('|') or ls.count('|') < 4:
                    continue
                cells = [c for c in ls.strip('|').split('|')]
                if len(cells) < 3:
                    continue
                ver = _unwrap(cells[-1])
                if ver not in ('ACCEPT', 'REJECT_NULL',
                               'REVERSE_SIGNIFICANT'):
                    continue
                s_mul = _unwrap(cells[-2])
                if not re.fullmatch(r'\d+\.\d{1,2}', s_mul):
                    issues.append(
                        f'[C2c] 裁定行 SE 倍数单元格无法解析: {s_mul!r}'
                        f'（裁定 {ver}）')
                    continue
                mk = _key_of2(cells[0])
                if mk is None or mk not in mc:
                    continue
                seen_verdict += 1
                got = float(s_mul)
                exp = mc[mk]['n_se']
                if abs(got - exp) > 5e-3 * max(abs(exp), 1.0):
                    issues.append(
                        f'[C2c] {mk} SE 倍数 正文={s_mul} '
                        f'存档={exp:.4f}')
                if ver != mc[mk]['verdict']:
                    issues.append(
                        f'[C2c] {mk} 裁定 正文={ver} '
                        f'存档={mc[mk]["verdict"]}')
                # 裁定与倍数自洽：|Δ|/SE >= 门槛 才可能是显著结论
                implied = 'ACCEPT' if got >= mult else 'REJECT_NULL'
                if implied != ver and ver != 'REVERSE_SIGNIFICANT':
                    issues.append(
                        f'[C2c] {mk} 裁定与倍数不自洽: 正文 {s_mul} SE '
                        f'（门槛 {mult}）应为 {implied}，正文写 {ver}')
                ok += 1

            # 受检数量守卫：裁定表有 3 个指标，命中数必须等于表内裁定行数。
            # 只要求「>0」不够——实测漏掉主结论行时仍有 2 个命中。
            n_verdict_rows = sum(
                1 for ln in text.splitlines()
                if ln.strip().startswith('|')
                and _unwrap(ln.strip().strip('|').split('|')[-1])
                in ('ACCEPT', 'REJECT_NULL', 'REVERSE_SIGNIFICANT'))
            if n_verdict_rows != seen_verdict:
                issues.append(
                    f'[C2c] 裁定行受检数不符: 表内 {n_verdict_rows} 行，'
                    f'实际校验 {seen_verdict} 行 → 有行未被检查（假绿风险）')
            # 注意：不能因「正文出现 REJECT_NULL 却没有裁定表」就报错。
            # 实测假红：ch3/ch4 在散文里定义判据（「$|\Delta| < 2\,SE$ 判
            # REJECT_NULL」）本就没有裁定表，属正常写法。守卫只应检查
            # 「有表但没检完」，不应要求「提到裁定词就必须有表」。

    # ---- C3 跨机器红线 ----
    # 红线判定只依据 run 名归属，不依据文中是否出现 "3090"/"5090" 字样：
    # 越线的典型形态正是「表格只写 run 名，不标机器」。
    R3090 = ('B002_baseline150', 'ABL_A1_cd_balance', 'ABL_A2_cd_boost_bwd',
             'ABL_C1_uniform', 'ABL_AC_combo', 'ABL_D1_scale_qk')
    R5090 = ('B002_baseline150_5090', 'ABL_B1_adv_fixed',
             'ABL_B2_adv_adaptive')

    def runs_in(line):
        """返回该行命中的 (3090组, 5090组) run 名集合。"""
        g5 = {k for k in R5090 if k in line}
        g3 = set()
        for k in R3090:
            if k not in line:
                continue
            # B002_baseline150 是 B002_baseline150_5090 的前缀，需排除误命中
            if k == 'B002_baseline150':
                stripped = line.replace('B002_baseline150_5090', '')
                if 'B002_baseline150' not in stripped:
                    continue
            g3.add(k)
        return g3, g5

    # 逐表格行检查
    for line in text.splitlines():
        if not line.strip().startswith('|'):
            continue
        g3, g5 = runs_in(line)
        if g3 and g5:
            warns.append('[C3] 同一表格行跨机器并列: %s | 3090组=%s 5090组=%s'
                         % (line.strip()[:80], sorted(g3), sorted(g5)))

    # 逐表格块检查（同一张表内混排两组，即使不在同一行）
    block, in_tbl = [], False
    for line in text.splitlines() + ['']:
        if line.strip().startswith('|'):
            block.append(line)
            in_tbl = True
            continue
        if in_tbl:
            b3, b5 = set(), set()
            for ln in block:
                a, b = runs_in(ln)
                b3 |= a
                b5 |= b
            if b3 and b5:
                warns.append('[C3] 同一表格块跨机器并列: 3090组=%s 5090组=%s'
                             % (sorted(b3), sorted(b5)))
            block, in_tbl = [], False

    return name, issues, warns, ok


def main():
    if not os.path.exists(RESULTS):
        print('!! 缺少权威汇总 docs/_thesis_results.json，'
              '先运行 scripts/build_thesis_results.py')
        return 2
    pool = load_authoritative()
    print('权威数字池: %d 个唯一数值' % len(pool))

    targets = sys.argv[1:]
    if targets:
        files = [os.path.join(CH_DIR, t) for t in targets]
    else:
        files = sorted(glob.glob(os.path.join(CH_DIR, 'ch*.md')))
        files = [f for f in files
                 if not f.endswith(('.bak', 'pre_outline_v2.md'))]

    total_i = total_w = total_ok = 0
    for f in files:
        if not os.path.exists(f):
            print('MISS', f)
            continue
        name, issues, warns, ok = audit_file(f, pool)
        total_i += len(issues)
        total_w += len(warns)
        total_ok += ok
        status = 'PASS' if not issues else 'FAIL'
        print()
        print('=' * 74)
        print('%-34s %s   ok=%d issues=%d warns=%d'
              % (name, status, ok, len(issues), len(warns)))
        print('=' * 74)
        for x in issues:
            print('  ' + x)
        for x in warns[:12]:
            print('  ' + x)
        if len(warns) > 12:
            print('  ... 另有 %d 条 warn' % (len(warns) - 12))

    print()
    print('总计: ok=%d issues=%d warns=%d' % (total_ok, total_i, total_w))
    return 0 if total_i == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
