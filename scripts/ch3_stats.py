# -*- coding: utf-8 -*-
"""第 3 章及后续消融所需统计量：全部从 runs/<run> 真实产物计算。

用法:
    python scripts/ch3_stats.py                     # 默认 B002_baseline150 -> docs/_ch3_stats.json
    python scripts/ch3_stats.py ABL_A1_cd_balance   # -> docs/_stats_ABL_A1_cd_balance.json

同口径约束: 平台区定义、多窗口收敛判据、秩相关、选点比较、开销统计全部沿用同一实现,
不为单个 run 另写口径; 配置回显从 config.yaml 实读而非硬编码, 以免换 run 后误判。
"""
import json, io, sys, os, re
from statistics import mean, pstdev

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else 'B002_baseline150'
RUN = os.path.join(ROOT, 'runs', RUN_NAME)
assert os.path.isdir(RUN), 'run 目录不存在: %s' % RUN

hist = json.load(open(os.path.join(RUN, 'history.json'), encoding='utf-8'))
summ = json.load(open(os.path.join(RUN, 'summary_stats.json'), encoding='utf-8'))
cfg_txt = open(os.path.join(RUN, 'config.yaml'), encoding='utf-8').read()
env = json.load(open(os.path.join(RUN, 'env.json'), encoding='utf-8'))

out = {'source_run': 'runs/%s' % RUN_NAME, 'n_epochs': len(hist), 'env': env}

# --- 1. 训练/监控 bwd 占比 ---
for tag in ('train_cd_bwd_share', 'monitor_loss_cd_bwd_share'):
    v = [r[tag] for r in hist]
    out[tag] = {
        'min': min(v), 'max': max(v), 'mean': mean(v), 'std': pstdev(v),
        'min_epoch': v.index(min(v)), 'max_epoch': v.index(max(v)),
        'n_below_0.5': sum(1 for x in v if x < 0.5),
        'first': v[0], 'last': v[-1],
    }

# --- 2. 平台区（后 50%）三指标 ---
pl = summ['plateau']
out['plateau'] = {
    'epoch_range': pl['epoch_range'], 'plateau_n': pl['plateau_n'],
    'cd': {k: pl['cd'][k] for k in ('plateau_mean', 'plateau_std', 'best', 'best_epoch')},
    'hd': {k: pl['hd'][k] for k in ('plateau_mean', 'plateau_std', 'best', 'best_epoch')},
    'nuc': {k: pl['nuc'][k] for k in ('plateau_mean', 'plateau_std', 'best', 'best_epoch')},
}
for k in ('cd', 'hd', 'nuc'):
    m = pl[k]['plateau_mean']; s = pl[k]['plateau_std']
    out['plateau'][k]['cv_pct'] = 100.0 * s / m          # 相对波动
    out['plateau'][k]['best_gain_pct'] = 100.0 * (m - pl[k]['best']) / m

# --- 3. 多窗口收敛：CD 收敛而 HD/NUC 不收敛的窗口计数 ---
mw = summ['convergence_multi_window']
conv = {k: {} for k in ('cd', 'hd', 'nuc')}
for wname, w in mw.items():
    for k in ('cd', 'hd', 'nuc'):
        conv[k][wname] = {'change_pct': w[k]['change_pct'], 'converged': w[k]['converged']}
out['multi_window'] = conv
out['multi_window_summary'] = {
    k: {'n_windows': len(mw), 'n_converged': sum(1 for w in mw.values() if w[k]['converged'])}
    for k in ('cd', 'hd', 'nuc')
}

# --- 4. 指标趋势脱耦：平台区内 CD 与 HD/NUC 的秩相关 ---
def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0

lo, hi = pl['epoch_range']
seg = [r for r in hist if lo <= r['epoch'] <= hi]
cd = [r['monitor_cd'] for r in seg]
hd = [r['monitor_hd'] for r in seg]
nuc = [r['monitor_nuc'] for r in seg]
out['plateau_rank_corr'] = {
    'n': len(seg),
    'cd_vs_hd': spearman(cd, hd),
    'cd_vs_nuc': spearman(cd, nuc),
    'hd_vs_nuc': spearman(hd, nuc),
}

# --- 5. 选点分歧：加权选点 vs 仅看 CD ---
out['selection'] = {
    'weights': summ['selection']['weights'],
    'best_epoch_weighted': summ['selection']['best_epoch'],
    'best_epoch_cd_only': summ['shadow_cd_only_epoch'],
    'agree': summ['selection']['best_epoch'] == summ['shadow_cd_only_epoch'],
}

# --- 6. 训练开销 ---
sec = [r['sec'] for r in hist]
out['cost'] = {
    'sec_per_epoch_mean': mean(sec[1:]),          # 去掉首个 epoch 的预热
    'sec_per_epoch_first': sec[0],
    'total_hours': sum(sec) / 3600.0,
    'gpu_peak_gb': max(r['gpu_peak_gb'] for r in hist),
}

# --- 7. 关键配置回显（从 config.yaml 实读，供正文核对，防写错超参）---
def cfg_get(key, cast=str):
    m = re.search(r'^\s*%s\s*:\s*(\S+)' % re.escape(key), cfg_txt, re.M)
    if not m:
        return None
    raw = m.group(1).strip().strip('"\'')
    if cast is bool:
        return raw.lower() in ('true', 'yes', '1')
    try:
        return cast(raw)
    except ValueError:
        return raw

echo = {}
for key, cast in (('epochs', int), ('batch_size', int), ('lr', float), ('lr_step', int),
                  ('lr_decay', float), ('seed', int), ('up_ratio', int),
                  ('attention_type', str), ('tail_mode', str), ('w_cd', float),
                  ('w_adv', float), ('w_uniform', float), ('squared_cd', bool),
                  ('select_warmup', int), ('cd_bwd_weight', float),
                  ('adv_target_ratio', float)):
    v = cfg_get(key, cast)
    if v is not None:
        echo[key] = v

# 白名单必然漏掉"将来才出现的消融键"(实测: A1 的 w_cd_fwd/w_cd_bwd 被漏, 导致
# compare_runs.py 误报"配置未生效")。故对 loss: 段做全量兜底, 并在下方硬校验。
def _loss_block_keys():
    m = re.search(r'^loss:\s*$(.*?)(?=^\S|\Z)', cfg_txt, re.M | re.S)
    if not m:
        return {}
    got = {}
    for k, raw in re.findall(r'^\s+([A-Za-z_][\w]*)\s*:\s*(\S+)\s*$', m.group(1), re.M):
        raw = raw.strip().strip('"\'')
        if raw.lower() in ('true', 'false'):
            got[k] = raw.lower() == 'true'
        elif raw.lower() in ('null', 'none'):
            got[k] = None
        else:
            try:
                got[k] = float(raw) if ('.' in raw or 'e' in raw.lower()) else int(raw)
            except ValueError:
                got[k] = raw
    return got

_loss_keys = _loss_block_keys()
for k, v in _loss_keys.items():
    echo.setdefault(k, v)
echo['select_weights'] = summ['selection']['weights']
out['config_echo'] = echo

# 硬校验: 只断言"能从 config 读到且与 history 长度一致"这类跨 run 通用事实,
# 不断言 baseline 专属的 w_adv=0 —— 那会在消融 run 上误判。
_missing = [k for k in _loss_keys if k not in echo]
assert not _missing, 'config.yaml loss 段有键未进 config_echo: %s' % _missing
assert _loss_keys, 'config.yaml 未解析到 loss 段, 回显机制已失效'
assert echo.get('epochs') == len(hist), \
    'config epochs=%s 与 history 长度 %d 不一致' % (echo.get('epochs'), len(hist))
assert echo.get('seed') is not None, 'config.yaml 未记录随机种子, 结果不可复现'
out['is_clean_baseline'] = (echo.get('w_adv') == 0.0 and echo.get('w_uniform') == 0.0)
# 仅"损失全关"不足以判定干净 baseline: A1 也把 w_adv/w_uniform 关了, 但它带
# w_cd_fwd/w_cd_bwd 非对称权重。故任何非默认的消融键出现即取消该标记。
_ABL_KEYS = ('w_cd_fwd', 'w_cd_bwd', 'cd_bwd_weight')
out['ablation_keys'] = {k: echo[k] for k in _ABL_KEYS if k in echo}
if out['ablation_keys']:
    out['is_clean_baseline'] = False

if RUN_NAME == 'B002_baseline150':
    dst = os.path.join(ROOT, 'docs', '_ch3_stats.json')
else:
    dst = os.path.join(ROOT, 'docs', '_stats_%s.json' % RUN_NAME)
json.dump(out, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('written', dst)
print(json.dumps(out, ensure_ascii=False, indent=1))
