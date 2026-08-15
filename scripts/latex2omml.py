# -*- coding: utf-8 -*-
"""LaTeX -> OMML（Office Math Markup Language）转换器。

为什么自写：latex2mathml 未安装且未获授权装包；全文独立公式仅 15 个、
行内 928 处，实测符号集只有 57 种命令，且**无矩阵、无多行、无 cases**
（scripts/_tmp/audit_formula_syntax.py 枚举确认），手写映射比引入依赖可控。

为什么必须做：用户意见第 1 条——此前把公式做纯文本降级，
`pii=₁N`、`p\pi(1)`、`\mathcalP` 这类**字符本身就是错的**，格式刷无法修正。
验收硬指标：导出 docx 的 document.xml 中反斜杠序列计数 = 0，
`<m:oMath>` 计数 >= 公式数（模板本身反斜杠为 0）。

支持的结构：
  · 上下标 _{} ^{} 与单字符形式，含同时带上下标（sSubSup）
  · 分式 \frac{}{}
  · 根号 \sqrt{}
  · 带限算子 \sum_{}^{}、\min_{}、\max_{}、\sup_{}、\inf_{}（nary / limLow）
  · 范数 \lVert \rVert -> ‖，绝对值 \lvert \rvert 与 \| -> |
  · 集合花括号 \{ \}
  · 字体命令 \mathrm \mathcal \mathbb \mathbf \text \mathit（取内容，
    \mathcal/\mathbb 映射到 Unicode 数学字母以保留语义）
  · 希腊字母与常用运算符 -> Unicode
  · 括号 \bigl \bigr \Bigl \Bigr \left \right -> 普通括号
  · 间距 \, \; \: \quad \qquad -> 空格
"""
import re

# OMML / WordprocessingML 命名空间
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

GREEK = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
    'epsilon': 'ε', 'varepsilon': 'ε', 'zeta': 'ζ', 'eta': 'η',
    'theta': 'θ', 'vartheta': 'ϑ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π',
    'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ',
    'phi': 'φ', 'varphi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
    'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ',
    'Xi': 'Ξ', 'Pi': 'Π', 'Sigma': 'Σ', 'Phi': 'Φ', 'Psi': 'Ψ',
    'Omega': 'Ω',
}

OPS = {
    'times': '×', 'cdot': '·', 'div': '÷', 'pm': '±', 'mp': '∓',
    'leq': '≤', 'le': '≤', 'geq': '≥', 'ge': '≥', 'neq': '≠',
    'approx': '≈', 'equiv': '≡', 'sim': '∼', 'propto': '∝',
    'll': '≪', 'gg': '≫', 'to': '→', 'rightarrow': '→',
    'leftarrow': '←', 'mapsto': '↦', 'triangleq': '≜',
    'in': '∈', 'notin': '∉', 'subset': '⊂', 'subseteq': '⊆',
    'supset': '⊃', 'cup': '∪', 'cap': '∩', 'emptyset': '∅',
    'infty': '∞', 'nabla': '∇', 'partial': '∂', 'forall': '∀',
    'exists': '∃', 'neg': '¬', 'land': '∧', 'lor': '∨',
    'dots': '…', 'ldots': '…', 'cdots': '⋯', 'bullet': '∙',
    'top': '⊤', 'bot': '⊥', 'langle': '⟨', 'rangle': '⟩',
    'lVert': '‖', 'rVert': '‖', 'lvert': '|', 'rvert': '|',
    'prime': '′', 'circ': '∘', 'ast': '∗',
}

# 花体（\mathcal）与黑板体（\mathbb）到 Unicode
CAL = {
    'A': '𝒜', 'B': 'ℬ', 'C': '𝒞', 'D': '𝒟', 'E': 'ℰ', 'F': 'ℱ',
    'G': '𝒢', 'H': 'ℋ', 'I': 'ℐ', 'J': '𝒥', 'K': '𝒦', 'L': 'ℒ',
    'M': 'ℳ', 'N': '𝒩', 'O': '𝒪', 'P': '𝒫', 'Q': '𝒬', 'R': 'ℛ',
    'S': '𝒮', 'T': '𝒯', 'U': '𝒰', 'V': '𝒱', 'W': '𝒲', 'X': '𝒳',
    'Y': '𝒴', 'Z': '𝒵',
}
BB = {
    'A': '𝔸', 'B': '𝔹', 'C': 'ℂ', 'D': '𝔻', 'E': '𝔼', 'F': '𝔽',
    'G': '𝔾', 'H': 'ℍ', 'I': '𝕀', 'J': '𝕁', 'K': '𝕂', 'L': '𝕃',
    'M': '𝕄', 'N': 'ℕ', 'O': '𝕆', 'P': 'ℙ', 'Q': 'ℚ', 'R': 'ℝ',
    'S': '𝕊', 'T': '𝕋', 'U': '𝕌', 'V': '𝕍', 'W': '𝕎', 'X': '𝕏',
    'Y': '𝕐', 'Z': 'ℤ',
}
NARY = {'sum': '∑', 'prod': '∏', 'int': '∫', 'bigcup': '⋃',
        'bigcap': '⋂'}
FUNCS = ('min', 'max', 'sup', 'inf', 'lim', 'log', 'exp', 'sin',
         'cos', 'tan', 'arg', 'det', 'dim', 'ker')


def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;'))


def run(text, italic=False):
    """一个 m:r 数学 run。斜体由 m:rPr/m:sty 控制。"""
    if text == '':
        return ''
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if not italic else ''
    return (f'<m:r>{sty}<w:rPr><w:rFonts w:ascii="Cambria Math" '
            f'w:hAnsi="Cambria Math"/></w:rPr>'
            f'<m:t xml:space="preserve">{esc(text)}</m:t></m:r>')


class Tok:
    """词法单元：('sym', 文本, 是否斜体) 或 ('grp', 子节点列表)。"""


def _split_group(s, i):
    """从 s[i]=='{' 起取平衡花括号内容，返回 (内容, 下一位置)。"""
    assert s[i] == '{'
    depth, j = 0, i
    while j < len(s):
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _arg(s, i):
    """取一个参数：{...} 或单字符/单命令。返回 (latex 子串, 下一位置)。"""
    while i < len(s) and s[i] == ' ':
        i += 1
    if i >= len(s):
        return '', i
    if s[i] == '{':
        return _split_group(s, i)
    m = re.match(r'\\[a-zA-Z]+', s[i:])
    if m:
        return m.group(0), i + m.end()
    return s[i], i + 1


def to_omml_inner(latex):
    """把 LaTeX 片段转为 OMML 子元素串（不含 m:oMath 外壳）。"""
    s = latex
    out = []
    buf = []          # 待输出的普通字符缓冲
    i = 0

    def flush(italic=True):
        if buf:
            out.append(run(''.join(buf), italic=italic))
            buf.clear()

    while i < len(s):
        c = s[i]

        # ---- 命令 ----
        if c == '\\':
            m = re.match(r'\\([a-zA-Z]+)', s[i:])
            if not m:
                # 转义符号 \{ \} \| \% \, \; 等
                nxt = s[i + 1] if i + 1 < len(s) else ''
                if nxt in '{}':
                    buf.append(nxt)
                    i += 2
                    continue
                if nxt == '|':
                    buf.append('|')
                    i += 2
                    continue
                if nxt in ',;: ':
                    buf.append(' ')
                    i += 2
                    continue
                if nxt == '%':
                    buf.append('%')
                    i += 2
                    continue
                i += 1
                continue
            cmd = m.group(1)
            i += m.end()

            if cmd in ('quad', 'qquad'):
                buf.append('  ')
                continue
            if cmd in ('left', 'right', 'bigl', 'bigr', 'Bigl', 'Bigr',
                       'Big', 'big', 'displaystyle', 'limits',
                       'nolimits', 'space'):
                continue
            if cmd in GREEK:
                buf.append(GREEK[cmd])
                continue
            if cmd in OPS:
                buf.append(OPS[cmd])
                continue
            if cmd == 'frac':
                flush()
                a, i = _arg(s, i)
                b, i = _arg(s, i)
                out.append(
                    '<m:f><m:fPr><m:ctrlPr/></m:fPr>'
                    f'<m:num>{to_omml_inner(a)}</m:num>'
                    f'<m:den>{to_omml_inner(b)}</m:den></m:f>')
                continue
            if cmd == 'sqrt':
                flush()
                a, i = _arg(s, i)
                out.append(
                    '<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr>'
                    f'<m:deg/><m:e>{to_omml_inner(a)}</m:e></m:rad>')
                continue
            if cmd in ('mathrm', 'text', 'operatorname', 'mathsf'):
                # 直立体内容并入缓冲，**不 flush**：否则会切断
                # `w_{\mathrm{cd}}` 这类「基底 + 下标」的相邻关系，
                # 使随后的 `_`/`^` 找不到正确基底（实测症状见上下标分支注释）。
                a, i = _arg(s, i)
                buf.append(_plain(a))
                continue
            if cmd == 'mathbf':
                a, i = _arg(s, i)
                flush()
                out.append(
                    '<m:r><m:rPr><m:sty m:val="b"/></m:rPr>'
                    '<w:rPr><w:rFonts w:ascii="Cambria Math" '
                    'w:hAnsi="Cambria Math"/></w:rPr>'
                    f'<m:t xml:space="preserve">{esc(_plain(a))}'
                    '</m:t></m:r>')
                continue
            if cmd == 'mathcal':
                a, i = _arg(s, i)
                buf.append(''.join(CAL.get(ch, ch) for ch in _plain(a)))
                continue
            if cmd == 'mathbb':
                a, i = _arg(s, i)
                buf.append(''.join(BB.get(ch, ch) for ch in _plain(a)))
                continue
            if cmd == 'mathit':
                a, i = _arg(s, i)
                buf.append(_plain(a))
                continue
            if cmd in ('hat', 'bar', 'tilde', 'vec', 'dot'):
                a, i = _arg(s, i)
                flush()
                chr_map = {'hat': '̂', 'bar': '̄', 'tilde': '̃',
                           'vec': '⃗', 'dot': '̇'}
                acc = {'hat': '0303', 'bar': '0304', 'tilde': '0303',
                       'vec': '20D7', 'dot': '0307'}[cmd]
                out.append(
                    f'<m:acc><m:accPr><m:chr m:val="&#x{acc};"/>'
                    '<m:ctrlPr/></m:accPr>'
                    f'<m:e>{to_omml_inner(a)}</m:e></m:acc>')
                continue
            if cmd in NARY:
                flush()
                sub = sup = ''
                while i < len(s) and s[i] in '_^':
                    k = s[i]
                    i += 1
                    a, i = _arg(s, i)
                    if k == '_':
                        sub = a
                    else:
                        sup = a
                # 取被作用体：到下一个同级运算符前的一段
                body, i = _nary_body(s, i)
                out.append(
                    '<m:nary><m:naryPr>'
                    f'<m:chr m:val="{NARY[cmd]}"/>'
                    '<m:limLoc m:val="undOvr"/><m:ctrlPr/></m:naryPr>'
                    f'<m:sub>{to_omml_inner(sub)}</m:sub>'
                    f'<m:sup>{to_omml_inner(sup)}</m:sup>'
                    f'<m:e>{to_omml_inner(body)}</m:e></m:nary>')
                continue
            if cmd in FUNCS:
                flush()
                sub = ''
                if i < len(s) and s[i] == '_':
                    i += 1
                    sub, i = _arg(s, i)
                if sub:
                    body, i = _nary_body(s, i)
                    out.append(
                        '<m:limLow><m:limLowPr><m:ctrlPr/></m:limLowPr>'
                        f'<m:e>{run(cmd, italic=False)}</m:e>'
                        f'<m:lim>{to_omml_inner(sub)}</m:lim></m:limLow>'
                        + to_omml_inner(body))
                else:
                    out.append(run(cmd, italic=False))
                continue
            # 未识别命令：退化为其名字（不留反斜杠）
            buf.append(cmd)
            continue

        # ---- 上下标 ----
        # 基底选取规则（首版在此出错，实测症状：`w_{\mathrm{cd}}` 渲染成
        # `cd`、`\nabla_\theta \mathcal{L}_{\mathrm{cd}}` 渲染成 `θ cd`）。
        # 原因：当缓冲区为空时直接 out.pop() 会把**整段前缀元素**当作基底
        # 弹走，而它可能是一个完整的 run（含多个字符）或结构元素。
        # 正确做法：
        #   · 缓冲区非空 -> 取其**最后一个字符**为基底，前缀先输出；
        #   · 缓冲区为空 -> 弹出 out 的最后一个元素为基底（它本身就是
        #     单个结构，如 \mathcal{L} 生成的 run 或 m:f）。
        if c in '_^':
            if buf:
                pre = ''.join(buf)
                buf.clear()
                if len(pre) > 1:
                    out.append(run(pre[:-1]))
                base = run(pre[-1])
            elif out:
                base = out.pop()
            else:
                base = run('')
            i += 1
            a1, i = _arg(s, i)
            k2 = None
            a2 = ''
            if i < len(s) and s[i] in '_^' and s[i] != c:
                k2 = s[i]
                i += 1
                a2, i = _arg(s, i)
            if k2:
                sub, sup = (a1, a2) if c == '_' else (a2, a1)
                out.append(
                    '<m:sSubSup><m:sSubSupPr><m:ctrlPr/></m:sSubSupPr>'
                    f'<m:e>{base}</m:e>'
                    f'<m:sub>{to_omml_inner(sub)}</m:sub>'
                    f'<m:sup>{to_omml_inner(sup)}</m:sup></m:sSubSup>')
            elif c == '_':
                out.append(
                    '<m:sSub><m:sSubPr><m:ctrlPr/></m:sSubPr>'
                    f'<m:e>{base}</m:e>'
                    f'<m:sub>{to_omml_inner(a1)}</m:sub></m:sSub>')
            else:
                out.append(
                    '<m:sSup><m:sSupPr><m:ctrlPr/></m:sSupPr>'
                    f'<m:e>{base}</m:e>'
                    f'<m:sup>{to_omml_inner(a1)}</m:sup></m:sSup>')
            continue

        # ---- 分组：透明处理 ----
        if c == '{':
            g, i = _split_group(s, i)
            flush()
            out.append(to_omml_inner(g))
            continue
        if c == '}':
            i += 1
            continue

        buf.append(c)
        i += 1

    flush()
    return ''.join(out)


def _nary_body(s, i):
    """取 \\sum / \\min 之后的被作用体（到同级 + - = , 或结尾为止）。"""
    while i < len(s) and s[i] == ' ':
        i += 1
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif depth == 0 and c in '+=,':
            break
        elif depth == 0 and c == '-' and j > i:
            break
        elif depth == 0 and c == '\\':
            m = re.match(r'\\([a-zA-Z]+)', s[j:])
            if m and m.group(1) in ('qquad', 'quad', 'tag'):
                break
        j += 1
    return s[i:j], j


def _plain(latex):
    """把 \\mathrm{} / \\text{} 的内容降为纯文本（其内不应再有结构）。

    须先处理转义符号：实测 `\\mathbf{-6.43\\%}` 会残留一个反斜杠
    （验收硬指标要求 document.xml 反斜杠计数为 0）。
    """
    t = latex
    for a, b in (('\\%', '%'), ('\\&', '&'), ('\\#', '#'),
                 ('\\_', '_'), ('\\$', '$'), ('\\{', '{'),
                 ('\\}', '}'), ('\\,', ' '), ('\\;', ' ')):
        t = t.replace(a, b)
    t = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', t)
    for k, v in {**GREEK, **OPS}.items():
        t = t.replace('\\' + k, v)
    t = re.sub(r'\\([a-zA-Z]+)', r'\1', t)
    return t.replace('{', '').replace('}', '').replace('\\', '')


def latex_to_omml(latex, display=False):
    """把一段 LaTeX 转为完整 m:oMath（display=True 时包 m:oMathPara）。"""
    latex = re.sub(r'\\tag\{[^}]*\}', '', latex).strip()
    inner = to_omml_inner(latex)
    omath = f'<m:oMath>{inner}</m:oMath>'
    if display:
        return ('<m:oMathPara><m:oMathParaPr>'
                '<m:jc m:val="center"/></m:oMathParaPr>'
                f'{omath}</m:oMathPara>')
    return omath


def has_backslash_residue(xml):
    """检查生成的 XML 是否残留反斜杠（验收硬指标）。"""
    return xml.count('\\')
