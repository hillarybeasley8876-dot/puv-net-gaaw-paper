# -*- coding: utf-8 -*-
"""
引用编号迁移器 —— 因相关性清理导致库编号平移后, 把已成稿章节里的 [N] 从旧编号
重映射到新编号。

安全设计:
  * 映射走 key: 旧number -> key -> 新number。绝不手改数字, 绝不猜。
  * 旧库里被移除的 key: 正文若引用了它 -> 报错停止, 人工决定怎么改写句子
  * 多编号连写 [12,13] / [12-14] 也处理
  * 先 dry-run 打印全部改动, 确认后再写盘 (--apply)
  * 写盘前备份 .bak

用法:
  python scripts/migrate_cite_numbers.py                 # dry-run
  python scripts/migrate_cite_numbers.py --apply         # 真写
"""
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
OLD = os.path.join(DOCS, "REFERENCES.before_relevance_purge.json")
NEW = os.path.join(DOCS, "REFERENCES.json")
CHAP_DIR = os.path.join(DOCS, "chapters")

APPLY = "--apply" in sys.argv


def load_map(path):
    d = json.load(open(path, encoding="utf-8"))
    num2key, key2num = {}, {}
    for r in d["references"]:
        num2key[int(r["number"])] = r["key"]
        key2num[r["key"]] = int(r["number"])
    return num2key, key2num


def main():
    if not os.path.isfile(OLD):
        print("[FATAL] 找不到旧库快照: %s" % OLD)
        return 1
    old_num2key, _ = load_map(OLD)
    _, new_key2num = load_map(NEW)

    removed = sorted(set(old_num2key.values()) - set(new_key2num))
    print("旧库 %d 条 / 新库 %d 条 / 移除 %d 条: %s"
          % (len(old_num2key), len(new_key2num), len(removed), removed))
    print()

    files = sorted(f for f in os.listdir(CHAP_DIR) if f.endswith(".md")) \
        if os.path.isdir(CHAP_DIR) else []
    if not files:
        print("[INFO] chapters/ 下没有 .md, 无需迁移")
        return 0

    total_changed = 0
    problems = []

    # 匹配 [12] / [12,13] / [12-14] / [12, 13-15]
    TOKEN = re.compile(r"\[((?:\d+\s*[-,–]\s*)*\d+)\]")

    for fn in files:
        path = os.path.join(CHAP_DIR, fn)
        src = open(path, encoding="utf-8").read()
        changes = []

        def expand(body):
            """把 '12, 14-16' 展开成 [12,14,15,16] 并记录原样式"""
            nums = []
            for part in re.split(r"\s*,\s*", body):
                m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
                if m:
                    a, b = int(m.group(1)), int(m.group(2))
                    nums.extend(range(a, b + 1))
                else:
                    nums.append(int(part))
            return nums

        def repl(m):
            body = m.group(1)
            olds = expand(body)
            news = []
            for o in olds:
                k = old_num2key.get(o)
                if k is None:
                    problems.append((fn, m.group(0), "旧编号 %d 不在旧库" % o))
                    return m.group(0)
                if k not in new_key2num:
                    problems.append((fn, m.group(0),
                                     "引用了已移除的 %s —— 需人工改写该句" % k))
                    return m.group(0)
                news.append(new_key2num[k])
            # 保持升序去重
            news = sorted(set(news))
            out = "[" + ",".join(str(x) for x in news) + "]"
            if out != m.group(0):
                changes.append((m.group(0), out,
                                [old_num2key[o] for o in olds]))
            return out

        dst = TOKEN.sub(repl, src)

        print("=" * 70)
        print("%s : %d 处标记变更" % (fn, len(changes)))
        for a, b, keys in changes:
            print("   %-12s -> %-12s  %s" % (a, b, ",".join(keys)))
        total_changed += len(changes)

        if APPLY and dst != src:
            shutil.copyfile(path, path + ".bak")
            with open(path, "w", encoding="utf-8") as f:
                f.write(dst)
            print("   [WRITTEN] %s (备份 %s.bak)" % (fn, fn))

    print()
    print("=" * 70)
    if problems:
        print("[BLOCKING] %d 处问题, 未自动处理:" % len(problems))
        for fn, tok, why in problems:
            print("   %s %s : %s" % (fn, tok, why))
        print("=> 先人工处理上述句子, 再重跑")
        return 1
    print("合计 %d 处标记需迁移。%s"
          % (total_changed, "已写盘。" if APPLY else "dry-run，加 --apply 写盘。"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
