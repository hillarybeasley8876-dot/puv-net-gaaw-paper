"""下载 PU1K / PU-GAN 数据集。

数据来源（见 docs/SOTA_SURVEY.md §2）：
  PU1K 训练+测试   : Google Drive 1oTAx34YNbL6GDwHYL2qqvjmYtTVWcELg
  PU1K 原始 mesh   : Google Drive 1tnMjJUeh1e27mCRSNmICwGCQDl20mFae
  PU-GAN 训练集    : Google Drive 13ZFDffOod_neuF3sOM0YiqNbIJEeSKdZ
  PU-GAN 测试 mesh : Google Drive 1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC

Google Drive 大文件会返回病毒扫描确认页而非文件本体，
需要解析确认 token 后二次请求 —— 本脚本已处理。

用法
----
    python scripts/download_data.py --probe          # 只探测可达性与文件大小，不下载
    python scripts/download_data.py --which pu1k     # 下载指定数据集
    python scripts/download_data.py --all            # 全部下载
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

DATASETS: dict[str, dict[str, str]] = {
    "pu1k": {
        "id": "1oTAx34YNbL6GDwHYL2qqvjmYtTVWcELg",
        "filename": "PU1K.zip",
        "desc": "PU1K 训练集(h5) + 测试集",
    },
    "pu1k_mesh": {
        "id": "1tnMjJUeh1e27mCRSNmICwGCQDl20mFae",
        "filename": "PU1K_meshes.zip",
        "desc": "PU1K 原始 mesh（P2F 指标必需）",
    },
    "pugan_train": {
        "id": "13ZFDffOod_neuF3sOM0YiqNbIJEeSKdZ",
        "filename": "PUGAN_poisson_256_poisson_1024.h5",
        "desc": "PU-GAN 训练集",
    },
    "pugan_test_mesh": {
        "id": "1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC",
        "filename": "PUGAN_test_meshes.zip",
        "desc": "PU-GAN 测试 mesh（27 个）",
    },
}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def _extract_confirm_token(resp: requests.Response) -> str | None:
    """从 Drive 的确认页里抽 token。"""
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    m = re.search(r'name="confirm"\s+value="([^"]+)"', resp.text)
    if m:
        return m.group(1)
    m = re.search(r"confirm=([0-9A-Za-z_-]+)", resp.text)
    if m:
        return m.group(1)
    return None


def probe(name: str, info: dict[str, str]) -> None:
    """只探测：报告能否拿到文件流与大小，不落盘。"""
    url = "https://drive.usercontent.google.com/download"
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    try:
        r = sess.get(url, params={"id": info["id"], "export": "download"},
                     stream=True, timeout=30, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "")
        clen = r.headers.get("Content-Length")

        if "text/html" in ctype:
            token = _extract_confirm_token(r)
            if token:
                r.close()
                r = sess.get(url,
                             params={"id": info["id"], "export": "download",
                                     "confirm": token},
                             stream=True, timeout=30, allow_redirects=True)
                ctype = r.headers.get("Content-Type", "")
                clen = r.headers.get("Content-Length")

        if "text/html" in ctype:
            snippet = r.text[:200].replace("\n", " ")
            print(f"  [BLOCKED] {name:<16} 仍返回 HTML: {snippet}")
        else:
            size = f"{int(clen)/1e6:.1f} MB" if clen else "未知大小"
            print(f"  [OK]      {name:<16} Content-Type={ctype}  size={size}")
        r.close()
    except Exception as e:
        print(f"  [FAIL]    {name:<16} {type(e).__name__}: {e}")


def download(name: str, info: dict[str, str]) -> bool:
    url = "https://drive.usercontent.google.com/download"
    out = RAW / info["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 0:
        print(f"  [SKIP] {info['filename']} 已存在 "
              f"({out.stat().st_size/1e6:.1f} MB)")
        return True

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    print(f"  下载 {name}: {info['desc']}")

    try:
        r = sess.get(url, params={"id": info["id"], "export": "download"},
                     stream=True, timeout=60, allow_redirects=True)
        if "text/html" in r.headers.get("Content-Type", ""):
            token = _extract_confirm_token(r)
            if not token:
                print(f"  [FAIL] 无法取得确认 token，需手动下载")
                print(f"         https://drive.google.com/file/d/{info['id']}/view")
                return False
            r.close()
            r = sess.get(url,
                         params={"id": info["id"], "export": "download",
                                 "confirm": token},
                         stream=True, timeout=60, allow_redirects=True)

        if "text/html" in r.headers.get("Content-Type", ""):
            print(f"  [FAIL] 仍被拦截，需手动下载")
            print(f"         https://drive.google.com/file/d/{info['id']}/view")
            return False

        total = int(r.headers.get("Content-Length", 0))
        done = 0
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r    {done/1e6:8.1f} / {total/1e6:.1f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r    {done/1e6:8.1f} MB", end="", flush=True)
        print()
        tmp.rename(out)
        print(f"  [OK] {out.name} = {out.stat().st_size/1e6:.1f} MB")
        return True
    except Exception as e:
        print(f"\n  [FAIL] {type(e).__name__}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只探测，不下载")
    ap.add_argument("--which", nargs="*", choices=list(DATASETS),
                    help="下载指定数据集")
    ap.add_argument("--all", action="store_true", help="下载全部")
    args = ap.parse_args()

    print("=" * 70)
    print(f"数据集目标目录: {RAW}")
    print("=" * 70)

    if args.probe:
        print("\n探测 Google Drive 可达性（不下载）：\n")
        for name, info in DATASETS.items():
            probe(name, info)
        print("\n说明：[OK] = 能拿到二进制流；[BLOCKED] = 需手动下载。")
        return 0

    targets = list(DATASETS) if args.all else (args.which or [])
    if not targets:
        ap.print_help()
        return 1

    results = {}
    for name in targets:
        print()
        results[name] = download(name, DATASETS[name])

    print("\n" + "=" * 70)
    print("下载结果汇总")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name:<16} {DATASETS[name]['desc']}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
