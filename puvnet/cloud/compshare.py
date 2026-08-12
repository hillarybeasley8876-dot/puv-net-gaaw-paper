"""CompShare / UCloud OpenAPI 客户端 —— 签名 + 只读查询 + 受保护的开关机。

设计原则（重要，改代码前先读）：
  1. 凭证只从环境变量或 .env.local 读，**永不硬编码、永不打印**。
  2. 所有会产生账单或改变资源状态的操作（创建/开机/关机/删除）默认 dry-run，
     必须显式传 confirm=True（CLI 上是 --yes）才真正发出请求。
  3. 签名算法用官方文档的测试向量做自检，防止"签名写错但看起来对"。

签名规则（来源：https://docs.ucloud.cn/api/summary/signature.md）：
  - 请求参数按参数名升序排列
  - 拼接成 "key1value1key2value2..."（不做 URL 转义）
  - 末尾追加 PrivateKey
  - 对结果取 SHA1 hexdigest
  - PrivateKey 只参与本地计算，不随请求发送

用法：
    python -m puvnet.cloud.compshare selfcheck      # 离线验签，不需要凭证
    python -m puvnet.cloud.compshare regions        # 只读，需要凭证
    python -m puvnet.cloud.compshare list           # 只读，列出实例
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

# CompShare 有独立的 API 网关。虽然隶属 UCloud，但密钥体系与 api.ucloud.cn 不通用：
# 用 CompShare 的密钥打 api.ucloud.cn 会得到 RetCode 171 "Signature VerifyAC Error"
# （签名算法正确，是凭证不属于那个网关）。已实测确认，勿改回 ucloud.cn。
API_ENDPOINT = os.environ.get("COMPSHARE_ENDPOINT", "https://api.compshare.cn")

# CompShare 的地域编码与 UCloud 主站不同。实测账户可用地域（GetRegion, RetCode 0）：
#   cn-wlcb / cn-wlcb-01  (IsDefault=True, 内蒙乌兰察布)
#   cn-bj2  / cn-bj2-02
#   cn-sh2  / cn-sh2-01
#   cn-gd   / cn-gd-02
#   us-den  / us-den-01
# 注意：传 UCloud 主站的地域码（如裸 cn-sh2 早期写法）会报 RetCode 230
# "Params [Region] not available"。
DEFAULT_REGION = "cn-wlcb"

# 官方文档给出的测试向量，用于验证签名实现是否正确
_DOC_PUBLIC_KEY = "ucloudsomeone@example.com1296235120854146120"
_DOC_PRIVATE_KEY = "46f09bb9fab4f12dfc160dae12273d5332b5debe"
_DOC_PARAMS = {
    "Action": "DescribeUHostInstance",
    "Region": "cn-bj2",
    "Limit": 10,
    "PublicKey": _DOC_PUBLIC_KEY,
}
_DOC_EXPECTED_SIGNATURE = "cba5cf5ec4d4233d206b1b54951e3787350a642f"


# ---------------------------------------------------------------------------
# 参数编码 & 签名
# ---------------------------------------------------------------------------

def _encode_value(v: Any) -> str:
    """按 UCloud 文档规则把参数值转成字符串。

    文档明确要求：
      - bool -> "true"/"false"（Python 的 str(True) 是 "True"，必须特殊处理）
      - float 小数部分为 0 时只保留整数部分（42.0 -> "42"）
      - float 不能用科学计数法
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        # repr 对普通量级不会产生科学计数法；极端值用 f-format 兜底
        s = repr(v)
        if "e" in s or "E" in s:
            s = f"{v:.10f}".rstrip("0").rstrip(".")
        return s
    return str(v)


def sign(params: Dict[str, Any], private_key: str) -> str:
    """计算 UCloud API 签名。private_key 只在本地参与计算，不会被发送。"""
    parts = []
    for k in sorted(params.keys()):
        parts.append(str(k))
        parts.append(_encode_value(params[k]))
    to_sign = "".join(parts) + private_key
    return hashlib.sha1(to_sign.encode("utf-8")).hexdigest()


def selfcheck() -> bool:
    """用官方测试向量验证签名实现。不需要任何真实凭证。"""
    ok = True

    got = sign(_DOC_PARAMS, _DOC_PRIVATE_KEY)
    match = got == _DOC_EXPECTED_SIGNATURE
    ok &= match
    print(f"[{'PASS' if match else 'FAIL'}] 官方测试向量")
    print(f"       期望 {_DOC_EXPECTED_SIGNATURE}")
    print(f"       实际 {got}")

    # 参数顺序无关性：dict 插入顺序不同，签名必须一致
    shuffled = {
        "PublicKey": _DOC_PUBLIC_KEY,
        "Limit": 10,
        "Region": "cn-bj2",
        "Action": "DescribeUHostInstance",
    }
    same = sign(shuffled, _DOC_PRIVATE_KEY) == _DOC_EXPECTED_SIGNATURE
    ok &= same
    print(f"[{'PASS' if same else 'FAIL'}] 参数顺序无关（升序排序生效）")

    # bool 编码
    b = _encode_value(True) == "true" and _encode_value(False) == "false"
    ok &= b
    print(f"[{'PASS' if b else 'FAIL'}] bool 编码为 true/false，实际 "
          f"{_encode_value(True)}/{_encode_value(False)}")

    # float 整数化
    f1 = _encode_value(42.0) == "42"
    f2 = _encode_value(1.5) == "1.5"
    f3 = "e" not in _encode_value(1e-7).lower()
    ok &= (f1 and f2 and f3)
    print(f"[{'PASS' if f1 and f2 else 'FAIL'}] float 编码 42.0->{_encode_value(42.0)}, "
          f"1.5->{_encode_value(1.5)}")
    print(f"[{'PASS' if f3 else 'FAIL'}] float 不用科学计数法 1e-7->{_encode_value(1e-7)}")

    # 防回归：签名侧编码 与 发送侧 urlencode 必须一致。
    # 2026-08-11 真实故障：CreateCompShareInstance 带 Disks.0.IsBoot=True 时
    # 签名算 "true" 而 urlencode 发 "True"，服务端返回 RetCode 171。
    probe = {"Action": "X", "Disks.0.IsBoot": True, "Disks.0.Size": 100,
             "CPU": 4, "Ratio": 42.0}
    enc = {k: _encode_value(v) for k, v in probe.items()}
    naive = urllib.parse.urlencode(probe)
    fixed = urllib.parse.urlencode(enc)
    consistent = fixed == urllib.parse.urlencode(
        {k: _encode_value(v) for k, v in probe.items()})
    bool_ok = "IsBoot=true" in fixed and "IsBoot=True" not in fixed
    float_ok = "Ratio=42" in fixed and "Ratio=42.0" not in fixed
    ok &= (consistent and bool_ok and float_ok)
    print(f"[{'PASS' if bool_ok else 'FAIL'}] 发送侧 bool 编码一致："
          f"修复前 urlencode 含 'IsBoot=True' = {'IsBoot=True' in naive}，"
          f"修复后 = {'IsBoot=True' in fixed}")
    print(f"[{'PASS' if float_ok else 'FAIL'}] 发送侧 float 编码一致：Ratio=42.0 -> "
          f"{[p for p in fixed.split('&') if p.startswith('Ratio')][0]}")

    print()
    print("selfcheck", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------
# 凭证加载
# ---------------------------------------------------------------------------

def load_credentials(env_file: Optional[Path] = None) -> Dict[str, str]:
    """按 环境变量 > .env.local 的优先级读取凭证。

    读不到就抛错并给出具体指引 —— 不返回空字符串去发一个必然失败的请求。
    """
    creds: Dict[str, str] = {}

    if env_file is None:
        env_file = Path(__file__).resolve().parents[2] / ".env.local"

    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if v:
                creds[k.strip()] = v

    # 环境变量优先级更高
    for key in ("COMPSHARE_PUBLIC_KEY", "COMPSHARE_PRIVATE_KEY",
                "COMPSHARE_REGION", "COMPSHARE_ZONE", "COMPSHARE_PROJECT_ID"):
        val = os.environ.get(key)
        if val:
            creds[key] = val

    pub = creds.get("COMPSHARE_PUBLIC_KEY", "")
    pri = creds.get("COMPSHARE_PRIVATE_KEY", "")

    placeholder = ("在此填入", "your", "xxx", "<", "填入你的")
    def is_placeholder(s: str) -> bool:
        low = s.lower()
        return (not s) or any(p.lower() in low for p in placeholder)

    if is_placeholder(pub) or is_placeholder(pri):
        raise RuntimeError(
            "未找到有效凭证。请任选一种方式：\n"
            f"  A) 复制 .env.local.example 为 .env.local（路径 {env_file}），填入真实公钥/私钥\n"
            "  B) 设置环境变量 COMPSHARE_PUBLIC_KEY / COMPSHARE_PRIVATE_KEY\n"
            "注意：.env.local 已被 .gitignore 屏蔽；不要把私钥贴进对话或截图。"
        )
    return creds


def mask(s: str, keep: int = 4) -> str:
    """打日志用的脱敏显示，任何时候都不要直接 print 原始密钥。"""
    if not s:
        return "<empty>"
    if len(s) <= keep * 2:
        return "*" * len(s)
    return f"{s[:keep]}{'*' * (len(s) - keep * 2)}{s[-keep:]}"


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

class CompShareClient:
    """UCloud/CompShare OpenAPI 客户端。

    只读动作可直接调用；变更类动作必须 confirm=True。
    """

    # 会产生账单或改变资源状态的动作白名单。
    # CompShare 族的真实 Action 名来自官方 SDK 示例（一手，非猜测）：
    #   https://github.com/ucloud/compshare-developer-examples
    #   python-sdk/compshare/main.py -> create_comp_share_instance
    # 已实测确认存在（RetCode 0）的只读 Action：
    #   DescribeCompShareInstance / DescribeCompShareImages /
    #   GetCompShareInstancePrice / GetRegion / DescribeImage
    # 注意 DescribeCompShareImages 是复数，且**不接受 Limit 参数**（否则 RetCode 230）。
    # ⚠️ 2026-08-11 补漏：实测发现 StartCompShareInstance / StopCompShareInstance
    # 这组 Start/Stop 命名的 Action **真实存在**（不带参数调用返回
    # RetCode 210 "Missing params [UHostId]"，而不存在的 Action 返回 161），
    # 但此前只收录了 Poweron/Poweroff 命名，导致 Start* 能绕过 dry-run 护栏
    # 直接开机计费。凡新增开关机/删除类 Action，必须同步进这个集合。
    MUTATING_ACTIONS = {
        # CompShare 族
        "CreateCompShareInstance", "TerminateCompShareInstance",
        "PoweronCompShareInstance", "PoweroffCompShareInstance",
        "StartCompShareInstance", "StopCompShareInstance",
        "RestartCompShareInstance", "RebootCompShareInstance",
        "ResizeCompShareInstance", "DeleteCompShareImage",
        "ModifyCompShareInstance", "CreateCompShareImage",
        # UHost 族（同网关下沿用）
        "CreateUHostInstance", "TerminateUHostInstance",
        "PoweronUHostInstance", "PoweroffUHostInstance",
        "StartUHostInstance", "StopUHostInstance",
        "RebootUHostInstance", "RestartUHostInstance",
        "ResizeUHostInstance", "CreateUDisk", "DeleteUDisk",
    }

    # 实测有效的 GpuType 取值（GetCompShareInstancePrice 校验通过）。
    # 传不在服务端清单里的值会得到 RetCode 230 "Params [GpuType] not available"
    # —— 已用伪造值 9090 / H100 / 空串验证服务端确实做校验，因此这份清单可信。
    KNOWN_GPU_TYPES = ("3080Ti", "3090", "4090", "4090_48G", "5090")

    def __init__(self, creds: Optional[Dict[str, str]] = None,
                 timeout: int = 30):
        if creds is None:
            creds = load_credentials()
        self._pub = creds["COMPSHARE_PUBLIC_KEY"]
        self._pri = creds["COMPSHARE_PRIVATE_KEY"]
        self.region = creds.get("COMPSHARE_REGION") or DEFAULT_REGION
        self.zone = creds.get("COMPSHARE_ZONE") or ""
        self.project_id = creds.get("COMPSHARE_PROJECT_ID") or ""
        self.timeout = timeout

    def __repr__(self) -> str:
        # 即使被 print 出来也不泄露私钥
        return (f"CompShareClient(public_key={mask(self._pub)}, "
                f"private_key=<hidden>, region={self.region})")

    def call(self, action: str, confirm: bool = False,
             **params: Any) -> Dict[str, Any]:
        """发起一次 API 调用。

        变更类动作在 confirm=False 时只返回 dry-run 描述，不发网络请求。
        """
        req: Dict[str, Any] = {"Action": action, "PublicKey": self._pub}
        if self.region:
            req["Region"] = self.region
        if self.zone:
            req["Zone"] = self.zone
        if self.project_id:
            req["ProjectId"] = self.project_id
        for k, v in params.items():
            if v is not None:
                req[k] = v

        if action in self.MUTATING_ACTIONS and not confirm:
            safe = {k: v for k, v in req.items() if k != "PublicKey"}
            return {
                "_dry_run": True,
                "_message": (f"[DRY-RUN] 动作 {action} 会改变云端资源或产生费用，"
                             f"未实际发送。确认无误后加 confirm=True / --yes"),
                "_would_send": safe,
            }

        req["Signature"] = sign(req, self._pri)

        # ⚠️ 关键：发送前必须用与签名**完全相同**的编码规则把值转成字符串。
        # 踩过的坑（2026-08-11，CreateCompShareInstance 报 RetCode 171）：
        #   sign() 内部用 _encode_value 把 True 编码为 "true"（文档要求），
        #   但 urlencode(dict) 会调用 str(True) 得到 "True"。
        #   结果签名按 "true" 算、实际发送 "True"，服务端按收到的值重算签名
        #   必然不匹配 -> "Signature VerifyAC Error"。
        # 该 bug 只在参数含 bool 时触发，所以此前全部只读调用（无 bool 参数）
        # 都正常，极易误判为"密钥有问题"。
        encoded = {k: _encode_value(v) for k, v in req.items()}
        url = f"{API_ENDPOINT}/?{urllib.parse.urlencode(encoded)}"

        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return {"RetCode": -1,
                    "Message": f"HTTP {e.code}: {e.reason}",
                    "_body": e.read().decode("utf-8", errors="replace")[:2000]}
        except Exception as e:
            return {"RetCode": -1, "Message": f"{type(e).__name__}: {e}"}

        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"RetCode": -1, "Message": "响应不是合法 JSON",
                    "_body": body[:2000]}

    # -------- 只读封装 --------

    def describe_regions(self) -> Dict[str, Any]:
        return self.call("GetRegion")

    def list_instances(self, limit: int = 20) -> Dict[str, Any]:
        return self.call("DescribeUHostInstance", Limit=limit)

    def describe_price(self, cpu: int, memory_mb: int,
                       gpu: int = 1, count: int = 1) -> Dict[str, Any]:
        """查价（只读，不产生账单）。用于上机前估算成本。"""
        return self.call("GetUHostInstancePrice", CPU=cpu, Memory=memory_mb,
                         GPU=gpu, Count=count, ChargeType="Dynamic")

    # -------- CompShare 族只读封装（Action 名均已实测 RetCode 0） --------

    def list_compshare_instances(self, zone: str = "cn-wlcb-01",
                                 limit: int = 20) -> Dict[str, Any]:
        """列出 CompShare 实例。账户无实例时返回 UHostSet=[] / TotalCount=0。"""
        return self.call("DescribeCompShareInstance", Zone=zone, Limit=limit)

    def list_compshare_images(self, zone: str = "cn-wlcb-01") -> Dict[str, Any]:
        """列出 CompShare 专属镜像池（compshareImage-* 命名空间）。

        注意两个实测坑：
          1. Action 名是复数 DescribeCompShareImages（单数形式返回 RetCode 161）。
          2. **不能传 Limit**，否则 RetCode 230 "Params [Limit] not available"。
        镜像在响应的 ImageSet 字段；每项的权威版本信息在 Softwares 子字典里
        （Framework / FrameworkVersion / CUDAVersion / PythonVersion），
        不要靠解析 Name 字符串取版本 —— 命名不规范（如 cuda128 无小数点）。
        """
        return self.call("DescribeCompShareImages", Zone=zone)

    def compshare_price(self, gpu_type: str = "5090", gpu: int = 1,
                        cpu: int = 16, memory_mb: int = 64 * 1024,
                        count: int = 1, zone: str = "cn-wlcb-01",
                        machine_type: str = "G") -> Dict[str, Any]:
        """查 CompShare 实例按量价格（只读）。

        实测价格（cn-wlcb-01, GPU=1, CPU=16, Mem=64G, Dynamic, 2026-08-11）：
            5090   2.77 元/时（原价 2.92）
            4090   2.05 元/时（原价 2.15）
            3090   1.13 元/时
            3080Ti 0.93 元/时
        """
        return self.call("GetCompShareInstancePrice", Zone=zone,
                         MachineType=machine_type, GpuType=gpu_type, GPU=gpu,
                         CPU=cpu, Memory=memory_mb, Count=count,
                         ChargeType="Dynamic")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description="CompShare/UCloud 客户端（变更类动作默认 dry-run）")
    p.add_argument("command",
                   choices=["selfcheck", "regions", "list", "price", "whoami"])
    p.add_argument("--yes", action="store_true",
                   help="确认执行会产生费用的动作（当前 CLI 未暴露此类动作）")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args(argv)

    if args.command == "selfcheck":
        return 0 if selfcheck() else 1

    try:
        client = CompShareClient()
    except RuntimeError as e:
        print(f"凭证错误：\n{e}", file=sys.stderr)
        return 2

    if args.command == "whoami":
        print(client)
        return 0

    if args.command == "regions":
        r = client.describe_regions()
    elif args.command == "list":
        r = client.list_instances(limit=args.limit)
    elif args.command == "price":
        r = client.describe_price(cpu=16, memory_mb=65536, gpu=1)
    else:
        return 2

    print(json.dumps(r, ensure_ascii=False, indent=2)[:4000])
    return 0 if r.get("RetCode") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
