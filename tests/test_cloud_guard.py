"""dry-run 护栏验证 —— 用假凭证确认变更类动作不会真发请求。"""
from puvnet.cloud.compshare import CompShareClient

c = CompShareClient()
print("repr 脱敏:", c)
print()

r = c.call("CreateUHostInstance", Name="puvnet-train", CPU=16, Memory=65536)
print("dry_run =", r.get("_dry_run"))
print(r.get("_message"))
print("would_send =", r.get("_would_send"))
print()

r2 = c.call("PoweroffUHostInstance", UHostId="uhost-fake")
print("关机也被拦 =", r2.get("_dry_run"))
print()

leaked = "fake-pri-for-guard-test" in repr(c)
print("私钥是否泄漏进 repr =", leaked, "(必须为 False)")
