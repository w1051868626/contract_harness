"""临时脚本：查看 GitHub Actions 运行状态。"""
import json
import ssl
import urllib.request
import sys

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request(
    "https://api.github.com/repos/w1051868626/contract_harness/actions/runs?per_page=8",
    headers={"User-Agent": "contract-harness/1.0"},
)
try:
    resp = urllib.request.urlopen(req, context=ctx, timeout=20)
    data = json.loads(resp.read())
    for run in data.get("workflow_runs", []):
        name = run["name"]
        conclusion = run.get("conclusion", "?")
        status = run.get("status", "?")
        msg = run["head_commit"]["message"][:60].replace("\n", " ")
        branch = run["head_branch"]
        print(f"{name}: {conclusion} ({status})  [{branch}]")
        print(f"  {msg}")
        print(f"  {run['html_url']}")
        print()
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
