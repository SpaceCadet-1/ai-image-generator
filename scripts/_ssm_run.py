"""Run a command on the AWS instance via SSM and print the output."""
import json
import subprocess
import sys
import time

INSTANCE_ID = "i-068e8e5b4c7fe01e9"
PROFILE = "SysmexAI"
REGION = "us-east-1"

commands = sys.argv[1:]
if not commands:
    print("Usage: python _ssm_run.py 'command1' 'command2' ...")
    sys.exit(1)

result = subprocess.run(
    [
        "aws", "ssm", "send-command",
        "--instance-ids", INSTANCE_ID,
        "--document-name", "AWS-RunShellScript",
        "--parameters", json.dumps({"commands": commands}),
        "--profile", PROFILE,
        "--region", REGION,
        "--output", "json",
    ],
    capture_output=True, text=True,
)

if result.returncode != 0:
    print("SEND FAILED:", result.stderr)
    sys.exit(1)

cmd_id = json.loads(result.stdout)["Command"]["CommandId"]

for _ in range(24):
    time.sleep(5)
    result = subprocess.run(
        [
            "aws", "ssm", "get-command-invocation",
            "--command-id", cmd_id,
            "--instance-id", INSTANCE_ID,
            "--profile", PROFILE,
            "--region", REGION,
            "--output", "json",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        continue
    data = json.loads(result.stdout.decode("utf-8"))
    if data["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
        print(f"Status: {data['Status']}")
        out = data.get("StandardOutputContent", "")
        if out:
            print(out.encode("ascii", "replace").decode())
        err = data.get("StandardErrorContent", "")
        if err:
            print("STDERR:", err.encode("ascii", "replace").decode())
        sys.exit(0 if data["Status"] == "Success" else 1)

print("Timed out")
sys.exit(1)
