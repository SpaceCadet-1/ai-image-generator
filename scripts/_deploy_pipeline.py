"""Deploy updated pipeline.py to the AWS instance and restart the service."""
import base64
import json
import subprocess
import sys
import time

INSTANCE_ID = "i-068e8e5b4c7fe01e9"
PROFILE = "SysmexAI"
REGION = "us-east-1"

# Read the local pipeline.py
with open(r"C:\Development\ai-image-generator\server\pipeline.py", "rb") as f:
    content = f.read()

b64 = base64.b64encode(content).decode()

commands = [
    f"echo '{b64}' | base64 -d > /home/ubuntu/ai-image-generator/server/pipeline.py",
    "echo 'File written, restarting service...'",
    "sudo systemctl restart ai-image-generator",
    "sleep 3",
    "systemctl is-active ai-image-generator",
]

r = subprocess.run(
    ["aws", "ssm", "send-command",
     "--instance-ids", INSTANCE_ID,
     "--document-name", "AWS-RunShellScript",
     "--parameters", json.dumps({"commands": commands}),
     "--profile", PROFILE, "--region", REGION, "--output", "json"],
    capture_output=True,
)

if r.returncode != 0:
    print("SEND FAILED:", r.stderr.decode())
    sys.exit(1)

cmd_id = json.loads(r.stdout)["Command"]["CommandId"]
print(f"Command: {cmd_id}")

for _ in range(30):
    time.sleep(5)
    r2 = subprocess.run(
        ["aws", "ssm", "get-command-invocation",
         "--command-id", cmd_id,
         "--instance-id", INSTANCE_ID,
         "--profile", PROFILE, "--region", REGION, "--output", "json"],
        capture_output=True,
    )
    if r2.returncode != 0:
        continue
    d = json.loads(r2.stdout.decode("utf-8"))
    if d["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
        print(f"Status: {d['Status']}")
        out = d.get("StandardOutputContent", "")
        if out:
            print(out.encode("ascii", "replace").decode())
        err = d.get("StandardErrorContent", "")
        if err:
            print("STDERR:", err.encode("ascii", "replace").decode())
        sys.exit(0 if d["Status"] == "Success" else 1)

print("Timed out")
sys.exit(1)
