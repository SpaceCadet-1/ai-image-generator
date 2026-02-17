"""Deploy built frontend (client/dist/) to the AWS instance.

Sends each file individually via SSM to avoid command length limits.
"""
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

INSTANCE_ID = "i-068e8e5b4c7fe01e9"
PROFILE = "SysmexAI"
REGION = "us-east-1"
DIST_DIR = Path(r"C:\Development\ai-image-generator\client\dist")
REMOTE_DIR = "/home/ubuntu/ai-image-generator/client/dist"


def run_ssm(commands, timeout_iters=24):
    """Run commands via SSM and wait for completion."""
    r = subprocess.run(
        ["aws", "ssm", "send-command",
         "--instance-ids", INSTANCE_ID,
         "--document-name", "AWS-RunShellScript",
         "--parameters", json.dumps({"commands": commands}),
         "--profile", PROFILE, "--region", REGION, "--output", "json"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("SEND FAILED:", r.stderr)
        return False

    cmd_id = json.loads(r.stdout)["Command"]["CommandId"]

    for _ in range(timeout_iters):
        time.sleep(5)
        r2 = subprocess.run(
            ["aws", "ssm", "get-command-invocation",
             "--command-id", cmd_id, "--instance-id", INSTANCE_ID,
             "--profile", PROFILE, "--region", REGION, "--output", "json"],
            capture_output=True,
        )
        if r2.returncode != 0:
            continue
        d = json.loads(r2.stdout.decode("utf-8"))
        if d["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
            out = d.get("StandardOutputContent", "")
            if out:
                print(out.strip())
            if d["Status"] != "Success":
                err = d.get("StandardErrorContent", "")
                if err:
                    print("STDERR:", err[:500])
                return False
            return True
    print("  Timed out")
    return False


# 1. Clear old dist and create directories
print("Preparing remote directories...")
run_ssm([
    f"rm -rf {REMOTE_DIR}",
    f"mkdir -p {REMOTE_DIR}/assets",
    f"echo 'Directories ready'",
])

# 2. Deploy each file
# SSM + Windows command line has ~32K char limit. Base64 expands by 4/3.
# So max raw bytes per chunk ≈ 20KB to be safe.
CHUNK_SIZE = 20_000

for filepath in sorted(DIST_DIR.rglob("*")):
    if not filepath.is_file():
        continue
    rel = filepath.relative_to(DIST_DIR).as_posix()
    raw = filepath.read_bytes()
    remote_path = f"{REMOTE_DIR}/{rel}"

    if len(raw) <= CHUNK_SIZE:
        b64 = base64.b64encode(raw).decode()
        print(f"Deploying {rel} ({len(raw):,} bytes)...")
        ok = run_ssm([f"echo '{b64}' | base64 -d > {remote_path}", f"echo 'Wrote {rel}'"])
    else:
        # Split into chunks, append to file
        chunks = []
        for i in range(0, len(raw), CHUNK_SIZE):
            chunks.append(raw[i:i + CHUNK_SIZE])
        print(f"Deploying {rel} ({len(raw):,} bytes, {len(chunks)} chunks)...")
        for ci, chunk in enumerate(chunks):
            b64 = base64.b64encode(chunk).decode()
            op = ">" if ci == 0 else ">>"
            ok = run_ssm([f"echo '{b64}' | base64 -d {op} {remote_path}"])
            if not ok:
                print(f"  FAILED chunk {ci}")
                sys.exit(1)
        # Verify size
        ok = run_ssm([f"wc -c < {remote_path}"])
        print(f"  Wrote {rel}")

    if not ok:
        print(f"  FAILED to deploy {rel}")
        sys.exit(1)

# 3. Restart service
print("Restarting service...")
run_ssm([
    "sudo systemctl restart ai-image-generator",
    "sleep 3",
    "systemctl is-active ai-image-generator",
])

print("Frontend deployed.")
