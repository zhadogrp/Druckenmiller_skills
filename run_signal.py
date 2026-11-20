"""Run one signal script with a data-quality guard.

Yahoo Finance can rate-limit cloud IPs. When that happens the signal scripts
swallow the errors and fall back to neutral 50 scores, which would publish
fake data. This wrapper counts fetch errors in the script output; if too many
fetches failed it waits, retries, and as a last resort deletes today's
(placeholder) output for that signal so the synthesizer falls back to the most
recent good report and marks the component as stale.

Usage: python run_signal.py <script.py> <report_prefix> --output-dir DIR
"""
import argparse, glob, os, re, subprocess, sys, time
from datetime import datetime

ERR = re.compile(r"RateLimit|Too Many Requests|Failed download|YFPricesMissing|possibly delisted|No data found|\[WARN\]|Error", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("script")
ap.add_argument("prefix")
ap.add_argument("--output-dir", default="public/reports/")
ap.add_argument("--max-errors", type=int, default=3)
ap.add_argument("--retries", type=int, default=2)
ap.add_argument("--wait", type=int, default=120)
a = ap.parse_args()

today = datetime.today().strftime("%Y-%m-%d")
for attempt in range(a.retries + 1):
    p = subprocess.run([sys.executable, a.script, "--output-dir", a.output_dir],
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    print(out)
    errors = [l for l in out.splitlines() if ERR.search(l)]
    if p.returncode == 0 and len(errors) <= a.max_errors:
        print(f"[guard] {a.prefix}: OK ({len(errors)} fetch errors)")
        sys.exit(0)
    print(f"[guard] {a.prefix}: attempt {attempt + 1} failed "
          f"(rc={p.returncode}, {len(errors)} fetch errors)")
    if attempt < a.retries:
        time.sleep(a.wait)

for f in glob.glob(os.path.join(a.output_dir, f"{a.prefix}_{today}.*")):
    os.remove(f)
    print(f"[guard] removed placeholder {f}")
print(f"[guard] {a.prefix}: giving up - synthesizer will use last good report (stale)")
sys.exit(1)
