# Standalone Raspberry Pi Phase A Harness

## Quick Deployment (via SCP / USB)
1. Copy this folder to Raspberry Pi:
   ```bash
   scp -r artifacts/pi_phase_a/ pi@<PI_IP>:~/pi_phase_a
   ```
2. On Raspberry Pi:
   ```bash
   cd ~/pi_phase_a
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-pi.txt
   bash run_phase_a_pi.sh
   ```
3. Retrieve results:
   ```bash
   scp pi@<PI_IP>:~/pi_phase_a/benchmarks/pi/phase_a_pi_summary.csv ./
   ```
