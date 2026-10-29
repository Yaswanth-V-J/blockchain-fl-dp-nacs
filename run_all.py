import subprocess
import sys
import time

PHASE_SCRIPTS = [
    ("Phase 1: Baseline FL", "run_phase1.py"),
    ("Phase 2: Differential Privacy", "run_phase2_dp.py"),
    ("Phase 3: Connected FL + DP + Blockchain", "run_phase3_connected.py"),
    ("Phase 4: NACS Architecture", "run_phase4_nacs.py"),
    ("Phase 5: HAR Dataset Benchmark", "run_phase5_har.py"),
    ("Phase 6: NIH Chest X-Ray Benchmark", "run_phase6_nih.py"),
]

def run_phase(name, script_path):
    print("\n" + "=" * 70, flush=True)
    print(f"  STARTING {name.upper()}", flush=True)
    print("=" * 70 + "\n", flush=True)
    start_time = time.time()
    
    # Run python script unbuffered so live output streams immediately
    res = subprocess.run([sys.executable, "-u", script_path])
    
    elapsed = time.time() - start_time
    if res.returncode == 0:
        print(f"\n[SUCCESS] {name} completed in {elapsed:.2f} seconds.\n", flush=True)
        return True
    else:
        print(f"\n[FAILED] {name} failed with exit code {res.returncode} after {elapsed:.2f} seconds.\n", flush=True)
        return False

def main():
    print("Starting Sequential Execution of All Phases...", flush=True)
    total_start = time.time()
    
    summary = []
    for name, script in PHASE_SCRIPTS:
        success = run_phase(name, script)
        summary.append((name, "SUCCESS" if success else "FAILED"))
        if not success:
            print(f"Stopping execution pipeline due to failure in {name}.", flush=True)
            break
            
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70, flush=True)
    print("  PIPELINE EXECUTION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for name, status in summary:
        print(f"  - {name}: {status}", flush=True)
    print(f"\nTotal Time Elapsed: {total_elapsed / 60:.2f} minutes", flush=True)
    print("=" * 70 + "\n", flush=True)

if __name__ == "__main__":
    main()
