"""5-GPU scheduler for the rebuttal experiments (E1 then E2).

Runs at most ONE training job per GPU at a time (avoids OOM / contention),
launching jobs in priority order: all E1 (dev-split) runs first, then all E2
(controlled-LR) runs. Every run's full stdout+stderr goes to
<log_dir>/train.log under /root/autodl-tmp; the scheduler itself logs to
/root/autodl-tmp/logs-rebuttal/orchestrator.log.

Usage:
    python3 scripts/run_rebuttal_queue.py            # GPUs 0-4
    python3 scripts/run_rebuttal_queue.py 0 1 2 3    # custom GPU set
"""
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = "/root/GLiREL"
ORCH_DIR = "/root/autodl-tmp/logs-rebuttal"
os.makedirs(ORCH_DIR, exist_ok=True)
ORCH_LOG = os.path.join(ORCH_DIR, "orchestrator.log")

ENV = {
    **os.environ,
    "PYTHONPATH": REPO,
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

EXPS = ["exp1", "exp2", "exp3"]


def build_jobs():
    jobs = []
    # E1 (priority 1): dev-split, 3 methods x 3 splits.
    for method in ["repro", "cascade", "innovation2"]:
        for e in EXPS:
            jobs.append({
                "name": f"E1-{method}-{e}",
                "entry": "train_leakage_free.py",
                "config": f"configs/e1_devsplit/config_wiki_zsl_{method}_dev_{e}.yaml",
                "log_dir": f"/root/autodl-tmp/logs-e1/{method}_wikizsl/{e}",
            })
    # E2 (priority 2): controlled LR, 3 variants x 3 splits.
    for variant in ["baseline_lowlr", "cca_highlr", "scar_highlr"]:
        for e in EXPS:
            jobs.append({
                "name": f"E2-{variant}-{e}",
                "config": f"configs/e2_controlled_lr/config_{variant}_{e}.yaml",
                "log_dir": f"/root/autodl-tmp/logs-e2/{variant}_wikizsl/{e}",
            })
    return jobs


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ORCH_LOG, "a") as f:
        f.write(line + "\n")


MAX_ATTEMPTS = 3


def launch(job, gpu):
    os.makedirs(job["log_dir"], exist_ok=True)
    job["attempt"] = job.get("attempt", 0) + 1
    logf = open(os.path.join(job["log_dir"], "train.log"), "w")
    env = {**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}
    # start_new_session=True (setsid) so a session/SIGHUP event on the launcher
    # does not take the training children down with it.
    proc = subprocess.Popen(
        ["python3", job.get("entry", "train.py"), "--config", job["config"], "--log_dir", job["log_dir"]],
        cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job["_logf"] = logf
    job["_start"] = time.time()
    log(f"LAUNCH {job['name']} (attempt {job['attempt']}/{MAX_ATTEMPTS}) on GPU {gpu} "
        f"(pid {proc.pid}) -> {job['log_dir']}/train.log")
    return proc


def main():
    gpus = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4]
    jobs = build_jobs()

    # Skip jobs that already completed successfully (marker written on rc=0),
    # so re-running the queue only reruns what failed.
    pending, skipped = [], []
    for j in jobs:
        if os.path.exists(os.path.join(j["log_dir"], "_SUCCESS")):
            skipped.append(j["name"])
        else:
            pending.append(j)
    log(f"=== rebuttal queue start: {len(pending)} to run, {len(skipped)} already done, GPUs {gpus} ===")
    if skipped:
        log(f"skipping (already _SUCCESS): {skipped}")

    gpu_state = {g: None for g in gpus}  # gpu -> (proc, job) or None
    done, failed = [], []

    while pending or any(v is not None for v in gpu_state.values()):
        for g in gpus:
            if gpu_state[g] is None and pending:
                job = pending.pop(0)
                proc = launch(job, g)
                gpu_state[g] = (proc, job)
        for g in gpus:
            if gpu_state[g] is not None:
                proc, job = gpu_state[g]
                rc = proc.poll()
                if rc is not None:
                    job["_logf"].close()
                    dur = int(time.time() - job["_start"])
                    if rc == 0:
                        open(os.path.join(job["log_dir"], "_SUCCESS"), "w").close()
                        done.append(job["name"])
                        log(f"DONE  {job['name']} (rc=0, {dur}s). done={len(done)}")
                    elif job["attempt"] < MAX_ATTEMPTS:
                        log(f"RETRY {job['name']} (rc={rc}, {dur}s, attempt {job['attempt']}). re-queueing")
                        pending.append(job)  # retry on a later free GPU
                    else:
                        failed.append(job["name"])
                        log(f"FAIL  {job['name']} (rc={rc}, {dur}s, gave up after {job['attempt']}). "
                            f"See {job['log_dir']}/train.log")
                    gpu_state[g] = None
        time.sleep(20)

    log(f"=== all finished. done={len(done)} failed={len(failed)} failed_list={failed} ===")


if __name__ == "__main__":
    main()
