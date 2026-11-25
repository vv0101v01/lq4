#!/usr/bin/env python3
"""Wait for a t4-small on Transynergy/6in1, then run distillation chunks until done.

ZeroGPU capacity comes and goes ("Scheduling failure: not enough hardware
capacity"), so this re-requests the hardware and polls instead of failing on the
first PAUSED.  Usage:

    HF_TOKEN=... python3 resume_training.py --chunks 12 --steps 8

Each chunk calls /distill_step, which resumes from /data/ckpt/step_*.pt.  Progress
is appended to resume_training.log so a second run does not redo work.
"""
import argparse, json, os, sys, time

SPACE = "Transynergy/6in1"
TOKEN = os.environ.get("HF_TOKEN")


def wait_for_gpu(max_wait=3600, verbose=True):
    from huggingface_hub import HfApi

    api = HfApi(token=TOKEN)
    t0 = time.time()
    last = None
    while time.time() - t0 < max_wait:
        rt = api.space_info(SPACE).runtime
        key = (rt.stage, rt.hardware)
        if verbose and key != last:
            print(f"  [{time.time() - t0:6.0f}s] {rt.stage} hw={rt.hardware}", flush=True)
            last = key
        if rt.stage == "RUNNING" and rt.hardware:
            return True
        if rt.stage == "PAUSED":
            api.request_space_hardware(SPACE, "t4-small", sleep_time=172800)
        time.sleep(20)
    return False


def client():
    from gradio_client import Client

    return Client(SPACE, token=TOKEN, verbose=False)


def run_chunk(c, steps=8, k=256, lr=4e-4, max_len=384, limit=1200, teachers_n=2):
    md = c.predict(steps, k, lr, max_len, limit, False, teachers_n, api_name="/distill_step")
    out = {}
    for line in md.splitlines():
        s = line.strip("- * ")
        if "**step" in line:
            out["step"] = s
        elif "loss:" in line:
            out["loss"] = s
    return md, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--max-wait", type=int, default=3600)
    ap.add_argument("--log", default="resume_training.log")
    a = ap.parse_args()
    if not TOKEN:
        sys.exit("HF_TOKEN is not set")

    print("waiting for t4-small ...")
    if not wait_for_gpu(a.max_wait):
        sys.exit("no GPU within the wait window — rerun later; /data/ckpt is untouched")

    c = client()
    ok = 0
    for i in range(a.chunks):
        t0 = time.time()
        try:
            md, info = run_chunk(c, a.steps)
            ok += 1
            line = f"[{i:02d} {time.time() - t0:3.0f}s] {info.get('step', '?')} | {info.get('loss', '?')}"
        except Exception as e:  # a crashed worker usually just needs a fresh client
            line = f"[{i:02d}] ERR {type(e).__name__}: {str(e)[:70]}"
            try:
                c = client()
            except Exception:
                if not wait_for_gpu(a.max_wait):
                    break
                c = client()
        print(line, flush=True)
        with open(a.log, "a") as f:
            f.write(json.dumps({"t": time.time(), "line": line}) + "\n")
        time.sleep(2)
    print(f"done: {ok}/{a.chunks} chunks succeeded")


if __name__ == "__main__":
    main()
