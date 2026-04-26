#!/usr/bin/env python3
"""
Franken-Fit × Pioneer (Fastino Labs) — Day-1 → Day-2 fine-tune loop.

Narrative this script enables on stage:

  Day 1 (live):   Qwen model (PIONEER_QWEN_MODEL) handles love/hate inference
                  in the app — strong zero-shot predictions, good on stage.
                  Users swipe; events write JSONL to func_test/out/live_swipes.jsonl.

  Day 1 evening:  Run this script to build the training dataset from real swipe
                  data + curated seed rows, kick off a LoRA fine-tune on
                  fastino/gliner2-base-v1, and let it bake overnight.

  Day 2 morning:  Poll the job, run /felix/evaluations for the F1 slide, then
                  demo side-by-side Qwen vs fine-tuned GLiNER on the same garment.

  Pitch frame: "We started with a 7B-parameter model. By morning, your taste is
  captured in a task-specific model that is 10× smaller."

Classifier is BINARY (love / hate) — matching the swipe UX. No 'meh' class:
  - Users can only produce love/hate signals in the app.
  - Binary labels sharpen the decision boundary.
  - Side-by-side divergence (baseline=love, trained=hate) is unambiguous on stage.

Dataset sources (loaded in priority order, deduped):
  func_test/out/preference-training-data.jsonl  — 50-row hand-curated set
  func_test/out/pioneer_style_dataset.jsonl     — 10-row seed
  func_test/out/live_swipes.jsonl               — auto-collected from POST /v1/wardrobe/swipe

Usage (from repo root with venv active):
  python -m scripts.pioneer.train --phase=all --skip-generate
  python -m scripts.pioneer.train --phase=dataset --num-synthetic=10
  python -m scripts.pioneer.train --phase=train  --dataset-name=frankenfit-binary-v3
  python -m scripts.pioneer.train --phase=poll
  python -m scripts.pioneer.train --phase=eval
  python -m scripts.pioneer.train --phase=infer --base-model=$PIONEER_QWEN_MODEL

Env (from .env or shell):
  PIONEER_API_KEY           required
  PIONEER_API_BASE          default https://api.pioneer.ai
  PIONEER_BASE_MODEL        GLiNER base to fine-tune, default fastino/gliner2-base-v1
  PIONEER_QWEN_MODEL        Qwen model ID for the side-by-side baseline (--phase=infer)
  PIONEER_TRAINED_MODEL_ID  resume a known training job (overrides last_pioneer_training.json)
  PIONEER_DATASET_NAME      default dataset name (overridable with --dataset-name)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Resolve repo root (scripts/pioneer/train.py → repo root is 2 levels up)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(_REPO_ROOT / ".env", override=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FUNC_TEST_OUT = _REPO_ROOT / "func_test" / "out"
_OUT_DIR = _REPO_ROOT / "scripts" / "pioneer" / "out"

# Seed data lives in func_test/out (shared with the backend's live_swipes path)
SEED_SOURCES: list[Path] = [
    _FUNC_TEST_OUT / "preference-training-data.jsonl",
    _FUNC_TEST_OUT / "pioneer_style_dataset.jsonl",
    _FUNC_TEST_OUT / "live_swipes.jsonl",
]

DATASET_JSONL = _OUT_DIR / "pioneer_training_dataset.jsonl"
DATASET_META  = _OUT_DIR / "last_pioneer_dataset.json"
TRAINING_META = _OUT_DIR / "last_pioneer_training.json"
EVAL_META     = _OUT_DIR / "last_pioneer_eval.json"
SIDEBYSIDE_META = _OUT_DIR / "last_pioneer_sidebyside.json"

# ---------------------------------------------------------------------------
# Labels — BINARY, matches swipe UX
# ---------------------------------------------------------------------------

LABELS = ["love", "hate"]

SIDEBYSIDE_PROBES: list[str] = [
    "Slouchy camel-tone wool coat with hidden button placket",
    "Glitter mesh halter top, club-ready, neon pink",
    "Plain navy polo, cotton pique, no branding",
    "Stonewashed vintage Levi's trucker jacket, cropped",
    "Y2K bedazzled bell sleeve top with rhinestones, low rise",
]

# ---------------------------------------------------------------------------
# Seed loader
# ---------------------------------------------------------------------------

def _load_seed_rows() -> list[dict[str, str]]:
    """Merge SEED_SOURCES, normalise keys ({text, label}), drop non-LABELS rows, dedupe."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in SEED_SOURCES:
        if not path.is_file():
            print(f"  (skip: {path.name} not found)", file=sys.stderr)
            continue
        kept = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text  = (obj.get("text") or obj.get("description") or "").strip()
            label = (obj.get("label") or obj.get("sentiment") or "").strip().lower()
            if not text or label not in LABELS:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "label": label})
            kept += 1
        print(f"  + {kept:3d} rows from {path.name}", file=sys.stderr)
    if not rows:
        raise RuntimeError(
            "No seed rows found — populate at least one of:\n  "
            + "\n  ".join(str(p) for p in SEED_SOURCES)
        )
    counts = {lbl: sum(1 for r in rows if r["label"] == lbl) for lbl in LABELS}
    print(f"  total: {len(rows)} rows  balance: {counts}", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Pioneer HTTP client
# ---------------------------------------------------------------------------

@dataclass
class PioneerClient:
    api_key: str
    base: str
    _client: Any = field(default=None, init=False, repr=False)

    def __enter__(self) -> "PioneerClient":
        import httpx
        self._client = httpx.Client(timeout=120.0, headers=self._headers())
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._client is not None:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post(f"{self.base}{path}", json=body)
        if r.status_code >= 300:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text}")
        return r.json() if r.content else {}

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        r = self._client.get(f"{self.base}{path}", params=params)
        if r.status_code >= 300:
            raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text}")
        return r.json() if r.content else {}


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_dataset(
    client: PioneerClient,
    *,
    dataset_name: str,
    num_examples: int,
    domain_description: str,
    skip_generate: bool,
) -> dict[str, Any]:
    """Load seed rows → optionally augment via Pioneer /generate → write JSONL."""
    rows = _load_seed_rows()
    synthetic_job: dict[str, Any] | None = None

    if not skip_generate and num_examples > 0:
        body: dict[str, Any] = {
            "task_type": "classification",
            "dataset_name": dataset_name,
            "num_examples": int(num_examples),
            "labels": LABELS,
            "domain_description": domain_description,
            "classified_examples": [{"text": r["text"], "label": r["label"]} for r in rows],
        }
        print(f"  POST /generate (num_examples={num_examples}) …", file=sys.stderr)
        try:
            created = client.post("/generate", body)
        except Exception as e:  # noqa: BLE001
            print(f"WARN: /generate failed ({e}); seed-only fallback.", file=sys.stderr)
            created = {"error": str(e)}
        job_id = created.get("job_id") or created.get("id")
        synthetic_job = {"request": body, "response": created, "job_id": job_id}
        if job_id:
            poll_step, poll_max = 3.0, 360  # up to 18 min
            elapsed = 0.0
            last_state: str | None = None
            for _ in range(poll_max):
                time.sleep(poll_step)
                elapsed += poll_step
                try:
                    status = client.get(f"/generate/jobs/{job_id}")
                except Exception as e:  # noqa: BLE001
                    print(f"WARN: poll failed: {e}", file=sys.stderr)
                    break
                state = str(status.get("status") or status.get("state") or "")
                synthetic_job["last_status"] = status
                if state and state != last_state:
                    print(f"  /generate state={state} ({int(elapsed)}s)", file=sys.stderr)
                    last_state = state
                if state.lower() in {"success", "succeeded", "completed", "done"}:
                    print("  /generate done.", file=sys.stderr)
                    break
                if state.lower() in {"failed", "error", "cancelled"}:
                    print(f"WARN: /generate terminal ({state!r})", file=sys.stderr)
                    break
            else:
                print(
                    f"WARN: /generate timed out after {int(poll_max * poll_step)}s — "
                    "re-run --phase=train once Pioneer confirms the dataset is ready.",
                    file=sys.stderr,
                )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta: dict[str, Any] = {
        "dataset_name": dataset_name,
        "seed_rows": len(rows),
        "target_synthetic_rows": int(num_examples),
        "labels": LABELS,
        "domain_description": domain_description,
        "local_seed_jsonl": str(DATASET_JSONL),
        "synthetic_job": synthetic_job,
    }
    DATASET_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {DATASET_JSONL.name} ({len(rows)} seed rows)", file=sys.stderr)
    return meta


def phase_train(
    client: PioneerClient,
    *,
    dataset_name: str,
    base_model: str,
    model_name: str,
    nr_epochs: int,
    learning_rate: float,
    training_type: str,
) -> dict[str, Any]:
    body = {
        "model_name": model_name,
        "base_model": base_model,
        "datasets": [{"name": dataset_name}],
        "training_type": training_type,
        "nr_epochs": nr_epochs,
        "learning_rate": learning_rate,
    }
    print(f"  POST /felix/training-jobs base_model={base_model} …", file=sys.stderr)
    resp = client.post("/felix/training-jobs", body)
    job_id = resp.get("id") or resp.get("training_job_id") or resp.get("job_id")
    if not job_id:
        raise RuntimeError(f"Training job id missing: {resp}")
    meta = {"request": body, "response": resp, "training_job_id": job_id,
            "dataset_name": dataset_name}
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  training_job_id={job_id}", file=sys.stderr)
    print(f"  → set PIONEER_TRAINED_MODEL_ID={job_id} in .env after training completes",
          file=sys.stderr)
    return meta


def phase_poll(
    client: PioneerClient, *, training_job_id: str, timeout_seconds: float
) -> dict[str, Any]:
    t0 = time.time()
    last_state: str | None = None
    while time.time() - t0 < timeout_seconds:
        status = client.get(f"/felix/training-jobs/{training_job_id}")
        state = str(status.get("status") or status.get("state") or "")
        if state != last_state:
            print(f"  status={state} ({int(time.time() - t0)}s)", file=sys.stderr)
            last_state = state
        if state.lower() in {"success", "succeeded", "completed", "done"}:
            return status
        if state.lower() in {"failed", "error", "cancelled"}:
            raise RuntimeError(f"Training failed: {state!r}\n{status}")
        time.sleep(10.0)
    raise TimeoutError(f"Training did not complete within {timeout_seconds:.0f}s")


def phase_eval(
    client: PioneerClient, *, training_job_id: str, dataset_name: str
) -> dict[str, Any]:
    body = {"base_model": training_job_id, "dataset_name": dataset_name}
    print(f"  POST /felix/evaluations …", file=sys.stderr)
    resp = client.post("/felix/evaluations", body)
    eval_id = resp.get("id") or resp.get("evaluation_id")
    result = resp
    if eval_id:
        for _ in range(60):
            time.sleep(5.0)
            try:
                result = client.get(f"/felix/evaluations/{eval_id}")
            except Exception as e:  # noqa: BLE001
                print(f"WARN: eval poll: {e}", file=sys.stderr)
                break
            state = str(result.get("status") or result.get("state") or "")
            if state.lower() in {"success", "succeeded", "completed", "done", "failed", "error"}:
                break
    meta = {"request": body, "response": result}
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    f1 = result.get("f1") or result.get("metrics", {}).get("f1")
    print(f"  eval done — F1: {f1}", file=sys.stderr)
    return meta


def phase_infer(
    client: PioneerClient,
    *,
    trained_model_id: str,
    baseline_model_id: str,
    probes: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for text in probes:
        print(f"  probe: {text!r}", file=sys.stderr)
        try:
            baseline = client.post("/inference", {
                "model_id": baseline_model_id, "task": "classify_text",
                "text": text, "schema": {"categories": LABELS},
            })
        except Exception as e:  # noqa: BLE001
            baseline = {"error": str(e)}
        try:
            trained = client.post("/inference", {
                "model_id": trained_model_id, "task": "classify_text",
                "text": text, "schema": {"categories": LABELS},
            })
        except Exception as e:  # noqa: BLE001
            trained = {"error": str(e)}
        rows.append({"text": text, "baseline": baseline, "trained": trained})

    meta = {
        "baseline_model_id": baseline_model_id,
        "trained_model_id": trained_model_id,
        "rows": rows,
    }
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIDEBYSIDE_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {SIDEBYSIDE_META.name}", file=sys.stderr)
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_training_meta() -> dict[str, Any]:
    if not TRAINING_META.is_file():
        return {}
    try:
        return json.loads(TRAINING_META.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Franken-Fit × Pioneer fine-tune loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=("dataset", "train", "poll", "eval", "infer", "all"),
        default="all",
        help="Which phase(s) to run. 'all' runs dataset→train→poll→eval→infer in sequence.",
    )
    parser.add_argument(
        "--dataset-name",
        default=os.environ.get("PIONEER_DATASET_NAME", "frankenfit-binary-v3"),
        help="Pioneer dataset name (used as key on Pioneer's side).",
    )
    parser.add_argument(
        "--num-synthetic", type=int, default=0,
        help="Rows to synthesize via Pioneer /generate (0 = skip, seed-only). "
             "Recommended: keep at 0 when Pioneer's generate queue is slow.",
    )
    parser.add_argument(
        "--domain-description",
        default=(
            "Second-hand and vintage fashion wardrobes: short phrases describing "
            "individual garments (coats, jackets, dresses, tops, bottoms, footwear) "
            "with fabric, era, silhouette and styling cues. Labels express personal "
            "taste as a binary swipe signal: 'love' = would keep or buy, "
            "'hate' = would toss."
        ),
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip Pioneer /generate entirely; use seed rows only (equivalent to --num-synthetic=0).",
    )
    parser.add_argument(
        "--base-model",
        default=os.environ.get("PIONEER_BASE_MODEL", "fastino/gliner2-base-v1"),
        help="GLiNER base for fine-tuning. For --phase=infer, override with "
             "PIONEER_QWEN_MODEL to show Qwen vs fine-tuned side-by-side.",
    )
    parser.add_argument(
        "--model-name", default="frankenfit-binary-lora",
        help="Name for the trained model artifact.",
    )
    parser.add_argument("--training-type", default="lora", choices=("lora", "full"))
    parser.add_argument("--nr-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--training-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--training-job-id", default=None,
        help="Resume a specific training job (skips train phase, goes straight to poll/eval/infer).",
    )
    parser.add_argument(
        "--probes", default=None,
        help="Comma-separated inference probes for --phase=infer. Default: built-in fashion probes.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("PIONEER_API_KEY")
    if not api_key:
        print(
            "SKIP: PIONEER_API_KEY not set. Add it to .env.\n"
            "  Docs: https://agent.pioneer.ai/docs/api-reference",
            file=sys.stderr,
        )
        return 0

    base = os.environ.get("PIONEER_API_BASE", "https://api.pioneer.ai").rstrip("/")
    skip_gen = args.skip_generate or args.num_synthetic == 0

    with PioneerClient(api_key=api_key, base=base) as client:
        existing = _load_training_meta()
        training_job_id = (
            args.training_job_id
            or os.environ.get("PIONEER_TRAINED_MODEL_ID")
            or existing.get("training_job_id")
        )

        ran: list[str] = []

        if args.phase in ("dataset", "all"):
            print("\n== phase: dataset ==", file=sys.stderr)
            phase_dataset(
                client,
                dataset_name=args.dataset_name,
                num_examples=args.num_synthetic,
                domain_description=args.domain_description,
                skip_generate=skip_gen,
            )
            ran.append("dataset")

        if args.phase in ("train", "all"):
            print("\n== phase: train ==", file=sys.stderr)
            meta = phase_train(
                client,
                dataset_name=args.dataset_name,
                base_model=args.base_model,
                model_name=args.model_name,
                nr_epochs=args.nr_epochs,
                learning_rate=args.learning_rate,
                training_type=args.training_type,
            )
            training_job_id = meta["training_job_id"]
            ran.append("train")

        if args.phase in ("poll", "all"):
            print("\n== phase: poll ==", file=sys.stderr)
            if not training_job_id:
                print("FAIL: no training_job_id — run --phase=train first.", file=sys.stderr)
                return 1
            final = phase_poll(
                client,
                training_job_id=training_job_id,
                timeout_seconds=args.training_timeout_seconds,
            )
            existing = _load_training_meta()
            existing["final_status"] = final
            TRAINING_META.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            ran.append("poll")

        if args.phase in ("eval", "all"):
            print("\n== phase: eval ==", file=sys.stderr)
            if not training_job_id:
                print("FAIL: no training_job_id — run --phase=train first.", file=sys.stderr)
                return 1
            phase_eval(client, training_job_id=training_job_id,
                       dataset_name=args.dataset_name)
            ran.append("eval")

        if args.phase in ("infer", "all"):
            print("\n== phase: infer (side-by-side) ==", file=sys.stderr)
            if not training_job_id:
                print("FAIL: no training_job_id — run --phase=train first.", file=sys.stderr)
                return 1
            probes = (
                [p.strip() for p in args.probes.split(",") if p.strip()]
                if args.probes
                else SIDEBYSIDE_PROBES
            )
            phase_infer(
                client,
                trained_model_id=training_job_id,
                baseline_model_id=args.base_model,
                probes=probes,
            )
            ran.append("infer")

    print(f"\nDone. phases ran: {', '.join(ran) or '(none)'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001
        err = str(e).lower()
        if any(x in err for x in ("429", "rate limit", "quota", "resource exhausted")):
            print(f"WARN (quota): {e}", file=sys.stderr)
            raise SystemExit(0)
        print(f"FAIL: {e}", file=sys.stderr)
        raise SystemExit(1)
