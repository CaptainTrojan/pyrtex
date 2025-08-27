"""Example 09: Asynchronous / Multi-Process Workflow (Serialize & Reconnect)

This example demonstrates how to break the classic synchronous chain:

    job.submit().wait().results()

into three independent stages that can be executed by different processes,
containers or scheduled jobs.

Stage A (Initiator / Submitter):
    * Creates the Job, adds requests, submits.
    * Immediately serializes the job state to a JSON file (default: job_state.json).
    * Does NOT call .wait() or .results(). Then exits quickly.

Stage B (Poller):
    * Reconnects using the saved JSON state.
    * Checks status (.status or .is_done) and exits if still running.

Stage C (Collector):
    * Reconnects again using the same JSON state once job is complete.
    * Streams structured results with schema-aware parsing.

You can combine Poller & Collector logic; they are conceptually separate for clarity.

Run (simulation mode - no GCP needed):
    python 09_async_reconnect.py start --simulate
    python 09_async_reconnect.py status --state job_state.json --simulate
    python 09_async_reconnect.py results --state job_state.json --simulate

Run (real mode - requires GCP project + auth):
    export GOOGLE_PROJECT_ID=your-project
    gcloud auth application-default login
    python 09_async_reconnect.py start
    # ... later (cron / another machine) ...
    python 09_async_reconnect.py status --state job_state.json
    # ... later when finished ...
    python 09_async_reconnect.py results --state job_state.json

Notes:
  * Reconnected jobs are "read-only" — you CANNOT add new requests.
  * Persist the JSON however you like (database, GCS, message queue, etc.).
  * The instance map embedded in the state preserves per-request output schemas.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from pyrtex.client import Job


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class BasicInput(BaseModel):
    text: str = Field(description="Input text to echo back")


class BasicOutput(BaseModel):
    result: str = Field(description="Echoed text")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
STATE_FILE_DEFAULT = Path("job_state.json")


def write_state(state_json: str, path: Path):
    path.write_text(state_json, encoding="utf-8")
    print(f"[state] wrote serialized job state to {path}")


def read_state(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def start(args):
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=BasicOutput,
        prompt_template="Return exactly the text '{{ text }}' in field 'result'.",
        simulation_mode=args.simulate,
    )

    # Add a few requests
    job.add_request("first", BasicInput(text="alpha"))
    job.add_request("second", BasicInput(text="beta"))

    job.submit()  # Non-blocking: we intentionally do NOT call .wait()

    state_json = job.serialize()
    write_state(state_json, args.state)
    print("[start] Job submitted and state persisted. Exit now.")


def status(args):
    state_json = read_state(args.state)
    job = Job.reconnect_from_state(state_json)
    st = job.status  # triggers refresh
    print(f"[status] Job status: {st}")
    if job.is_done:
        print("[status] Job finished. Run 'results' command to fetch outputs.")
    else:
        print("[status] Job still in progress. Poll again later.")


def results(args):
    state_json = read_state(args.state)
    job = Job.reconnect_from_state(state_json)

    # Optional polling (short) if you want to wait a bit here for convenience
    if not job.is_done and args.wait:
        print(f"[results] Waiting up to {args.wait}s for completion...")
        deadline = time.time() + args.wait
        while not job.is_done and time.time() < deadline:
            time.sleep(min(10, max(1, args.wait / 10)))
        if not job.is_done:
            print("[results] Still not done; exiting.")
            return

    if not job.is_done:
        print("[results] Job not finished yet. Use the status command.")
        return

    print("[results] Streaming structured results:")
    for r in job.results():
        if r.was_successful:
            print(f" - {r.request_key}: {r.output.result}")
        else:
            print(f" - {r.request_key}: ERROR -> {r.error}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Async submit / reconnect example")
    sub = p.add_subparsers(dest="command", required=True)

    # start
    sp_start = sub.add_parser("start", help="Submit job and persist state")
    sp_start.add_argument(
        "--state", type=Path, default=STATE_FILE_DEFAULT, help="state JSON"
    )
    sp_start.add_argument(
        "--simulate", action="store_true", help="Run in simulation_mode (no GCP)"
    )
    sp_start.set_defaults(func=start)

    # status
    sp_status = sub.add_parser("status", help="Reconnect and print status")
    sp_status.add_argument(
        "--state", type=Path, default=STATE_FILE_DEFAULT, help="saved state JSON"
    )
    sp_status.add_argument(
        "--simulate", action="store_true", help="(Unused here) for symmetry"
    )
    sp_status.set_defaults(func=status)

    # results
    sp_results = sub.add_parser(
        "results", help="Reconnect and stream results (optionally wait briefly)"
    )
    sp_results.add_argument(
        "--state", type=Path, default=STATE_FILE_DEFAULT, help="saved state JSON"
    )
    sp_results.add_argument(
        "--wait", type=int, default=0, help="Optional seconds to wait for completion"
    )
    sp_results.add_argument(
        "--simulate", action="store_true", help="(Unused here) for symmetry"
    )
    sp_results.set_defaults(func=results)

    return p


def main(argv: Optional[list[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - manual run example
    main()
