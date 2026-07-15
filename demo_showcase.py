"""
demo_showcase.py -- a guided, live tour of the unified pipeline.

This is not a scripted/faked demo: every answer below comes from actually
running UnifiedAIPipeline.run() (and friends) against the real code in this
repo. It uses its own isolated database (var/demo_showcase/engine.db, wiped
at the start of every run) rather than the shared data/engine.db, so:

  1. it's perfectly reproducible -- run it ten times in a row during an
     interview and you'll see the same four scenes every time, and
  2. it won't leave "taught" facts sitting in your real project database.

Usage:
    python demo_showcase.py                # the four scenes (~a few seconds)
    python demo_showcase.py --run-tests     # scenes, then the full pytest suite
    python demo_showcase.py --pause         # wait for Enter between scenes
                                             # (handy when presenting live)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

from ai_pipeline import UnifiedAIPipeline

WIDTH = 72
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "var", "demo_showcase")
DEMO_DB = os.path.join(DEMO_DIR, "engine.db")


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f" {title}")
    print("=" * WIDTH)


def narrate(text: str) -> None:
    print(f"  >> {text}")
    print()


def show_answer(label: str, result: dict) -> None:
    print(f"[{label}]")
    print(result["final_output"])
    retrieved = result["closed_loop"]["retrieved"]
    if retrieved:
        print(f"  (top retrieval score: {retrieved[0]['score']:.2f}, "
              f"intent: {result['closed_loop']['intent']})")
    else:
        print(f"  (no documents retrieved, intent: {result['closed_loop']['intent']})")
    print()


def pause_if_requested(args: argparse.Namespace) -> None:
    if args.pause:
        input("  -- press Enter to continue --")


def fresh_pipeline() -> UnifiedAIPipeline:
    """A pipeline wired to our own throwaway db, initialized from scratch."""
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)
    pipeline = UnifiedAIPipeline(config_path="config.yaml")
    pipeline.closed_loop.config["paths"]["sqlite_db"] = DEMO_DB
    pipeline.closed_loop.memory.sqlite_path = pipeline.closed_loop.memory._resolve_path(DEMO_DB)
    pipeline.initialize()
    return pipeline


def scene_1_retrieval_grounded(pipeline: UnifiedAIPipeline, args: argparse.Namespace) -> None:
    banner("SCENE 1 / 4 -- Answers are grounded in retrieved memory")
    narrate(
        "The bootstrap dataset has an exact worked example for this question. "
        "The pipeline retrieves it AND the answer below actually uses it -- "
        "it doesn't just retrieve context and then ignore it."
    )
    result = pipeline.run("Solve the integral of 2x from 0 to 4.").to_dict()
    show_answer("Q: Solve the integral of 2x from 0 to 4.", result)
    pause_if_requested(args)


def scene_2_honest_fallback(pipeline: UnifiedAIPipeline, args: argparse.Namespace) -> None:
    banner("SCENE 2 / 4 -- Honest about what it doesn't know")
    narrate(
        "This question has nothing to do with math, code, or engineering, and "
        "there's nothing relevant in memory. Watch the intent classification "
        "and the fact that it gives a plan, not a fabricated answer."
    )
    result = pipeline.run("What's a thoughtful birthday gift for a partner?").to_dict()
    show_answer("Q: What's a thoughtful birthday gift for a partner?", result)
    pause_if_requested(args)


def scene_3_multiturn_trace(pipeline: UnifiedAIPipeline, args: argparse.Namespace) -> None:
    banner("SCENE 3 / 4 -- Multi-turn session, correct per-turn trace")
    narrate(
        "Two turns in one session. Each turn's trace should describe THAT "
        "turn -- not the previous one."
    )
    turns = ["First question about gearbox torque", "Second question about bearing wear"]
    session = pipeline.run_session(turns)
    for i, turn in enumerate(session["turns"]):
        norm_stage = next(s for s in turn["trace"] if s["name"] == "normalize")
        print(f"  turn {i}: input={turn['input_text']!r}")
        print(f"           trace says normalized_text={norm_stage['data']['normalized_text']!r}")
    print()
    pause_if_requested(args)


def scene_4_teaches_and_learns(pipeline: UnifiedAIPipeline, args: argparse.Namespace) -> None:
    banner("SCENE 4 / 4 -- Teaching it something new, live")
    question = "What is the safe working load for a 10mm shaft bearing?"
    answer = "The safe working load for a 10mm shaft bearing is 2.4 kN."

    narrate("First, ask something not in memory yet:")
    before = pipeline.run(question).to_dict()
    show_answer("Q (before teaching)", before)

    narrate(f'Now teach it: teach_from_example(question, "{answer}")')
    pipeline.teach_from_example(question, answer)
    print("  (this both stores the corrected answer in memory AND runs one")
    print("   real training step on FractalBrain's weights)")
    print()

    narrate("Ask the exact same question again:")
    after = pipeline.run(question).to_dict()
    show_answer("Q (after teaching)", after)
    pause_if_requested(args)


def run_test_suite() -> None:
    banner("BONUS -- running the real test suite live (not a canned number)")
    t0 = time.time()
    subprocess.run([sys.executable, "tests/test_smoke.py"], check=False)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)
    print(f"\n  (finished in {time.time() - t0:.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-tests", action="store_true", help="also run the full test suite as a finale")
    parser.add_argument("--pause", action="store_true", help="wait for Enter between scenes (good for live presenting)")
    args = parser.parse_args()

    t0 = time.time()
    banner("HYBRID COGNITIVE AI PIPELINE -- live showcase")
    narrate("Zero external dependencies. Every scene below is a real call into the actual pipeline code.")

    pipeline = fresh_pipeline()
    scene_1_retrieval_grounded(pipeline, args)
    scene_2_honest_fallback(pipeline, args)
    scene_3_multiturn_trace(pipeline, args)
    scene_4_teaches_and_learns(pipeline, args)

    banner("DONE")
    print(f"  Ran all 4 scenes in {time.time() - t0:.2f}s.")
    print(f"  Demo database: {DEMO_DB} (isolated -- your real data/engine.db was not touched)")

    if args.run_tests:
        run_test_suite()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
