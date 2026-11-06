"""End-to-end EN=>RU pipeline benchmark (STT -> dedup -> MT -> dedup).

Mirrors the production EN=>RU (listen) branch on CPU:

  wav -> whisper-server (large-v3-turbo, temp 0.0 / inc 0.2, prompt)
     -> collapse_repeated_spans (source)
     -> Helsinki opus-mt-en-ru (beam 4)
     -> collapse_repeated_spans (translated)

Env knobs for before/after comparison:
  E2E_MODEL      ggml model file (default ggml-large-v3-turbo.bin)
  E2E_TEMP_INC   temperature fallback (default 0.2; "0" disables)
  E2E_DEDUP      on|off (default on)

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 python tests/run_en_to_ru_e2e.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT_LOCAL = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_LOCAL) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_LOCAL))

from core.text.repetition import collapse_repeated_spans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = PROJECT_ROOT / "recordings" / "benchmarks" / "en_to_ru" / "source"
REF_DIR = PROJECT_ROOT / "recordings" / "benchmarks" / "en_to_ru" / "ru_reference"
WHISPER_SERVER = PROJECT_ROOT / "tmp" / "whisper.cpp" / "build" / "bin" / "whisper-server"
MODEL_DIR = Path("/media/yaroslav/DATA/ai_models/whisper.cpp/models")
PORT = 8210
BASE = f"http://127.0.0.1:{PORT}"

MODEL = os.environ.get("E2E_MODEL", "ggml-large-v3-turbo.bin")
TEMPERATURE_INC = float(os.environ.get("E2E_TEMP_INC", "0.2"))
DEDUP = os.environ.get("E2E_DEDUP", "on").lower() != "off"
PROMPT = (
    "Yaroslav. Jaroslav. Ukraine. 26 years old. "
    "Hello everyone, my name is Yaroslav."
)


def wer(reference: str, hypothesis: str) -> float:
    def tokens(text: str) -> list[str]:
        for punct in ".,!?;:()[]\"'-":
            text = text.replace(punct, " ")
        return [t for t in text.lower().split() if t]

    ref, hyp = tokens(reference), tokens(hypothesis)
    if not ref:
        return 0.0
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[0, :] = np.arange(len(hyp) + 1)
    d[:, 0] = np.arange(len(ref) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(ref), len(hyp)]) / len(ref)


def wait_ready(timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            requests.get(f"{BASE}/health", timeout=2.0)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not become ready")


def stt(wav_path: Path) -> tuple[str, float]:
    with open(wav_path, "rb") as f:
        started = time.perf_counter()
        resp = requests.post(
            f"{BASE}/inference",
            files={
                "file": (wav_path.name, f, "audio/wav"),
                "language": (None, "en"),
                "response_format": (None, "json"),
                "temperature": (None, "0.0"),
                "temperature_inc": (None, f"{TEMPERATURE_INC:.1f}"),
                "no_timestamps": (None, "true"),
                "suppress_nst": (None, "true"),
                "prompt": (None, PROMPT),
            },
            timeout=300,
        )
        elapsed = time.perf_counter() - started
    resp.raise_for_status()
    return (resp.json().get("text") or "").strip(), elapsed


def main() -> None:
    proc = subprocess.Popen(
        [
            str(WHISPER_SERVER),
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "-m", str(MODEL_DIR / MODEL),
            "-l", "en",
            "-nt",
            "-sns",
            "-ng",
            "-t", "16",
            "--prompt", PROMPT,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    results: list[dict] = []
    try:
        wait_ready()

        from transformers import MarianMTModel, MarianTokenizer

        print("[mt] loading opus-mt-en-ru ...")
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
        model.eval()

        import torch

        def translate(text: str) -> tuple[str, float]:
            started = time.perf_counter()
            inputs = tokenizer([text], return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, num_beams=4, max_length=256)
            out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return out, time.perf_counter() - started

        for test_dir in sorted(BENCH_DIR.iterdir()):
            if not test_dir.is_dir():
                continue
            for wav in sorted(test_dir.glob("*.wav")):
                test_id = f"{test_dir.name}/{wav.stem}"
                expected_en = (wav.with_suffix(".expected.txt")).read_text(encoding="utf-8").strip()
                ref_ru = (REF_DIR / test_dir.name / f"{wav.stem}.txt").read_text(encoding="utf-8").strip()

                en_text, stt_sec = stt(wav)

                src_loops = 0
                if DEDUP:
                    en_text, src_loops = collapse_repeated_spans(en_text)

                ru_text, mt_sec = translate(en_text)

                tr_loops = 0
                if DEDUP:
                    ru_text, tr_loops = collapse_repeated_spans(ru_text)

                row = {
                    "test": test_id,
                    "en_wer": round(wer(expected_en, en_text), 4),
                    "ru_wer": round(wer(ref_ru, ru_text), 4),
                    "source_loops_removed": int(src_loops),
                    "translated_loops_removed": int(tr_loops),
                    "stt_sec": round(stt_sec, 2),
                    "mt_sec": round(mt_sec, 2),
                    "en": en_text,
                    "ru": ru_text,
                }
                results.append(row)
                print(
                    f"{test_id}: en_wer={row['en_wer']} ru_wer={row['ru_wer']} "
                    f"loops={src_loops}/{tr_loops} t={stt_sec:.1f}/{mt_sec:.1f}"
                )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def agg(key: str) -> tuple[float, float]:
        vals = [r[key] for r in results]
        return round(float(np.mean(vals)), 4), round(float(np.max(vals)), 4)

    en_mean, en_max = agg("en_wer")
    ru_mean, ru_max = agg("ru_wer")
    stt_mean, stt_max = agg("stt_sec")
    mt_mean, mt_max = agg("mt_sec")
    loops_src = sum(r["source_loops_removed"] for r in results)
    loops_tr = sum(r["translated_loops_removed"] for r in results)

    summary = {
        "model": MODEL,
        "temperature_inc": TEMPERATURE_INC,
        "dedup": DEDUP,
        "n": len(results),
        "en_wer_mean": en_mean,
        "en_wer_max": en_max,
        "ru_wer_mean": ru_mean,
        "ru_wer_max": ru_max,
        "stt_mean_sec": stt_mean,
        "stt_max_sec": stt_max,
        "mt_mean_sec": mt_mean,
        "mt_max_sec": mt_max,
        "source_loops_removed": loops_src,
        "translated_loops_removed": loops_tr,
    }
    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    tag = f"{MODEL.replace('ggml-', '').replace('.bin', '')}_inc{TEMPERATURE_INC:.1f}_dedup{'on' if DEDUP else 'off'}"
    out_path = PROJECT_ROOT / "recordings" / "benchmarks" / "en_to_ru" / f"e2e_{tag}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
