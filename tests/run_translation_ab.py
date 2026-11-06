"""A/B translation benchmark: Helsinki opus-mt (current) vs NLLB-200-3.3B.

CPU only (GPU is occupied by the user's work). Measures per-sentence
latency and prints side-by-side quality for manual review.

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 python tests/run_translation_ab.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EN_TO_RU = [
    "Hello, my name is Yaroslav and I am from Ukraine.",
    "I need you to check the budget with the finance team before Monday.",
    "The client wants a final proposal by the end of the week.",
    "Can you invite the engineering lead to the meeting?",
    "She has a lot of questions about the API.",
    "Yesterday I was at the airport and I missed my flight because of the traffic.",
    "We talked about the project last time and you said the design is almost ready.",
    "The weather is great today, we should go for a walk.",
    "I am 26 years old and I work as a software engineer.",
    "University of California is famous for its computer science program.",
    "Please send me the invoice number and the payment details.",
    "What do you think about the new version of the software?",
]

RU_TO_EN = [
    "Привет, меня зовут Ярослав, я из Украины.",
    "Мне нужно проверить бюджет с финансовой командой до понедельника.",
    "Клиент хочет получить итоговое предложение к концу недели.",
    "Погода сегодня отличная, нам стоит пойти на прогулку.",
    "Я работаю программистом уже пять лет.",
    "Можешь пригласить руководителя отдела разработки на встречу?",
    "У меня есть много вопросов по поводу API.",
    "Вчера я был в аэропорту и опоздал на самолёт из-за пробок.",
    "Как тебе новая версия программного обеспечения?",
    "Университет Калифорнии знаменит своей программой по компьютерным наукам.",
]


def bench_marian(direction: str, sentences: list[str]) -> list[dict]:
    from transformers import MarianMTModel, MarianTokenizer

    model_name = "Helsinki-NLP/opus-mt-en-ru" if direction == "en_to_ru" else "Helsinki-NLP/opus-mt-ru-en"
    print(f"\n[marian] loading {model_name} ...")
    t0 = time.perf_counter()
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    print(f"[marian] loaded in {time.perf_counter() - t0:.1f}s")

    results = []
    for text in sentences:
        started = time.perf_counter()
        inputs = tokenizer([text], return_tensors="pt")
        with __import__("torch").no_grad():
            outputs = model.generate(**inputs, num_beams=4, max_length=256)
        out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        results.append({"source": text, "text": out_text, "latency_sec": round(time.perf_counter() - started, 2)})
    del model, tokenizer
    return results


def bench_nllb(direction: str, sentences: list[str]) -> list[dict]:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "facebook/nllb-200-3.3B"
    print(f"\n[nllb] loading {model_name} ...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    print(f"[nllb] loaded in {time.perf_counter() - t0:.1f}s")

    src_lang = "eng_Latn" if direction == "en_to_ru" else "rus_Cyrl"
    tgt_lang = "rus_Cyrl" if direction == "en_to_ru" else "eng_Latn"
    decoder_start = tokenizer.convert_tokens_to_ids(tgt_lang)

    results = []
    for text in sentences:
        started = time.perf_counter()
        inputs = tokenizer(f"{src_lang} {text}", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=128,
                decoder_start_token_id=decoder_start,
                do_sample=False,
            )
        out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        results.append({"source": text, "text": out_text, "latency_sec": round(time.perf_counter() - started, 2)})
    del model, tokenizer
    return results


def report(direction: str, marian: list[dict], nllb: list[dict]) -> None:
    print(f"\n===== {direction.upper()} =====")
    print(f"{'#':<3}{'marian (current)':<58}{'nllb-3.3b':<58}lat(mar/nll)")
    for i, (m, n) in enumerate(zip(marian, nllb), 1):
        print(f"{i:<3}{m['text'][:56]:<58}{n['text'][:56]:<58}{m['latency_sec']:.1f}/{n['latency_sec']:.1f}")
    m_avg = sum(r["latency_sec"] for r in marian) / len(marian)
    n_avg = sum(r["latency_sec"] for r in nllb) / len(nllb)
    print(f"avg latency: marian={m_avg:.2f}s nllb={n_avg:.2f}s")


def main() -> None:
    out: dict = {}

    marian_en = bench_marian("en_to_ru", EN_TO_RU)
    nllb_en = bench_nllb("en_to_ru", EN_TO_RU)
    report("en_to_ru", marian_en, nllb_en)
    out["en_to_ru"] = {"marian": marian_en, "nllb": nllb_en}

    import torch

    torch.cuda.empty_cache()

    marian_ru = bench_marian("ru_to_en", RU_TO_EN)
    nllb_ru = bench_nllb("ru_to_en", RU_TO_EN)
    report("ru_to_en", marian_ru, nllb_ru)
    out["ru_to_en"] = {"marian": marian_ru, "nllb": nllb_ru}

    out_path = PROJECT_ROOT / "recordings" / "benchmarks" / "translation_ab_cpu.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
