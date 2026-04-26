from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.text import PhraseFragmentReconstructor


def test_reconstructs_final_text_from_noisy_fragments() -> None:
    reconstructor = PhraseFragmentReconstructor()

    result = reconstructor.reconstruct(
        [
            "всем привет",
            "меня зовут Ярослав",
            "мне 20 6 лет",
            "мне 26 лет",
            "я из Украины",
            "я работал в компании I'm days",
            "я работал в компании AMDays",
            "я работал в компании AEMDays",
            "и делал лендинги",
        ]
    )

    assert result.text == (
        "Всем привет, меня зовут Ярослав, мне 26 лет, я из Украины. "
        "Я работал в компании AEMDays и делал лендинги."
    )


def test_keeps_overlap_upgrade_without_creating_duplicate_phrase() -> None:
    reconstructor = PhraseFragmentReconstructor()

    result = reconstructor.reconstruct(
        [
            "я его создал для помощи в преодолении языкового барьера",
            "так как мой английский слабее уровня комфорта",
            "так как мой английский слабее уровня комфортного общения",
        ]
    )

    assert result.phrases == [
        "я его создал для помощи в преодолении языкового барьера",
        "так как мой английский слабее уровня комфортного общения",
    ]


def test_merges_company_name_variants_into_single_phrase_cluster() -> None:
    reconstructor = PhraseFragmentReconstructor()

    result = reconstructor.reconstruct(
        [
            "я работал в компании I'm days",
            "я работал в компании AMDays",
            "я работал в компании AEMDays",
        ]
    )

    assert result.phrases == ["я работал в компании AEMDays"]


def test_avoids_backfill_pattern_for_repeated_intro_fragment() -> None:
    reconstructor = PhraseFragmentReconstructor()

    result = reconstructor.reconstruct(
        [
            "всем привет",
            "меня зовут Ярослав",
            "мне 26 лет",
            "всем привет",
            "я из Украины",
        ]
    )

    assert result.phrases == [
        "всем привет",
        "меня зовут Ярослав",
        "мне 26 лет",
        "я из Украины",
    ]


def test_prefers_refined_company_phrase_over_earlier_shorter_variant() -> None:
    reconstructor = PhraseFragmentReconstructor()

    result = reconstructor.reconstruct(
        [
            "Когда я работал в компании AMD",
            "Когда я работал в компании AMDays я делал лендинги",
            "Очень много лендингов",
        ]
    )

    assert result.phrases == [
        "Когда я работал в компании AMDays я делал лендинги",
        "Очень много лендингов",
    ]


def test_prefers_refined_wording_inside_same_fragment_cluster() -> None:
    reconstructor = PhraseFragmentReconstructor()

    result = reconstructor.reconstruct(
        [
            "но упирался то в лимит",
            "но опирался то в лимит",
            "в отсутствие гибкости",
        ]
    )

    assert result.phrases == [
        "но опирался то в лимит",
        "в отсутствие гибкости",
    ]
