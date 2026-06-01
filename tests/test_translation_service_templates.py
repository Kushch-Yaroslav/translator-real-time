from core.translation.translation_service import TranslationConfig, TranslationDirection, TranslationService


def test_name_intro_template_preserves_cyrillic_name_for_en_to_ru() -> None:
    service = TranslationService.__new__(TranslationService)
    service.config = TranslationConfig(direction=TranslationDirection.EN_TO_RU, enabled=True)

    assert (
        service._translate_name_intro_template("Hello everyone, my name is Ярослав.")
        == "Здравствуйте, меня зовут Ярослав."
    )


def test_name_intro_template_preserves_latin_name_for_en_to_ru() -> None:
    service = TranslationService.__new__(TranslationService)
    service.config = TranslationConfig(direction=TranslationDirection.EN_TO_RU, enabled=True)

    assert (
        service._translate_name_intro_template("Hi, my name is Yaroslav.")
        == "Привет, меня зовут Ярослав."
    )


def test_dialogue_template_maps_kick_in_phrase() -> None:
    service = TranslationService.__new__(TranslationService)
    service.config = TranslationConfig(direction=TranslationDirection.EN_TO_RU, enabled=True)

    assert (
        service._translate_en_ru_dialogue_template("Yeah, but it takes time to kick in.")
        == "Да, но нужно время, чтобы лекарство подействовало."
    )


def test_dialogue_template_maps_check_back_phrase() -> None:
    service = TranslationService.__new__(TranslationService)
    service.config = TranslationConfig(direction=TranslationDirection.EN_TO_RU, enabled=True)

    assert (
        service._translate_en_ru_dialogue_template("I'll check back on you later.")
        == "Я зайду к тебе позже."
    )


def test_normalize_source_text_fixes_kick_in_variant() -> None:
    service = TranslationService.__new__(TranslationService)
    service.config = TranslationConfig(direction=TranslationDirection.EN_TO_RU, enabled=True)

    assert (
        service._normalize_source_text("Yeah, but it affects, takes time to kick in.")
        == "Yeah, but it takes time to kick in."
    )


def test_normalize_source_text_fixes_alyssa_alias() -> None:
    service = TranslationService.__new__(TranslationService)
    service.config = TranslationConfig(direction=TranslationDirection.EN_TO_RU, enabled=True)

    assert service._normalize_source_text("Hi, Arasan.") == "Hi, Alyssa."


def test_dialogue_template_maps_known_name_greeting() -> None:
    service = TranslationService.__new__(TranslationService)
    service.config = TranslationConfig(direction=TranslationDirection.EN_TO_RU, enabled=True)

    assert service._translate_en_ru_dialogue_template("Hi, Alyssa.") == "Привет, Алисса."


def test_dialogue_template_maps_known_name_farewell() -> None:
    service = TranslationService.__new__(TranslationService)
    service.config = TranslationConfig(direction=TranslationDirection.EN_TO_RU, enabled=True)

    assert service._translate_en_ru_dialogue_template("Bye Jeremy.") == "Пока, Джереми."


def test_dialogue_template_translates_founder_ceo_intro() -> None:
    service = TranslationService.__new__(TranslationService)
    service.config = TranslationConfig(direction=TranslationDirection.EN_TO_RU, enabled=True)

    assert (
        service._translate_en_ru_dialogue_template("Hey, I'm Oli, founder and CEO of MicroOne.")
        == "Привет, я Оли, основатель и CEO MicroOne."
    )


def test_postprocess_en_ru_mixed_intro_repairs_untranslated_founder_ceo_words() -> None:
    service = TranslationService.__new__(TranslationService)
    service.config = TranslationConfig(direction=TranslationDirection.EN_TO_RU, enabled=True)

    assert (
        service._postprocess_en_ru_mixed_intro("Привет, Im Oli Founder And Ceo Of Microone.")
        == "Привет, я Оли, основатель и CEO MicroOne."
    )
