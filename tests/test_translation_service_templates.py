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
        == "Привет, меня зовут Yaroslav."
    )
