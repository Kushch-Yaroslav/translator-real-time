from core.text.repetition import collapse_repeated_spans


def test_collapses_english_sentence_loop_three_times() -> None:
    sentence = "The University of California was founded in 1868."
    text = f"{sentence} {sentence} {sentence}"

    collapsed, changed = collapse_repeated_spans(text)

    assert changed
    assert collapsed == sentence


def test_collapses_nine_word_loop_from_app_logs() -> None:
    unit = "University of California"
    text = " ".join([unit] * 9)

    collapsed, changed = collapse_repeated_spans(text)

    assert changed
    assert collapsed == unit


def test_collapses_russian_case_inflection_loop() -> None:
    # Observed in app logs: same phrase, different case endings.
    text = (
        "В Калифорнийском университете Калифорнийского университета "
        "Калифорнийском университете Калифорнийского университета"
    )

    collapsed, changed = collapse_repeated_spans(text)

    assert changed
    # Only the first copy survives.
    assert collapsed == "В Калифорнийском университете"


def test_collapses_two_word_loop() -> None:
    text = "very good very good very good"

    collapsed, changed = collapse_repeated_spans(text)

    assert changed
    assert collapsed == "very good"


def test_keeps_legitimate_text_with_recurring_bigrams() -> None:
    text = "I like apples and I like oranges and I like bananas"

    collapsed, changed = collapse_repeated_spans(text)

    assert not changed
    assert collapsed == text


def test_keeps_single_word_repetition() -> None:
    # Emphatic single-word repeats are legitimate speech, not a loop.
    text = "I said no, no, no and he agreed"

    collapsed, changed = collapse_repeated_spans(text)

    assert not changed
    assert collapsed == text


def test_keeps_two_copies_of_sentence() -> None:
    # Two occurrences are below the loop threshold (min_copies=3).
    text = "Hello my dear friend, it is a pleasure to see you. Hello my dear friend, it is a pleasure to see you."

    collapsed, changed = collapse_repeated_spans(text)

    assert not changed
    assert collapsed == text


def test_keeps_content_between_loops() -> None:
    text = (
        "The design is almost ready. "
        "Check the budget with the finance team. Check the budget with the finance team. Check the budget with the finance team. "
        "Also invite the engineering lead."
    )

    collapsed, changed = collapse_repeated_spans(text)

    assert changed
    assert collapsed == (
        "The design is almost ready. "
        "Check the budget with the finance team. "
        "Also invite the engineering lead."
    )


def test_punctuation_and_case_insensitive() -> None:
    text = "Thank you! thank you. THANK YOU thank you"

    collapsed, changed = collapse_repeated_spans(text)

    assert changed
    assert collapsed == "Thank you!"


def test_short_text_unchanged() -> None:
    for text in ("", "hello", "hello world", "hello hello hello"):
        collapsed, changed = collapse_repeated_spans(text)
        assert collapsed == text, text
        if text == "hello hello hello":
            assert not changed
