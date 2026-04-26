import json

from core.config.app_config import (
    LISTEN_BRANCH_ID,
    SPEAK_BRANCH_ID,
    DEFAULT_CONFIG,
    get_branch_config,
    load_app_config,
    replace_branch_config,
    save_app_config,
)


def test_load_app_config_normalizes_legacy_primary_secondary_branches(tmp_path) -> None:
    config_path = tmp_path / "legacy_config.json"
    config_path.write_text(
        json.dumps(
            {
                "stt": {"language": "en-US"},
                "translation": {"direction": "en_to_ru", "enabled": True},
                "tts": {"voice_name": "ru_RU-dmitri-medium"},
                "branches": {
                    "primary": {
                        "branch_id": "primary",
                        "label": "EN -> RU",
                        "translation_direction": "en_to_ru",
                    },
                    "secondary": {
                        "branch_id": "secondary",
                        "label": "RU -> EN",
                        "translation_direction": "ru_to_en",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert [branch.branch_id for branch in config.branches[:2]] == [LISTEN_BRANCH_ID, SPEAK_BRANCH_ID]
    assert get_branch_config(config, LISTEN_BRANCH_ID).translation_direction == "en_to_ru"
    assert get_branch_config(config, SPEAK_BRANCH_ID).translation_direction == "ru_to_en"


def test_load_app_config_supports_branch_list_format(tmp_path) -> None:
    config_path = tmp_path / "list_config.json"
    config_path.write_text(
        json.dumps(
            {
                "branches": [
                    {
                        "branch_id": "listen",
                        "label": "EN -> RU",
                        "translation_direction": "en_to_ru",
                    },
                    {
                        "branch_id": "speak",
                        "label": "RU -> EN",
                        "translation_direction": "ru_to_en",
                    },
                    {
                        "branch_id": "listen_de",
                        "label": "DE -> RU",
                        "translation_direction": "de_to_ru",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert [branch.branch_id for branch in config.branches] == ["listen", "speak", "listen_de"]


def test_save_app_config_persists_branches_as_list(tmp_path) -> None:
    config_path = tmp_path / "saved_config.json"
    config = replace_branch_config(
        DEFAULT_CONFIG,
        get_branch_config(DEFAULT_CONFIG, LISTEN_BRANCH_ID),
    )

    save_app_config(config, config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert isinstance(payload["branches"], list)
    assert payload["branches"][0]["branch_id"] == LISTEN_BRANCH_ID
