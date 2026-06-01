import time

import numpy as np

from core.audio.audio_session import AudioSession, AudioSessionConfig, PlaybackAudioBlock


def _block(item_id: str, source_type: str, block_index: int = 0, total_blocks: int = 1) -> PlaybackAudioBlock:
    return PlaybackAudioBlock(
        audio=np.zeros((1024, 1), dtype=np.float32),
        item_id=item_id,
        text=f"text {item_id}",
        source_type=source_type,
        created_at=time.monotonic(),
        block_index=block_index,
        total_blocks=total_blocks,
    )


def test_stale_cleanup_preserves_final_items_and_drops_queued_partials() -> None:
    session = AudioSession(AudioSessionConfig(input_device_index=0, output_device_index=0))
    skipped: list[tuple[str, str]] = []
    session.on_playback_skipped = lambda text, reason: skipped.append((text, reason))

    session.output_queue.put_nowait(_block("final-1", "final"))
    session.output_queue.put_nowait(_block("partial-1", "partial"))
    session.output_queue.put_nowait(_block("final-2", "final"))

    cleared = session.clear_output_queue(
        reason="stale_tts_audio_queue",
        preserve_final=True,
        preserve_playing=True,
    )

    assert cleared.cleared_blocks == 1
    assert cleared.skipped_items == 1
    assert cleared.preserved_blocks == 2
    assert [session.output_queue.get_nowait().item_id for _ in range(2)] == [
        "final-1",
        "final-2",
    ]
    assert skipped[0][1] == "stale_tts_audio_queue"
    assert "sourceType=partial" in skipped[0][0]


def test_stale_cleanup_preserves_remaining_blocks_for_playing_item() -> None:
    session = AudioSession(AudioSessionConfig(input_device_index=0, output_device_index=0))
    session._playing_item_id = "playing-final"

    session.output_queue.put_nowait(_block("playing-final", "final", block_index=1, total_blocks=3))
    session.output_queue.put_nowait(_block("partial-1", "partial"))

    cleared = session.clear_output_queue(
        reason="stale_tts_audio_queue",
        preserve_final=False,
        preserve_playing=True,
    )

    assert cleared.cleared_blocks == 1
    assert cleared.preserved_blocks == 1
    assert session.output_queue.get_nowait().item_id == "playing-final"
