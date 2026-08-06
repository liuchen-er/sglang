import pytest

from sglang.srt.managers.adaptive_chunk_policy import (
    AdaptiveChunkConfig,
    AdaptiveChunkPolicy,
)


@pytest.fixture
def policy():
    config = AdaptiveChunkConfig(
        min_chunk=512,
        small_chunk=1024,
        medium_chunk=2048,
        max_chunk=4096,
        decode_low_watermark=4,
        decode_high_watermark=12,
        prefill_queue_high_watermark=8,
    )

    return AdaptiveChunkPolicy(
        config,
        page_size=16,
        launch_chunk_size=4096,
    )


def test_no_decode_uses_max_chunk(policy):
    assert policy.choose(
        decode_batch_size=0,
        waiting_prefill_size=0,
    ) == 4096


def test_low_decode_load_uses_medium_chunk(policy):
    assert policy.choose(
        decode_batch_size=4,
        waiting_prefill_size=0,
    ) == 2048


def test_medium_decode_load_uses_small_chunk(policy):
    assert policy.choose(
        decode_batch_size=8,
        waiting_prefill_size=0,
    ) == 1024


def test_high_decode_load_uses_min_chunk(policy):
    assert policy.choose(
        decode_batch_size=16,
        waiting_prefill_size=0,
    ) == 512


def test_prefill_pressure_increases_chunk(policy):
    assert policy.choose(
        decode_batch_size=16,
        waiting_prefill_size=8,
    ) == 1024


def test_pressure_does_not_exceed_max(policy):
    assert policy.choose(
        decode_batch_size=0,
        waiting_prefill_size=100,
    ) == 4096


def test_rejects_chunk_larger_than_launch_limit():
    config = AdaptiveChunkConfig(max_chunk=8192)

    with pytest.raises(ValueError):
        AdaptiveChunkPolicy(
            config,
            page_size=16,
            launch_chunk_size=4096,
        )


def test_rejects_non_page_aligned_chunk():
    config = AdaptiveChunkConfig(
        min_chunk=500,
        small_chunk=1024,
        medium_chunk=2048,
        max_chunk=4096,
    )

    with pytest.raises(ValueError):
        AdaptiveChunkPolicy(
            config,
            page_size=16,
            launch_chunk_size=4096,
        )


def test_rejects_negative_state(policy):
    with pytest.raises(ValueError):
        policy.choose(
            decode_batch_size=-1,
            waiting_prefill_size=0,
        )