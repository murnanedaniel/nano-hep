"""CPU unit tests for nano_hep.data + nano_hep.vocab.

Run with: python -m nano_hep.data_test
"""

import numpy as np
import torch

from .vocab import (
    Vocab,
    DEFAULT_MODALITY_CODEBOOK_SIZE,
    DEFAULT_MODALITY_NUM_QUANTIZERS,
    DEFAULT_POS_CODEBOOK_SIZE,
    DEFAULT_NUM_Q_POS,
)
from .data import HEPDataset


# Handy helpers: with content-first pos-inclusive layout, per-modality block
# size = num_q_content * V_content + num_q_pos * V_pos.
def _mod_block(v: Vocab, m: str) -> int:
    return v.content_block_size(m) + v.pos_block_size(m)


def test_vocab_q0_only():
    """num_q_content=1 per modality; default pos block still allocated (num_q_pos=3, V_pos=1024)."""
    v = Vocab.build(["track", "truthpart"],
                    {"track": 256, "truthpart": 128},
                    {"track": 1, "truthpart": 1})
    assert v.offsets["track"] == 0
    # pos block for track sits right after content block
    assert v.pos_offsets["track"] == 256
    # truthpart content starts after track's full (content + pos) block
    trk_block = 256 + 3 * 1024
    assert v.offsets["truthpart"] == trk_block
    assert v.pos_offsets["truthpart"] == trk_block + 128
    tp_block = 128 + 3 * 1024
    # mod_starts appended after both full blocks
    assert v.mod_start["track"] == trk_block + tp_block
    assert v.mod_start["truthpart"] == trk_block + tp_block + 1
    assert v.eos == trk_block + tp_block + 2
    assert v.pad == trk_block + tp_block + 3
    assert v.total == trk_block + tp_block + 4

    codes = np.array([[0], [5], [255]])
    enc = v.encode_element_triples("track", codes)
    assert enc.tolist() == [0, 5, 255]

    # decode_global classifies pos tokens correctly
    pos_token_track = v.pos_offsets["track"] + 0  # pos q=0, local=0
    mods, qs, locals_ = v.decode_global([0, pos_token_track, v.mod_start["track"], v.eos])
    assert mods.tolist() == ["track", "track_pos", "<MOD:track_start>", "<EOS>"]
    print("[OK] vocab q0-only with pos blocks")


def test_vocab_multi_q():
    """num_q=3 per modality, content + pos blocks."""
    v = Vocab.build(["track", "truthpart"],
                    {"track": 256, "truthpart": 128},
                    {"track": 3, "truthpart": 3})
    # track: content 3*256=768, pos 3*1024=3072
    assert v.offsets["track"] == 0
    assert v.pos_offsets["track"] == 768
    # truthpart: content 3*128=384, pos 3*1024=3072; starts at 768+3072=3840
    assert v.offsets["truthpart"] == 3840
    assert v.pos_offsets["truthpart"] == 3840 + 384
    mod_starts_base = 3840 + 384 + 3072  # = 7296
    assert v.mod_start["track"] == mod_starts_base
    assert v.eos == mod_starts_base + 2
    assert v.total == mod_starts_base + 4

    # encode (content-only legacy path)
    codes = np.array([[5, 10, 15], [100, 200, 255]])
    enc = v.encode_element_triples("track", codes)
    # el0: q0=5, q1=256+10=266, q2=512+15=527
    # el1: q0=100, q1=456, q2=767
    assert enc.tolist() == [5, 266, 527, 100, 456, 767]
    recovered = v.decode_modality_triples("track", enc)
    assert recovered.tolist() == codes.tolist()
    print("[OK] vocab multi-q content-only round-trip")


def test_vocab_pos_layout():
    """pos_offsets disjoint from content; total arithmetic correct; element
    widths = num_q + num_q_pos."""
    v = Vocab.build(["track", "topo", "truthpart"],
                    {"track": 256, "topo": 256, "truthpart": 128},
                    {"track": 3, "topo": 3, "truthpart": 3})
    # disjointness: each content block's [lo, hi) doesn't overlap any pos block
    for m1 in v.modalities:
        c_lo = v.offsets[m1]; c_hi = c_lo + v.content_block_size(m1)
        for m2 in v.modalities:
            p_lo = v.pos_offsets[m2]; p_hi = p_lo + v.pos_block_size(m2)
            assert c_hi <= p_lo or p_hi <= c_lo, f"content[{m1}] overlaps pos[{m2}]"
    # element_token_width
    for m in v.modalities:
        assert v.element_token_width(m) == v.num_quantizers[m] + v.num_q_pos[m]
    # total = sum_of_all_blocks + n_mods + eos + pad
    total_blocks = sum(v.content_block_size(m) + v.pos_block_size(m) for m in v.modalities)
    assert v.total == total_blocks + len(v.modalities) + 2
    print(f"[OK] vocab pos layout: total={v.total}, disjoint OK")


def test_encode_decode_with_pos():
    """Synthetic (content + pos) round-trip via encode/decode_element_triples_with_pos."""
    v = Vocab.build(["track", "truthpart"],
                    {"track": 256, "truthpart": 128},
                    {"track": 3, "truthpart": 3})
    c = np.array([[5, 10, 20], [100, 50, 30]], dtype=np.int64)     # (2, 3) content
    p = np.array([[100, 500, 900], [1, 2, 3]], dtype=np.int64)      # (2, 3) pos
    enc = v.encode_element_triples_with_pos("track", c, p)
    assert enc.shape == (2 * 6,)
    # First element layout: [5, 266, 532, 768+100, 768+1024+500, 768+2048+900]
    expected_el0 = [5, 256 + 10, 512 + 20, 768 + 100, 768 + 1024 + 500, 768 + 2048 + 900]
    assert enc[:6].tolist() == expected_el0
    c_back, p_back = v.decode_element_triples_with_pos("track", enc)
    assert c_back.tolist() == c.tolist()
    assert p_back.tolist() == p.tolist()
    print("[OK] encode/decode_element_triples_with_pos round-trip")


def test_decode_strict_raises():
    """Strict decoder must reject out-of-range tokens."""
    v = Vocab.build(["track"], {"track": 256}, {"track": 3})
    # 6 tokens; put a pos token into the content slot
    bad = np.array([v.pos_offsets["track"], 1, 2, v.pos_offsets["track"] + 0,
                    v.pos_offsets["track"] + 1024,
                    v.pos_offsets["track"] + 2048], dtype=np.int64)
    try:
        v.decode_element_triples_with_pos("track", bad)
    except AssertionError:
        print("[OK] strict decoder rejects out-of-range tokens")
        return
    raise RuntimeError("expected AssertionError for out-of-range content tokens")


def test_dataset_shape():
    ds = HEPDataset(block_size=128, max_events=8, input_modalities=["track"],
                    output_modality="truthpart", num_quantizers=1, num_q_pos=3)
    assert len(ds) == 8
    x, y, m = ds[0]
    assert x.shape == (127,) and y.shape == (127,) and m.shape == (127,)
    seq = ds.build_sequence(0)
    assert np.all(x.numpy() == seq[:-1])
    assert np.all(y.numpy() == seq[1:])
    print("[OK] shapes + next-token alignment (with pos tokens)")


def test_loss_mask_bounds():
    ds = HEPDataset(block_size=128, max_events=16, num_quantizers=3, num_q_pos=3)
    for i in range(16):
        seq = ds.build_sequence(i)
        m = ds.build_loss_mask(seq)
        out_start = int(np.where(seq == ds.vocab.mod_start["truthpart"])[0][0])
        assert m[:out_start].sum() == 0, f"event {i}: loss leaked into input region"
    print("[OK] loss mask input bounds for 16 events")


def test_multi_input():
    """(track, topo) → truthpart with num_q=1; verify 3 mod_starts in order."""
    ds = HEPDataset(block_size=256, max_events=8,
                    input_modalities=["track", "topo"], output_modality="truthpart",
                    num_quantizers=1, num_q_pos=3)
    assert set(ds.vocab.modalities) == {"track", "topo", "truthpart"}
    seq = ds.build_sequence(0)
    for mod in ["track", "topo", "truthpart"]:
        assert (seq == ds.vocab.mod_start[mod]).sum() == 1, f"missing MOD_START[{mod}]"
    positions = {mod: int(np.where(seq == ds.vocab.mod_start[mod])[0][0]) for mod in ["track", "topo", "truthpart"]}
    assert positions["track"] < positions["topo"] < positions["truthpart"]
    print(f"[OK] multi-input positions={positions}")


def test_multi_q_dataset():
    """Per-element token width = num_q + num_q_pos = 6. Output region divisible by 6."""
    ds = HEPDataset(block_size=512, max_events=8,
                    input_modalities=["track", "topo"], output_modality="truthpart",
                    num_quantizers=3, num_q_pos=3)
    x, y, m = ds[0]
    assert x.shape == (511,)
    seq = ds.build_sequence(0)
    out_start = int(np.where(seq == ds.vocab.mod_start["truthpart"])[0][0])
    eos_hits = np.where(seq == ds.vocab.eos)[0]
    eos_pos = int(eos_hits[0]) if len(eos_hits) else ds.block_size - 1
    out_region = seq[out_start + 1:eos_pos]
    GROUP = ds.vocab.element_token_width("truthpart")
    assert GROUP == 6
    assert len(out_region) % GROUP == 0, \
        f"output region length {len(out_region)} not divisible by GROUP={GROUP}"
    n_elems = len(out_region) // GROUP
    c_back, p_back = ds.vocab.decode_element_triples_with_pos("truthpart", out_region)
    assert c_back.shape == (n_elems, 3)
    assert p_back.shape == (n_elems, 3)
    print(f"[OK] multi-q with pos: {n_elems} truthpart elements × {GROUP} tokens = {len(out_region)}")


def test_build_sequence_with_pos():
    """Round-trip build_sequence ↔ raw memmap for first event; content + pos both match."""
    ds = HEPDataset(block_size=512, max_events=4,
                    input_modalities=["track", "topo"], output_modality="truthpart",
                    num_quantizers=3, num_q_pos=3)
    seq = ds.build_sequence(0)
    out_start = int(np.where(seq == ds.vocab.mod_start["truthpart"])[0][0])
    eos_pos  = int(np.where(seq == ds.vocab.eos)[0][0])
    out_region = seq[out_start + 1:eos_pos]
    c_back, p_back = ds.vocab.decode_element_triples_with_pos("truthpart", out_region)
    # Compare against raw memmap columns for event 0
    a = int(ds.out_mm.offsets[0]); b = int(ds.out_mm.offsets[1])
    raw_c = ds.out_mm.data[a:b, :3].astype(np.int64)
    raw_p = ds.out_mm.data[a:b, 3:6].astype(np.int64)
    n = c_back.shape[0]
    assert n == raw_c.shape[0], f"decoded N={n} vs raw N={raw_c.shape[0]}"
    assert np.array_equal(c_back, raw_c[:n]), "content round-trip mismatch"
    assert np.array_equal(p_back, raw_p[:n]), "pos round-trip mismatch"
    print(f"[OK] build_sequence round-trip: N={n}, content+pos match raw memmap")


def test_loss_mask_covers_pos():
    """Loss mask is 1 on pos-token positions inside the output region."""
    ds = HEPDataset(block_size=512, max_events=4,
                    input_modalities=["track", "topo"], output_modality="truthpart",
                    num_quantizers=3, num_q_pos=3)
    seq = ds.build_sequence(0)
    m = ds.build_loss_mask(seq)
    out_start = int(np.where(seq == ds.vocab.mod_start["truthpart"])[0][0])
    eos_pos   = int(np.where(seq == ds.vocab.eos)[0][0])
    # Find positions in the output region whose token id falls in the truthpart pos range
    p_lo = ds.vocab.pos_offsets["truthpart"]
    p_hi = p_lo + ds.vocab.pos_block_size("truthpart")
    pos_positions = [i for i in range(out_start, eos_pos) if p_lo <= seq[i] < p_hi]
    assert len(pos_positions) > 0, "expected some pos tokens in output region"
    assert all(m[i] == 1 for i in pos_positions), \
        "not all pos-token positions have loss_mask=1"
    print(f"[OK] loss mask covers {len(pos_positions)} pos-token positions")


def test_event_pos_codes_columns():
    """event_pos_codes slices [n_codebooks : n_codebooks+num_q_pos], NOT [:num_q_pos]."""
    ds = HEPDataset(block_size=128, max_events=1, num_quantizers=3, num_q_pos=3)
    mm = ds.out_mm
    from_fn = mm.event_pos_codes(0, 3)
    from_raw = np.asarray(mm.data[int(mm.offsets[0]):int(mm.offsets[1]),
                                  mm.n_codebooks:mm.n_codebooks + 3],
                          dtype=np.int64)
    assert np.array_equal(from_fn, from_raw), "event_pos_codes sliced wrong columns!"
    # Also confirm it's DIFFERENT from event_codes (content)
    from_content = mm.event_codes(0, 3)
    assert not np.array_equal(from_fn, from_content), \
        "event_pos_codes returned content (same as event_codes) — column offset bug"
    print("[OK] event_pos_codes slices the pos columns (not content)")


if __name__ == "__main__":
    test_vocab_q0_only()
    test_vocab_multi_q()
    test_vocab_pos_layout()
    test_encode_decode_with_pos()
    test_decode_strict_raises()
    test_dataset_shape()
    test_loss_mask_bounds()
    test_multi_input()
    test_multi_q_dataset()
    test_build_sequence_with_pos()
    test_loss_mask_covers_pos()
    test_event_pos_codes_columns()
    print("\nAll data/vocab tests passed.")
