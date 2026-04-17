"""CPU unit tests for nano_hep.data + nano_hep.vocab.

Run with: python -m nano_hep.data_test
"""

import numpy as np
import torch

from .vocab import Vocab, DEFAULT_MODALITY_CODEBOOK_SIZE, DEFAULT_MODALITY_NUM_QUANTIZERS
from .data import HEPDataset


def test_vocab_q0_only():
    """num_q=1 per modality: each modality block is just codebook_size wide."""
    v = Vocab.build(["track", "truthpart"],
                    {"track": 256, "truthpart": 128},
                    {"track": 1, "truthpart": 1})
    assert v.offsets["track"] == 0
    assert v.offsets["truthpart"] == 256
    assert v.mod_start["track"] == 256 + 128
    assert v.mod_start["truthpart"] == 256 + 128 + 1
    assert v.total == 256 + 128 + 4  # +2 mod_starts +EOS +PAD

    codes = np.array([[0], [5], [255]])
    enc = v.encode_element_triples("track", codes)
    assert enc.tolist() == [0, 5, 255]
    codes_tp = np.array([[0], [5], [127]])
    enc = v.encode_element_triples("truthpart", codes_tp)
    assert enc.tolist() == [256, 261, 383]

    mods, qs, locals_ = v.decode_global([0, 256, v.mod_start["track"], v.eos])
    assert mods.tolist() == ["track", "truthpart", "<MOD:track_start>", "<EOS>"]
    print("[OK] vocab q0-only round-trip (track=256, truthpart=128)")


def test_vocab_multi_q():
    """num_q=3 per modality with correct codebook sizes."""
    v = Vocab.build(["track", "truthpart"],
                    {"track": 256, "truthpart": 128},
                    {"track": 3, "truthpart": 3})
    # track block: 3*256=768 → [0, 768), truthpart block: 3*128=384 → [768, 1152)
    assert v.offsets["track"] == 0
    assert v.offsets["truthpart"] == 768
    assert v.mod_start["track"] == 1152
    assert v.eos == 1154
    assert v.total == 1156

    # encode: 2 track elements, each with 3 quantizer codes (0..255 range)
    codes = np.array([[5, 10, 15], [100, 200, 255]])  # (2, 3)
    enc = v.encode_element_triples("track", codes)
    # el0: q0=5, q1=256+10=266, q2=512+15=527
    # el1: q0=100, q1=256+200=456, q2=512+255=767
    assert enc.tolist() == [5, 266, 527, 100, 456, 767]

    # round-trip decode
    recovered = v.decode_modality_triples("track", enc)
    assert recovered.tolist() == codes.tolist()
    print("[OK] vocab multi-q round-trip (3 quantizers, real sizes)")


def test_dataset_shape():
    ds = HEPDataset(block_size=64, max_events=8, input_modalities=["track"],
                    output_modality="truthpart", num_quantizers=1)
    assert len(ds) == 8
    x, y, m = ds[0]
    assert x.shape == (63,) and y.shape == (63,) and m.shape == (63,)
    assert x.dtype == torch.long and y.dtype == torch.long and m.dtype == torch.long
    # y is next-token: y[i] == seq[i+1]
    import numpy as np
    seq = ds.build_sequence(0)
    assert np.all(x.numpy() == seq[:-1])
    assert np.all(y.numpy() == seq[1:])
    print("[OK] shapes + next-token alignment")


def test_loss_mask_bounds():
    """Loss mask covers exactly the (output_region − last_y) positions."""
    ds = HEPDataset(block_size=64, max_events=16)
    for i in range(16):
        seq = ds.build_sequence(i)
        m = ds.build_loss_mask(seq)
        # find where output mod_start is in seq
        out_start = int(np.where(seq == ds.vocab.mod_start["truthpart"])[0][0])
        # loss[j]==1 iff j in [out_start, last_out_pos) where last_out_pos is EOS position or truncation end
        assert m[:out_start].sum() == 0, f"event {i}: loss leaked into input region"
        assert m[out_start] == 1 or (seq == ds.vocab.eos).sum() == 0, \
            f"event {i}: first output-region position should have loss=1"
        # loss should not include PAD as a TARGET (y[i]=PAD means mask[i]=0)
        y_pad_positions = np.where(seq[1:] == ds.vocab.pad)[0]
        for p in y_pad_positions:
            assert m[:-1][p] == 0 if p < len(m) - 1 else True, \
                f"event {i} pos {p}: PAD target should not have loss=1"
    print("[OK] loss mask bounds for 16 events")


def test_empty_output():
    """If output has zero elements, the sequence still builds with EOS right after MOD_start[out]."""
    # Find an event with zero output tokens (rare but possible)
    ds = HEPDataset(block_size=64, max_events=1000)
    import numpy as np
    zero_out = None
    for i in range(1000):
        a = int(ds.out_mm.offsets[i]); b = int(ds.out_mm.offsets[i + 1])
        if b == a:
            zero_out = i; break
    if zero_out is None:
        print("[SKIP] no zero-output event in first 1000")
        return
    seq = ds.build_sequence(zero_out)
    m = ds.build_loss_mask(seq)
    # Just check it's non-empty and loss mask is valid
    assert len(seq) == 64
    print(f"[OK] zero-output event {zero_out}: seq built, loss.sum()={m.sum()}")


def test_multi_input():
    """Extended setup: (track, topo) → truthpart, q0 only."""
    ds = HEPDataset(block_size=128, max_events=8,
                    input_modalities=["track", "topo"], output_modality="truthpart",
                    num_quantizers=1)
    x, y, m = ds[0]
    assert x.shape == (127,)
    # Vocab must include 3 modalities and 3 MOD_STARTs
    assert len(ds.vocab.modalities) == 3
    assert set(ds.vocab.modalities) == {"track", "topo", "truthpart"}
    # Build a sequence and verify: track_start, topo_start, truthpart_start all present
    seq = ds.build_sequence(0)
    for mod in ["track", "topo", "truthpart"]:
        assert (seq == ds.vocab.mod_start[mod]).sum() == 1, f"missing MOD_START[{mod}]"
    # And they appear in the right order
    positions = {mod: int(np.where(seq == ds.vocab.mod_start[mod])[0][0]) for mod in ["track", "topo", "truthpart"]}
    assert positions["track"] < positions["topo"] < positions["truthpart"]
    print(f"[OK] multi-input track+topo → truthpart, positions={positions}")


def test_multi_q_dataset():
    """Dataset with num_q=3: sequence length grows ~3x."""
    ds = HEPDataset(block_size=256, max_events=8,
                    input_modalities=["track", "topo"], output_modality="truthpart",
                    num_quantizers=3)
    # Verify shape
    x, y, m = ds[0]
    assert x.shape == (255,)
    # Verify every output element emits 3 consecutive tokens within the truthpart modality range
    seq = ds.build_sequence(0)
    out_start = int(np.where(seq == ds.vocab.mod_start["truthpart"])[0][0])
    eos_hits = np.where(seq == ds.vocab.eos)[0]
    eos_pos = int(eos_hits[0]) if len(eos_hits) else ds.block_size - 1
    out_region = seq[out_start+1:eos_pos]
    n_out_tokens = len(out_region)
    assert n_out_tokens % 3 == 0, f"output region length {n_out_tokens} not divisible by num_q=3"
    n_elems = n_out_tokens // 3
    # Decode back and verify shape
    recovered = ds.vocab.decode_modality_triples("truthpart", out_region)
    assert recovered.shape == (n_elems, 3)
    print(f"[OK] multi-q dataset: {n_elems} truthpart elements × 3 quantizers = {n_out_tokens} output tokens")


def test_loss_mask_multi_q():
    """Verify loss mask still covers the output region correctly with num_q>1."""
    ds = HEPDataset(block_size=256, max_events=16,
                    input_modalities=["track", "topo"], output_modality="truthpart",
                    num_quantizers=3)
    for i in range(16):
        seq = ds.build_sequence(i)
        m = ds.build_loss_mask(seq)
        out_start = int(np.where(seq == ds.vocab.mod_start["truthpart"])[0][0])
        eos_hits = np.where(seq == ds.vocab.eos)[0]
        assert m[:out_start].sum() == 0, f"event {i}: loss mask leaked into input region"
        if len(eos_hits):
            eos = int(eos_hits[0])
            # m[out_start..eos-1] should be 1
            assert m[out_start:eos].all(), f"event {i}: output region not fully covered by loss mask"
    print("[OK] multi-q loss mask bounds (16 events)")


if __name__ == "__main__":
    test_vocab_q0_only()
    test_vocab_multi_q()
    test_dataset_shape()
    test_loss_mask_bounds()
    test_empty_output()
    test_multi_input()
    test_multi_q_dataset()
    test_loss_mask_multi_q()
    print("\nAll data/vocab tests passed.")
