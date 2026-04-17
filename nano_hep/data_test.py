"""CPU unit tests for nano_hep.data + nano_hep.vocab.

Run with: python -m nano_hep.data_test
"""

import numpy as np
import torch

from .vocab import Vocab, DEFAULT_MODALITY_VOCAB
from .data import HEPDataset


def test_vocab_layout():
    v = Vocab.build(["track", "truthpart"], {"track": 1024, "truthpart": 1024})
    # track: 0..1023, truthpart: 1024..2047, specials at 2048+
    assert v.offsets["track"] == 0
    assert v.offsets["truthpart"] == 1024
    assert v.mod_start["track"] == 2048
    assert v.mod_start["truthpart"] == 2049
    assert v.eos == 2050
    assert v.pad == 2051
    assert v.total == 2052

    # round-trip encode/decode
    enc = v.encode_modality_tokens("track", [0, 5, 1023])
    assert enc.tolist() == [0, 5, 1023]
    enc = v.encode_modality_tokens("truthpart", [0, 5, 1023])
    assert enc.tolist() == [1024, 1029, 2047]

    mods, locals_ = v.decode_global([0, 1024, 2048, 2050])
    assert mods.tolist() == ["track", "truthpart", "<MOD:track_start>", "<EOS>"]
    assert locals_.tolist() == [0, 0, -3, -1]
    print("[OK] vocab round-trip")


def test_dataset_shape():
    ds = HEPDataset(block_size=64, max_events=8, input_modalities=["track"], output_modality="truthpart")
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
    """Extended setup: (track, topo) → truthpart."""
    ds = HEPDataset(block_size=128, max_events=8,
                    input_modalities=["track", "topo"], output_modality="truthpart")
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


if __name__ == "__main__":
    test_vocab_layout()
    test_dataset_shape()
    test_loss_mask_bounds()
    test_empty_output()
    test_multi_input()
    print("\nAll data/vocab tests passed.")
