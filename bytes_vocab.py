"""byte helpers: every byte-level BPE vocabulary is keyed by a
reversible byte -> printable-codepoint map (GPT-2's `bytes_to_unicode`).

    key("Ġhello") -> b" hello"      not "Ġhello"

`vocab_bytes` turns a tokenizer's vocabulary into {raw bytes -> token id}; that
is the *surface* alphabet, which is what has to be aligned across models.
"""

from __future__ import annotations


def byte_table() -> dict[str, int]:
    """unicode character (as in BPE keys) -> byte value"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_INV = byte_table()


def token_to_bytes(key: str) -> bytes | None:
    try:
        return bytes(_INV[ch] for ch in key)
    except KeyError:
        return None


def vocab_bytes(tok) -> dict[bytes, int]:
    """raw surface bytes -> token id, for every *regular* (non-added) token.

    Added/special tokens are skipped: they are not part of the surface stream
    the fused sampler emits (they get their own pseudo-token channel).
    """
    back = tok.backend_tokenizer
    vocab = back.get_vocab()
    out: dict[bytes, int] = {}
    for s, i in vocab.items():
        if not isinstance(s, str):
            continue
        b = token_to_bytes(s)
        if b is None or b in out:
            continue
        out[b] = int(i)
    return out


# --------------------------------------------------------------------------------------
def _selftest() -> None:
    """The byte table must be exactly GPT-2's, i.e. a bijection 256 <-> 256."""
    tab = byte_table()
    assert len(tab) == 256, len(tab)
    assert len(set(tab.values())) == 256, "not injective"
    assert set(tab.values()) == set(range(256))
    # printable ASCII (except space) maps to itself: offset 0 is deliberately empty
    # so that '!' -> 33.  Space is the GPT-2 quirk: it lives at chr(256), not ' '.
    for ch in "abcXYZ019!{}":
        assert tab[ch] == ord(ch), ch
    # space is byte 32 and is keyed by U+0120 ('Ġ'), the marker every BPE vocabulary
    # uses; U+0100 ('Ā') is byte 0, the START of the 256+n block — easy to confuse.
    assert " " not in tab and tab["\u0120"] == 32 and tab["\u0100"] == 0
    # the 68 bytes that are NOT printable get the 256+n codepoints
    assert max(ord(k) for k in tab) == 256 + 67

    # round trip through token_to_bytes
    assert token_to_bytes("hello") == b"hello"
    assert token_to_bytes("Ġhi") == b" hi"           # Qwen/Llama space marker
    assert token_to_bytes("\u0120hi") == b" hi"      # and the GPT-2 spelling of the same byte
    assert token_to_bytes("Ã©") == b"\xc3\xa9"       # 'é' in UTF-8
    assert token_to_bytes("\u0000nodigit") is None    # not in the table -> None, never raise

    # a 256-byte exhaustiveness check: every byte is reachable
    seen = set()
    for k in tab:
        b = token_to_bytes(k)
        assert b is not None and len(b) == 1
        seen.add(b[0])
    assert seen == set(range(256))

    print("bytes_vocab selftest OK · 256-byte bijection · GPT-2 table · safe on unknown keys")


if __name__ == "__main__":
    _selftest()
