"""R1 bare-string entry embeddings: the pooled embedding of the word alone (§A.1).

Encoder-final-layer object by construction, hence layer-independent -- which is
what makes it the fixed target of the layer sweep. All three prefix conventions
are computed; §A.2 step 3 picks one on the *tuning* slice only.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.encoders import Encoder
from dvlsr.util import get_logger, Timer

lg = get_logger("r1", "12_r1.log")


def main(enc_name):
    z = np.load(paths.ART / "vocab.npz")
    words = [str(w) for w in z["words"]]
    enc = Encoder(enc_name)
    out = {}
    variants = {"none": "", "doc": enc.cfg["d_prefix"], "query": enc.cfg["q_prefix"]}
    for tag, pre in variants.items():
        with Timer(f"R1 {enc_name}/{tag}", lg):
            texts = [pre + w for w in words]
            # encode_pooled adds the prefix itself, so bypass it with an empty one
            saved = (enc.cfg["d_prefix"], enc.cfg["q_prefix"])
            enc.cfg["d_prefix"] = ""
            out[tag] = enc.encode_pooled(texts, is_query=False)
            enc.cfg["d_prefix"] = saved[0]
    np.savez(paths.ART / f"r1_{enc_name}.npz", **out)
    lg.info(f"saved R1 {enc_name}: " + str({k: v.shape for k, v in out.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder", default="e5")
    main(ap.parse_args().encoder)
