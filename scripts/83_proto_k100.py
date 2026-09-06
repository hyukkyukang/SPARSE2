"""k=100 prototypes, free: average the two disjoint 50-occurrence halves already stored.

scripts/11_prototypes.py accumulates occurrence slots 0-49 into protoA and 50-99 into
protoB so Pilot B can measure stability across disjoint samples. protoB is otherwise
unused after that. Averaging the two, weighted by their contribution counts, gives a
k=100 prototype for every entry with no new encoding.

This tests whether the recovery a text-defined vocabulary leaves on the table is
prototype *noise* -- a sampling limit that more occurrences fix -- or an architectural
limit that they do not. Written as proto_{enc}_k100.npz so any stage can take
--vocab-tag-style selection through paths.proto_file(enc, "k100").
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvlsr import paths
from dvlsr.util import get_logger, save_json

lg = get_logger("k100", "83_proto_k100.log")


def main(enc="e5"):
    z = np.load(paths.proto_file(enc))
    A, B = z["protoA"].astype(np.float32), z["protoB"].astype(np.float32)
    cA, cB = z["cntA"].astype(np.float32), z["cntB"].astype(np.float32)
    w = (cA + cB)
    ok = w > 0
    # count-weighted mean of the two halves == the mean over all 100 sampled occurrences
    P = np.zeros_like(A)
    P[ok] = (A[ok] * cA[ok, None, None] + B[ok] * cB[ok, None, None]) / w[ok, None, None]
    P[~ok] = A[~ok]
    out = paths.proto_file(enc, "k100")
    np.savez(out, protoA=P.astype(np.float16), protoB=z["protoB"], cntA=w, cntB=cB,
             layers=z["layers"])
    cos = (A * P).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(P, axis=-1) + 1e-9)
    li = list(z["layers"]).index(12)
    save_json(dict(encoder=enc, n_entries=int(len(w)),
                   mean_count_k50=float(cA.mean()), mean_count_k100=float(w.mean()),
                   entries_with_zero=int((~ok).sum()),
                   mean_cos_k50_to_k100_layer12=float(cos[:, li].mean()),
                   p05_cos_layer12=float(np.percentile(cos[:, li], 5))),
              paths.RESULTS / f"83_proto_k100_{enc}.json")
    lg.info(f"{enc}: k=100 prototypes -> {out} | mean occurrences {cA.mean():.1f} -> {w.mean():.1f} "
            f"| mean cos(k50, k100) at layer 12 = {cos[:, li].mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder", default="e5")
    main(ap.parse_args().encoder)
