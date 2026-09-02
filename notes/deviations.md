# Deviations from the protocol, and why

Every item here is a place where the written protocol could not be followed
literally, or where a choice it left open had to be fixed. Nothing else deviates.

## D1 — Sense labelling without an LLM annotator (§A.1 polysemy set)
The protocol asks for occurrences "sense-labeled by an LLM with 10% manual
verification". No annotation API is available in this environment, and using the
same encoder under test to label would make the sense metric circular.

**What we do instead.** Each sense carries two disjoint word lists:
* `anchors` — the 5 entries the *scoring* uses (as in the protocol);
* `indicators` — a separate list used only to *label* an occurrence.

An occurrence of word w in a passage is labelled sense A if the passage contains
at least one sense-A indicator and no sense-B indicator (and vice versa);
ambiguous passages are dropped. Because the labels never touch the anchors, the
metric is not scored against the evidence that produced it. Manual verification
of a 10% sample is reported in `results/14_sense_verification.json`.

**Residual bias.** Labels still come from lexical context, and a contextual token
state attends to that same context. The number is therefore an upper bound on
sense separation from context *alone*; it is comparable across encoders, layers
and representations, which is all Pilot A and B use it for.

## D2 — C1 includes the top-100 of *both* dense candidates (§0.2)
C1 is specified with "the dense candidate encoder's top-100". The candidate is
not chosen until Pilot A finishes, so C1 contains both e5-base-v2's and
bge-base-en-v1.5's top-100. This only enlarges C1 and makes it *more* even-handed
towards the references, so the direction of the residual bias described in §0.2
is unchanged.

## D3 — Occurrence sampling unit (§0.3)
Prototype occurrences are sampled as distinct (passage, entry) pairs rather than
as raw token occurrences; a passage that contains the word twice contributes once,
and the entry's state from that passage is the mean over its matching word units.
Sampling is exact bottom-k with uniform random keys, so the sample is uniform
without replacement over those pairs.

## D4 — Retrieval engine for our own method
BM25 and SPLADE++ use Pyserini's prebuilt Lucene indexes (our BM25 reproduces the
published 0.187 MRR@10 exactly). Our own sparse retrieval uses `scipy.sparse` /
torch matmul over C1 as §0.7 specifies, not Lucene.

## D5 — Encoding length
Passages are truncated at 192 wordpieces and queries at 64. MS MARCO passages are
short; this affects a negligible tail and is applied identically to every system
we compare.
