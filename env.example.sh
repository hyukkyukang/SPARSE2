# Copy to env.sh (git-ignored) and source it before any script. Paths are examples.
export DVLSR_DATA=/scratch/$USER/dvlsr          # large artifacts: data/, artifacts/, ckpt/, emb/, domain/
export HF_HOME=$DVLSR_DATA/cache/hf              # model cache (Octen, jina-v5, e5, SPLADE)
export TMPDIR=$DVLSR_DATA/tmp
export DVLSR_RESULTS_DIR=results_gpu10           # small JSON results, committed
export DVLSR_REPORTS_DIR=reports_gpu10           # generated reports, committed
export DVLSR_LOGS_DIR=logs_gpu10                 # per-step logs, git-ignored
export OMP_NUM_THREADS=8; export OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1                          # after the first download; unset it to fetch
export OPENAI_API_KEY=not-used                   # pyserini imports an OpenAI client at import time
# export HF_TOKEN=...                            # only for the gated naver/splade-v3 reference row
# export JAVA_HOME=...                           # only for the BM25/Pyserini stages of the §0 build
mkdir -p $DVLSR_DATA $HF_HOME $TMPDIR
