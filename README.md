# Ampixa TurnBench entry: causal VAD + timing rules

A causal end-of-turn (EOT) and interruption (INT) detector for
[TurnBench](https://turnbench.sesame.com/). Per channel it runs a streaming Silero VAD and a fixed-scale energy
detector, then applies explicit timing policies. No model is trained, and no audio after the emission time is used.

## Results

Scored with the pinned TurnBench scorer
([38a6f874](https://github.com/SesameAILabs/turnbench/tree/38a6f874322430cb3ca71d8a52aa1e636e88bad8)) on the dev split
[mundo-ai/turn-benchmark-dev@8fa18a24](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev/tree/8fa18a24be51528a45397b35cbcaecd84202062b).

| Track | Held-out dev recall (mean of 1,000 split-halves) | FPR | Full-dev recall (in-sample) | FPR |
|---|---:|---:|---:|---:|
| EOT | 0.983 | 0.092 | 0.986 (1878/1904) | 0.090 |
| INT | 0.986 | 0.096 | 0.991 (344/347) | 0.090 |

Held-out: parameters chosen on 19 random dev conversations, scored on the other 19, repeated 1,000 times
(seed 731). Full-dev parameters are chosen on all 38 conversations and are what `parameters.json` contains.
Test predictions were produced once with this exact code; no test metric was computed by us.

## How it works

- **Frontend.** 48 kHz audio is low-passed with a causal 63-tap FIR (7.2 kHz cutoff) and decimated to 16 kHz.
  Silero VAD 6.2.3 runs in true streaming mode (state carried, 64 samples of context, one probability per
  512 samples = 32 ms). Energy probability is `rms / (rms + 0.01)` per frame, with no fitted statistics.
- **Units.** A speech unit starts after 2 frames above threshold and ends after a hangover below
  `threshold - drop`. All decisions use completed frames only.
- **EOT policy.** After a unit ends, wait at least `min_wait`; commit at the first of partner speech for
  `partner_duration`, silence reaching `deadline`, or a floor-held override; cancel if the speaker resumes quickly.
  A second confirmation emission follows at `confirmation` seconds.
- **INT policy.** On the energy channel, an onset while the partner holds the floor commits after a duration
  fallback.

Parameter vectors are in `parameters.json` (field order documented in the `eot_events` and `int_events` docstrings).
`FROZEN.json` records the selection results and the code and parameter SHA-256 values that `inference.py` checks
before it runs.

## Run

```bash
pip install numpy==2.2.6 scipy==1.18.1 numba==0.61.2 onnxruntime==1.32.0 pyarrow==26.0.1 soundfile==0.14.0
pip download --no-deps silero-vad==6.2.3 && unzip -o -j silero_vad-6.2.3-py3-none-any.whl silero_vad/data/silero_vad.onnx
# sha256 must be 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
python inference.py --split dev --data <dir with the 3 dev parquet shards> \
  --model silero_vad.onnx --output predictions-dev.json
```

The script reads only `conversation_id`, `speaker_1_audio` and `speaker_2_audio` from the parquet files, so
annotation columns are never loaded. It refuses to overwrite an existing output, and for `--split test` it
writes a `TEST-STARTED.json` marker so that a second test run fails. Runtime on CPU: about 23 minutes for
the 116 test conversations on 8 vCPUs.

## Limits

- Latency is not ranked by TurnBench. Median emission latency is 657 ms (EOT) and 855 ms (INT), too slow for a
  live voice agent.
- The EOT policy uses the partner's speech onset as evidence. In a deployed agent the partner is the agent
  itself, so this is not single-channel endpointing.
- TurnBench counts at most one false positive per annotated negative span and ignores emissions outside
  scored regions; precision and false alarms per hour are not measured.

## Data and licences

Only the TurnBench dev split (Mundo AI, Dataset Public License v1.0: non-commercial, attribution, no voice cloning)
was used to choose the parameters. No dataset content is in this repository. Code: MIT. Silero VAD: MIT.
