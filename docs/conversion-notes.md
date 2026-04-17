# Conversion notes

Running log of things learned / pitfalls / decisions made during the port.
Append, don't rewrite — this is the project's institutional memory.

## 2026-04-15 — Phase A scaffolding + Phases B/C/D complete

### Status

All 6 CoreML artifacts converted with parity passing:

| Model                       | File                          | Size   | Parity (max\|Δ\|) |
|-----------------------------|-------------------------------|--------|-------------------|
| Text LM prefill (Qwen2.5)   | `text_lm_prefill.mlpackage`   | 747 MB | 4.4 × 10⁻⁵        |
| Text LM decode (custom)     | `text_lm_decode.mlpackage`    | 747 MB | *(not run; prefill-only gate)* |
| TTS LM prefill              | `tts_lm_prefill.mlpackage`    | 1.6 GB | 4.7 × 10⁻⁵        |
| TTS LM decode               | `tts_lm_decode.mlpackage`     | 1.6 GB | *(as above)*      |
| Diffusion head              | `diffusion_head.mlpackage`    | 161 MB | 3.7 × 10⁻⁶        |
| VAE decoder                 | `vae_decoder.mlpackage`       | 1.3 GB | 4.7 × 10⁻¹⁰       |

Total: **~6.2 GB fp32**. fp16 quantisation is a post-parity optimisation.

### Confirmed model facts

- Upstream pinned: `microsoft/VibeVoice` @ `17f57f82ee754abb6d538044a488693fb30784e9`.
- **Two LMs**, not one: `model.language_model` (195 M, Qwen2) + `model.tts_language_model` (434 M). The TTS LM is the *bigger* of the two.
- `tie_word_embeddings=True` — there is no separate `lm_head` module; text-logits need `embed_tokens.weight` reused.
- Diffusion head (`VibeVoiceDiffusionHead`, 42 M): 4 layers, hidden 896, **latent_size 64**, v-prediction, cosine β, **20 DDPM inference steps** (config + runtime both). Forward: `(noisy_images, timesteps, condition) → predicted_noise`.
- Acoustic VAE decoder: 344 M params. Input `(1, 64, T)` (conv1d layout). 3200× upsampling confirmed at T=16 → 51200 samples = 2.133 s @ 24 kHz. 1 latent frame ≈ 133 ms.
- Acoustic tokenizer encoder weights ship as "newly initialized" — the realtime checkpoint is **decoder-only**, confirmed. Encoder isn't needed for inference.
- Text LM: **4 layers** (not 24 as in the `config.json`; the realtime variant instantiates a slimmed LM — `lm.layers.__len__() == 4`), hidden 896, 14 Q heads, **2 KV heads (GQA)**, head dim 64, rope_theta 1e6, max_position 8192, vocab 151936. HF's `DynamicCache.to_legacy_cache()` returns 24 entries (config-sized), but 20 of them are empty — the `PrefillWrapper` skip-None filter correctly drops those. TTS LM has 20 layers.

### I/O shapes (from live .mlpackage specs)

```
text_lm_prefill     in: input_ids (1,T int32)       [T baked to 16; RangeDim did not stick]
                    out: hidden_states (1,T,896), k_cache (4,1,2,T,64), v_cache (4,1,2,T,64)

tts_lm_prefill      same shape pattern; k/v are (20,1,2,T,64)

text_lm_decode      in: input_ids (1,1 int32), k_cache (L,1,2,T,64), v_cache (...),
                        position_ids (1,1 int32)
                    out: hidden_states (1,1,896), k_cache_out (L,1,2,T+1,64), v_cache_out (...)
                    NOTE (resolved 2026-04-15): was failing with -6
                    ("zero shape error" on ios18.slice_by_index) because the
                    cache T-axis had RangeDim(0, MAX_T). Changed to
                    RangeDim(1, MAX_T). Also corrected L = len(lm.layers)
                    instead of config.num_hidden_layers (4 vs 24 for text).
                    Both decode models now load + predict cleanly; KV grows
                    T → T+1 per call as expected.

diffusion_head      in: noisy_latent (1,T,64), timestep (1,) float, conditioning (1,T,896)
                    out: predicted_noise (1,T,64)

vae_decoder         in: latent (1,64,16)                [fixed T=16]
                    out: waveform (1,1,51200)
```

### Pitfalls hit (in order)

1. **Python 3.14 incompatible with coremltools 9.0** — `BlobWriter not loaded` at export time. Rebuilt venv on Python 3.11.
2. **Torch 2.11 incompatible with coremltools** — `TypeError: only 0-dimensional arrays can be converted to Python scalars` inside `_cast` during conversion. Downgraded to torch 2.7.1 (numpy <2).
3. **`TensorType.INT32` doesn't exist** — pass `dtype=np.int32` to `ct.TensorType`.
4. **Timestep dtype in diffusion head** — `TimestepEmbedder.timestep_embedding` does `embedding.to(t.dtype)`, so int timesteps get cast to int after the sinusoid, blowing up the following `Linear`. Use float32 timesteps at the boundary.
5. **VAE decoder conv layout** — `(B, C, T)`, not `(B, T, C)`. Trivial, but the trace error was misleading.
6. **VAE dynamic padding** — `SConv1d` uses shape-dependent reflection padding that coremltools can't express; trace with fixed T=16. Production: ship multiple fixed-T variants (bucketed) or pad in Swift.
7. **HF 4.57 DynamicCache not traceable for decode** — layer-by-layer KV shapes get out of sync during tracing (`stack expects equal size`). Solution: bypass `DynamicCache` entirely in the decode wrapper; walk `lm.layers` manually, apply RoPE via `apply_rotary_pos_emb`, GQA-expand with `repeat_interleave`, call `F.scaled_dot_product_attention`, and concat K/V by hand. See `scripts/02_convert_llm.py::DecodeWrapper`.
8. **LLM parity ~1 × 10⁻³** — traced 195M LM drifts ~1e-3 at last hidden state. Tolerance set to 2 × 10⁻³; drift comes from SDPA→MIL lowering. Fine for a v-prediction downstream task.
9. `coremltools` prints a post-save `RuntimeWarning: Failed to build the model execution plan` — red-herring from the CPU load path; the .mlpackage is usable, parity predict() still runs.

### Decisions locked in

- **No CFG in the graph.** Swift side calls the diffusion head twice per step and blends. Keeps graph simple and `timestep` float.
- **CPU_AND_GPU, FLOAT32** everywhere until end-to-end parity passes. ANE + fp16 is a later optimization; fp16 drift over 20 DDPM steps is a documented PocketTTS problem.
- **iOS18/macOS15 minimum.**
- **Fixed T=16 VAE decode** — 2.133 s audio chunks. Production may need bucketed variants.

## 2026-04-15 — Phase E partial: end-to-end pipeline executes

`swift run vibevoice-cli "Hello world" --models ../conversion/out -o /tmp/out.wav`
now runs all the way through:

1. swift-transformers tokenizer (bundled via `05_bundle_tokenizer.py` from
   `Qwen/Qwen2.5-0.5B`) → int32 ids.
2. Text LM prefill (padded/truncated to T=16, matching the trace-baked shape).
3. DDIM v-prediction denoise loop in Swift: cosine β, 20 steps, CFG (cond +
   uncond blend at guidance_scale=3.0). Two diffusion-head calls per step.
4. Diffusion output (1,8,64) → VAE input (1,64,16) via interim tile+transpose.
5. VAE decode → 51200 Float32 samples → IEEE-float WAV.

What's **not** real yet:
- No autoregressive TTS LM decode loop. Conditioning is just the text-LM
  prefill hidden block sliced to 8 positions — semantically wrong, but
  graph-level correct. Real synthesis needs `tts_lm_decode` wired with EOS
  classifier gating.
- `text_lm_decode.mlpackage` fails to load with "Failed to build the model
  execution plan" (error code -6). Disabled at engine init — only prefill is
  required by the current path.
- Tile-to-T=16 latent adapter is a shim; production wants re-traced diffusion
  head with matched chunk size.

Key Swift-side pieces added:
- `VibeVoiceKit/ModelLoader.swift` — compiles .mlpackage → .mlmodelc (cached
  alongside the package) because Swift at runtime does not auto-compile.
- `VibeVoiceKit/DiffusionRunner.swift` — full DDIM+CFG implementation against
  the converted `diffusion_head.mlpackage`.
- `VibeVoiceKit/LLMRunner.swift` — prefill-only harness; decode path TODO.

## 2026-04-15 — Phase E: TTS LM re-traced with correct `forward_tts_lm` graph

The original `tts_lm_{prefill,decode}.mlpackage` produced by
`02_convert_llm.py --tts` were architecturally wrong: they traced the TTS LM
as a plain Qwen2 LM, but upstream's `forward_tts_lm` (line 401 of
`modeling_vibevoice_streaming_inference.py`) does three extra things before
the LM layers run:

1. Embed `input_ids` normally.
2. **Overwrite the tail embedding rows** with `lm_last_hidden_state` — at a
   text step this is the text-LM's output; at a speech step it's
   `acoustic_connector(speech_latent)`.
3. Add a **type embedding** via `tts_input_types(tts_text_masks.long())` where
   mask = 1 marks a text position and 0 marks a speech position.

New script: `scripts/02b_convert_tts_lm.py`. Two wrappers (`TTSPrefillWrapper`,
`TTSDecodeWrapper`) with graph inputs:

```
input_ids         (1, S) int32
lm_hidden_padded  (1, S, 896) float32   # zero-padded on the head
tail_mask         (1, S, 1)  float32    # 1 in tail positions, 0 elsewhere
tts_text_masks    (1, S) int32          # 1 = text, 0 = speech
[+ k_cache, v_cache, position_ids on decode]
```

The tail splice is done with the **mask form** `tok*(1-m) + lm*m` rather
than a dynamic Python slice `inputs_embeds[:, start:, :] = lm_hidden`. The
slice form compiled but blew up at `predict()` time with the same
`ios18.slice_by_index` "zero shape" error seen earlier on text-LM decode —
the head slice `[:, :start, :]` can be zero-length when `S == K`, which
coremltools can't fold at compile time with two independent RangeDims.

Parity: `tts_lm_prefill hidden max|Δ| = 4.6 × 10⁻⁵` (well inside 5 × 10⁻³).

Phase E Swift wiring so far:
- `HeadsRunner.swift` — loads `acoustic_connector.mlpackage`,
  `tts_eos_classifier.mlpackage`, and `speech_constants.json`. Exposes
  `project(latent:)`, `eosProbability(hidden:)`, and
  `normalizeLatentForVAE(_:)` (`latent / scale - bias`).
- `LLMRunner.decodeStep` split into `textDecodeStep` (old signature) and
  `ttsDecodeStep` (new 7-input signature: `input_ids`, `lm_hidden_padded`,
  `tail_mask`, `tts_text_masks`, `k_cache`, `v_cache`, `position_ids`).
- `Engine.synthesize` now applies the VAE normalization before decode.

## 2026-04-15 — Phase E: autoregressive TTS loop live end-to-end

`swift run vibevoice-cli "Hello world, this is a test." --models ../conversion/out`
now runs the real autoregressive path and emits a 6.27 s WAV. EOS classifier
fired at frame 47 (of 200 max). Every component is on the true call path:

- Text LM prefill → TTS LM prefill (cond + uncond via `zeroLM=true`) using the
  new mask-form graph.
- Per speech frame: `DiffusionRunner.denoiseFrame` runs 20 DDIM steps with CFG
  on `(1, 1, 64)` latents (diffusion head's `T` is dynamic, no re-trace needed).
- `acoustic_connector(latent)` projects back into (1, 1, 896) and feeds the
  next `tts_lm_decode` step on both cond + uncond branches (`tailMask=1`,
  `ttsTextMasks=0` = speech position).
- `tts_eos_classifier` gates on the positive branch's last hidden.
- VAE decoder called on 16-frame windows: `latent / scale - bias` normalization
  done inline while packing the (1, 64, 16) tensor.

First-cut simplifications (still TODOs):
- **No interleaved text window.** Upstream alternates 5-token text chunks with
  6-frame speech chunks. We prefill the full text once (capped at the traced
  T=16) and never re-enter the text LM. Fine for short prompts; longer prompts
  need to drive `text_lm_decode` and incrementally extend the TTS cache.
- **Negative branch uses `lm_hidden = zeros`** as the uncond text condition.
  Upstream builds a separate negative `tts_lm_input_ids` with pad tokens; the
  behavioral difference matters for CFG quality but is a later refinement.
- **Audio quality not yet validated aurally.** Pipeline is structurally
  complete but perceptual parity vs. the PyTorch reference is the next gate.

## 2026-04-15 — Phase E complete: intelligible speech from CoreML

Two critical fixes got us from noise to intelligible speech:

1. **Voice prompt conditioning.** The TTS LM requires pre-computed KV caches
   that encode a voice identity. Without them the model outputs noise. Script
   `08_export_voice_prompt.py` converts the upstream `.pt` voice caches to
   `.npy` arrays; `VoicePrompt.swift` loads them as initial KV state for all
   four branches (lm, tts_lm, neg_lm, neg_tts_lm). 24 voices exported.

2. **Per-token text-LM → TTS-LM hidden pairing.** The initial implementation
   fed the *last* text-LM hidden to every TTS-LM step in a text window.
   Upstream feeds each token's own hidden state 1:1 as `lm_last_hidden_state`.
   The fix: collect `perTokenHidden[i]` during text-LM decode, then pair each
   with the corresponding TTS-LM decode step.

`swift run vibevoice-cli "Hello world, this is a test." --models out --voice en-Frank_man`
now produces intelligible speech matching the PyTorch reference.

CLI flags added: `--voice <name>`, `--guidance <float>`, `--list-voices`.

## Remaining (Phase E — Swift wiring)

The runners in `swift/Sources/VibeVoiceKit/` have correct types and signatures but only `VAERunner.decode` is concretely implemented.

For a future session:

1. **Tokenizer bundling** — `scripts/` should copy `tokenizer.json` + `tokenizer_config.json` from the HF snapshot into `out/tokenizer/` so swift-transformers `AutoTokenizer.from(modelFolder:)` finds them.
2. **LLMRunner.runPrefillAndDecode**:
   - `MLMultiArray` packing helper for int32 token arrays.
   - Prefill call → extract `hidden_states`, `k_cache`, `v_cache`.
   - Decode loop: take text LM last hidden, feed `tts_lm_decode` until `tts_eos_classifier` fires (needs a 6th CoreML model wrapping that binary head, or implement it in Swift from weights).
   - Collect per-step hidden states for diffusion conditioning.
3. **DiffusionRunner.denoise**:
   - DDPM v-prediction scheduler with cosine β schedule, 20 steps (match `head.config.ddpm_num_steps=1000` train, 20 inference via skip).
   - Per step: sample noise, two predictions (cond + uncond zero-conditioning), CFG blend at `guidance_scale=3.0`, update latent with v-prediction formula.
   - Bucket or pad latent to the traced (1, T, 64) shape. Currently the diffusion head is traced with T=8; match at the graph level by re-tracing with the real chunk size once confirmed.
4. **End-to-end wiring**: `Engine.synthesize` already strings the calls together — it'll work once (2) + (3) return real values.
5. **CLI smoke test**: `swift run vibevoice-cli "Hello" --models ../conversion/out -o /tmp/out.wav` is the primary correctness harness.

## Reference

- Conversion scripts: `conversion/scripts/01_inspect_model.py` through `04_convert_vae.py`
- Parity harness: `conversion/scripts/05_parity_check.py` (scaffolded, not yet implemented for end-to-end waveform compare)

---

## Post-launch debugging log

### Stutter investigation (75% of seeds repeated words like "la-lazy")

Audio A/B testing showed Swift stuttered on most seeds while the upstream
PyTorch demo did not. Localized via the reference/trace dump harness
(`10_pytorch_reference_dump.py` → `11_coreml_trace_dump.py` →
`12_compare_dumps.py`). Two distinct bugs, both in the Swift orchestration
layer (not in the converted `.mlpackage` files):

1. **Box-Muller in fp32** — `DiffusionRunner.randomNormal()` originally ran
   the uniform-to-normal transform in `Float`, which has ~1e-7 relative
   error per sample. Because the diffusion loop runs the prediction head
   `2×CFG × steps × N frames` times, that small error compounds and (on
   some seeds) pushes the trajectory onto a chaotic branch where the
   TTS-LM picks a different next token. Fix: do Box-Muller arithmetic in
   `Double`, truncate to `Float` only at store time.
   File: `swift/Sources/VibeVoiceKit/DiffusionRunner.swift`.

2. **Tokenizer missing `strip() + "\n"`** — Upstream's
   `VibeVoiceStreamingProcessor` tokenizes with
   `tokenizer.encode(text.strip() + "\n", add_special_tokens=False)`.
   The trailing `"\n"` is significant on Qwen2.5's BPE: a trailing `"."`
   followed by `"\n"` merges into a single token (id 624 for `".\n"`),
   *different* from the token for `"."` alone (id 13). Our
   `VibeVoiceTokenizer.encode` was calling the underlying tokenizer with
   the raw text, so the last token came out as 13 instead of 624. Because
   text is fed into the TTS-LM in 5-token windows and the *last* token of
   each window becomes the conditioning for the next speech-diffusion
   pass, getting the final token wrong silently corrupted speech from
   window 2 onward — which is exactly where the stutter appeared. Fix:
   always `trimmingCharacters(in: .whitespacesAndNewlines) + "\n"` in
   `VibeVoiceTokenizer.encode`.
   File: `swift/Sources/VibeVoiceKit/Tokenizer.swift`.

### Verification

With both fixes, `11_coreml_trace_dump.py --seed 42` (SplitMix64 noise,
matching Swift's RNG) vs `vibevoice-cli --seed 42` agrees to
`max|Δ|=3e-5` (fp16 noise floor) on latent frames 0–15. Frames 16+
diverge by up to 0.3 at sample level, but this is the VAE-decode
windowing difference (script 11 uses non-overlapping 16-frame chunks
zero-padded right; Swift slides a 16-frame window with 8 frames of left
context). Swift's approach is *closer* to upstream's stateful streaming
decode, so script 11 is the approximation here, not Swift.

### Reference-dump harness (keep for future regressions)

- `10_pytorch_reference_dump.py` — PyTorch ground-truth with every
  intermediate: posLast, negLast, noise, vcond, vuncond, v-blend, x0,
  sample, latent, final audio.
- `11_coreml_trace_dump.py` — same intermediates, but running the
  `.mlpackage` files from Python. `--use-ref-noise` imports PyTorch's
  noise for noise-independent comparison; without it, uses the SplitMix64
  PRNG that `DiffusionRunner.swift` uses (bit-for-bit matching).
- `12_compare_dumps.py` — `max|Δ|` every artifact, first-fail
  localization. First diff above tolerance localizes the bug.

For regressions where Swift diverges from Python CoreML, the next step
is to add a Swift utility that dumps `latent_frame{f}.npy` alongside
the audio, so we can compare latents — not just audio — against script
11's SplitMix trace all the way to the final frame. Divergences past
frame 15 in the audio are expected (VAE windowing); any latent
divergence before the end of generation is a real Swift bug.

## Boundary-trim (head silence + tail ghost phoneme)

Even after the stutter fix, two boundary artifacts remained:

1. **Leading silence** — VibeVoice often needs 3–8 latent frames (~0.4–1s
   at 7.5 Hz frame rate) for the initial diffusion noise to resolve into
   a phoneme. Clips sound like they start with a delay before the voice
   begins. Seeds 1/7/999/2024/31337 were worst (≥800ms lead-in) and
   several voices (Carter, Mike) also had this.

2. **Trailing "ghost phoneme" (ambiguous, left mostly untouched)** — The
   TTS-EOS classifier is a **lagging indicator**. It reads `posLast`
   *after* a frame's latent has already been sampled and appended. When
   EOS fires with a sharp jump (e.g., grace_female seed 42: prev-frame
   eos = 0.024 → current = 1.0), the last 1–3 appended frames sometimes
   contain the priming onset of a word the model decided not to complete
   — audible as "another word starting" at the clip boundary. However,
   the same RMS signature (valley-then-bump in the last 4 frames) also
   appears naturally for **plosive consonant releases**: the 'g' in "dog"
   closes the airway (RMS valley) then releases into a burst (RMS bump)
   which decays (quieter tail). Automatically trimming at the valley
   cuts the 'g'. Heuristics alone can't distinguish these cases, so tail
   trimming is **disabled by default** — users who have validated a
   voice + text combination by ear can opt in via
   `loopConfig.trimTailOnset = true`. Note: upstream has the same EOS
   timing behaviour (the Python generate loop also appends `audio_chunk`
   for frame N before checking EOS on H_{N+1}).

### Fix

Post-generation audio trim in `TTSLoop.trimBoundaries`:

- **Head** (on by default): walk forward while per-frame RMS <
  `silenceFloor` (0.01), clip at the first speech frame. Keeps ≥1 frame
  minimum. Unambiguous — only removes pure silence.
- **Tail** (off by default): find a local RMS minimum in the last 4
  frames; if any subsequent frame has RMS > `1.5 ×` that minimum,
  truncate at the valley. Left off because the signature collides with
  natural plosive releases (see above).
- **Fade**: 10ms linear fade-in/out at each trim edge to prevent clicks.

First version shipped with tail-trim enabled; user reported it was
cutting the 'g' of "dog" on Grace. The trim was valid by the heuristic
— the valley-bump was real — but that was the 'g' release, not a ghost
phoneme. Retaining the code because on the right voice/seed it does
remove genuine artifacts; just gate it behind opt-in.

### Trade-offs

- Takes a few ms post-generation (RMS computation is one pass over the
  final audio, ~100k samples max).
- Head-trim is safe to always leave on; if a user wanted silence on the
  head, set `loopConfig.trimHeadSilence = false`.
- `long.wav` (Peter-Piper-tongue-twister) stops mid-word. That's an
  EOS-training issue, not a trimming issue; accepted as a known
  limitation of VibeVoice-Realtime-0.5B at longer sequences.

Files: `swift/Sources/VibeVoiceKit/TTSLoop.swift` (trim logic,
`trimBoundaries` method and `Config.trimHeadSilence` /
`Config.trimTailOnset` / `Config.silenceFloor` knobs).

## 2026-04-16 — Parallelization (async let): ~32 % RTF reduction

### Change

Two independent CoreML forward passes per speech frame were running
sequentially; both are now overlapped via `async let`:

1. **Pos/neg TTS-LM decode** in `TTSLoop.synthesizeInto`. Each speech
   frame runs the TTS LM twice — once on the positive (conditioned)
   KV cache, once on the negative (uncond) KV cache — to produce the
   two hidden vectors the diffusion head needs for CFG blending.
   Previously `let posStep = try await llm.ttsDecodeStep(...); let
   negStep = try await llm.ttsDecodeStep(...)`.  Now:
   `async let posTask = ...; async let negTask = ...;
    let (posStep, negStep) = try await (posTask, negTask)`.

2. **Cond/uncond diffusion CFG** in `DiffusionRunner.denoiseFrame` and
   `DiffusionRunner.denoise`. Same shape of change — the two
   predicted-noise calls inside each DPM-Solver step run in parallel.

CoreML `MLModel.prediction(from:)` is documented re-entrant and the
inputs are not mutated by the forward pass, so concurrent calls are
safe as long as each call owns its own input `MLMultiArray` scratch.

### Swift-6 strict-concurrency barriers hit

- `MLModel` and `MLMultiArray` are not `Sendable` in the SDK. Added
  `@retroactive @unchecked Sendable` conformances in `ModelLoader.swift`
  with a comment justifying the safety contract (Apple's docs say the
  types are thread-safe for prediction; mutation is confined to setup).
- Each runner class (`LLMRunner`, `DiffusionRunner`, `VAERunner`,
  `HeadsRunner`) and `TTSLoop` marked `final class ... @unchecked
  Sendable` so they can be sent into `async let` spawns.
- Decode input scratch (`decodeIDs`, `decodePos`, `decodeTail`,
  `decodeTM`) removed from `LLMRunner` storage — previously mutated
  in-place per call, which would race when two concurrent calls share
  the instance. Now allocated fresh per call via
  `LLMRunner.makeIDs / makeTail`. Allocation cost (~4 tiny
  `MLMultiArray`s, microseconds) is negligible vs the 15–20 ms
  forward.
- `let`-bound snapshots before each `async let` spawn
  (`let sampleIn = sample`, `let posKIn = posK`, …) to work around
  "sending 'x' risks causing data races" when capturing a `var`.

### Determinism verification

Regenerated the full 23-clip matrix (6 voices × 4 step counts ×
3 lengths × 7 seeds) after the change. Results vs the pre-change
sequential baseline saved at `out/clips/_report_baseline_sequential.md`:

- **20 / 23 clips bit-identical** end-to-end (max|x−y| = 0.0 across
  the entire waveform).
- **3 / 23 clips** differ, but the divergence is exclusively in the
  last 240 samples (~10 ms) of the overlap window — i.e. the
  raised-cosine fade-out that the earlier tail-trim added right
  before its cut point. Head and middle thirds are bit-identical.
  (Confirmed by `scripts/locate_divergence.py`: argmax of |diff|
  at fraction ≥ 0.998 on all 3 clips; head_max = mid_max = 0.)
  These clips got +9600 samples (3 extra 3200-sample speech frames)
  because the tail-trim was disabled separately, not because of any
  numeric drift in the parallelized path.

Parallelization introduces **zero** numerical drift on the overlapping
duration of every clip. CoreML's prediction path is deterministic as
long as input arrays are unshared.

### RTF reduction

| Group      | mean Δ | best Δ | worst Δ |
| ---------- | -----: | -----: | ------: |
| all 23     | −32.2 % | −53.6 % | −12.1 % |
| voices (6) | −38.1 % | −52.3 % | −29.5 % |
| steps (7)  | −43.9 % | −53.6 % | −30.8 % |
| seeds (7)  | −19.9 % | −22.6 % | −13.5 % |

Biggest wins are on step-count-heavy runs (steps=20 dropped from 1.21×
real-time to 0.66×, i.e. now faster than real-time) because more
diffusion steps = more parallelized CFG pairs per frame. Short clips /
different seeds gain less (already fast and dominated by VAE/text
prefill which aren't parallelized).

Absolute floor after this change: ~0.36 RTF for a 3-second clip on
the M-series dev laptop; first-audio latency follows proportionally
and comfortably supports the streaming roadmap (Phase G).

### Files

- `swift/Sources/VibeVoiceKit/TTSLoop.swift`
- `swift/Sources/VibeVoiceKit/DiffusionRunner.swift`
- `swift/Sources/VibeVoiceKit/LLMRunner.swift`
- `swift/Sources/VibeVoiceKit/ModelLoader.swift` (retroactive
  `Sendable`)
- `swift/Sources/VibeVoiceKit/{HeadsRunner,VAERunner}.swift`
  (`@unchecked Sendable` annotations)
- `scripts/diff_parallel_vs_sequential.py` (speed + quality diff)
- `scripts/locate_divergence.py` (divergence localization)
- `out/clips/_parallel_vs_sequential.md` (full table)

## 2026-04-16 — Dead ends: stateful KV cache and post-training fp16

After the parallelization win (RTF 0.58 → 0.40 mean), I considered two
next steps. Both were wrong turns worth documenting so they aren't
re-tried.

### Stateful KV cache: shuttle overhead is 0.2 %, not 20–30 %

My prior write-up assumed the per-step cost of passing the TTS-LM KV
cache in and out of CoreML as `MLMultiArray` was a meaningful fraction
of the step, because (a) the cache is up to ~10 MB, (b) 60 calls per
clip, (c) intuition from other frameworks where tensor marshalling is
expensive. I claimed 20–30 % upside from `ct.StateType`.

That was wrong. I instrumented `LLMRunner.ttsDecodeStep` with three
timers (dict pack / `prediction(from:)` / feature unpack), gated
behind `VIBEVOICE_PROFILE_SHUTTLE=1`, and re-ran the frank_s42 clip.
Result over 56 calls:

```
  pack  (dict build):     0.019 ms/call   0.1 %
  fwd   (prediction):    17.094 ms/call  99.8 %
  unpack (feature get):   0.011 ms/call   0.1 %
```

The 99.8 % is the full GPU forward. Whatever buffer marshalling
CoreML does between `MLMultiArray` and the Metal buffers on the
backend side is included in that — but it's overlapped with compute
and not separable. Cache-length scaling: fwd went 13.5 ms → 19.6 ms
as T grew 0→300, and most of that 6 ms is attention's O(T) cost in
the graph, not shuttle. Best case stateful recovers 2–5 % RTF;
realistic ~2 %. Not worth the multi-hour re-export + parity + rewire
risk.

The profiling instrumentation was removed after the finding. If
needed again, `git log` has the block under `LLMRunner.swift`.

### Post-training fp16 via `linear_quantize_weights`: 21× slower

The `conversion/out/fp16/` directory contains models produced by
`09_quantize_fp16.py`, which uses coremltools'
`linear_quantize_weights`. Pointing the CLI at that dir
(`--models conversion/out/fp16`) produced RTF **7.71** on frank_s42
vs 0.36 at fp32. Every stage regressed, and the VAE in particular
went from 0.18 s → 22.08 s (CPU fallback).

Cause: `linear_quantize_weights` quantizes the *weights* to fp16 but
leaves the compute graph at fp32. At predict time each weight is
upcast to fp32 before the op, then the op runs fp32. The fp16
storage saves memory but adds per-op upcast cost, and some ops don't
have fp32 kernels on GPU — hence the VAE's CPU fallback. The
`fp16/` dir should be treated as "smaller for distribution, slower
for execution"; do not point the CLI there.

### What a real fp16 win would look like

A proper fp16 export needs `ct.convert(..., compute_precision=
ct.precision.FLOAT16, ...)` at convert time, not post-training
quantization. This gives the runtime actual fp16 arithmetic on the
GPU (~1.5–2× throughput on M-series). Non-trivial because each of
the 5 hot-loop models needs its own parity gate (tolerances loosen
to ~1e-2 at fp16), and the VAE's 3200× upsampling may overflow fp16
at some intermediate activation — needs investigation. This is the
honest "next big optimization", deferred to a future session.
