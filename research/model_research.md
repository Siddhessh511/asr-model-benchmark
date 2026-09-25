# Primary Model Research & Architectural Analysis

This document presents a rigorous architectural, algorithmic, and operational analysis of the three Automatic Speech Recognition (ASR) systems evaluated in this comparative benchmark:
1. **OpenAI Whisper (`base`)**
2. **Faster-Whisper (`base`)**
3. **Wav2Vec 2.0 (`facebook/wav2vec2-base-960h`)**

All technical claims cite authoritative primary sources cataloged in [references.md](./references.md). This research strictly separates published theoretical and empirical claims from our local benchmark measurements.

---

## 1. OpenAI Whisper Base

### 1.1 Model Overview
OpenAI Whisper is an automatic speech recognition (ASR) and speech translation model family developed by Radford et al. [1, 2, 3]. Whisper Base is a compact variant comprising approximately 74 million parameters [1, 2]. Unlike conventional single-task acoustic models, Whisper is formulated as an end-to-end multi-task system capable of performing multilingual speech recognition, speech-to-English translation, voice activity detection (VAD), and phrase-level timestamp alignment within a single unified sequence-to-sequence architecture [1].

### 1.2 Architecture
Whisper utilizes a standard encoder-decoder sequence-to-sequence Transformer architecture [1].
- **Acoustic Front-End:** Audio input is resampled to 16 kHz and transformed into an 80-channel log-magnitude Mel-spectrogram computed over 25 ms windows with a 10 ms hop size [1]. The spectrum is normalized to roughly $[-1, 1]$ with zero mean across the pretraining corpus [1].
- **Convolutional Downsampler & Encoder:** Two initial 1D convolutional layers with a filter width of 3 and a stride of 2 compress the temporal dimension by a factor of 2 [1]. Sinusoidal positional embeddings are added to the downsampled features before being processed by a 6-layer Transformer encoder stack (8 attention heads, hidden dimension $d_{\text{model}} = 512$, feed-forward dimension 2048) [1, 2].
- **Autoregressive Transformer Decoder:** The decoder comprises a 6-layer Transformer stack (8 attention heads, $d_{\text{model}} = 512$) utilizing causal self-attention and cross-attention over the encoder representations [1, 2]. Predictions are generated token-by-token conditioned on preceding tokens.
- **Tokenization:** Multi-task processing relies on Byte-Pair Encoding (BPE) with a vocabulary of 51,865 tokens [1, 3]. Special prompt tokens (`<|startoftranscript|>`, `<|language|>`, `<|transcribe|>`, `<|notimestamps|>`) guide task execution.
- **30-Second Chunking Window:** The official implementation processes audio strictly in fixed 30-second windows [1, 3]. Shorter audio clips are zero-padded to 30 seconds; longer recordings are transcribed sequentially using a 30-second sliding window with heuristic timestamp tracking and temperature fallbacks [1, 3].

### 1.3 Training Data
The Whisper model family was trained on **680,000 hours** of weakly supervised audio collected from the public web [1].
- **English Corpus:** 438,000 hours of English speech-to-text pairs [1].
- **Multilingual Corpus:** 126,000 hours of audio spanning 96 non-English languages (including major Indo-Aryan languages such as Hindi, Urdu, Bengali, and Marathi) [1, 2].
- **Translation Corpus:** 117,000 hours of non-English audio paired with English text translations [1].
- **Data Filtering:** Weak supervision relied on automated heuristics rather than manual human verification, using pre-trained language identification and transcript-audio alignment filters to discard machine-generated transcripts and low-quality pairs [1].

### 1.4 Licensing
The official OpenAI Whisper code repository and model weights are released under the **MIT License** [3]. This permissive open-source license allows unrestricted academic and commercial deployment, modification, and integration without royalty obligations.

### 1.5 Hardware / Deployment Requirements
- **Runtime:** Built natively for PyTorch (`torch.nn`) [3].
- **Execution Engine:** Standard execution runs FP16 on CUDA-enabled GPUs and falls back to FP32 on CPU architectures [1, 3].
- **System Dependencies:** Strictly requires an external FFmpeg executable available on the system PATH to handle audio decoding and format transcoding [3].
- **Memory & Latency Overhead:** Autoregressive decoding requires $O(N)$ sequential forward passes through the decoder for an $N$-token sequence [1]. On CPU platforms, PyTorch lacks default fused quantization kernels, leading to significant memory bandwidth pressure and higher compute latency [6].
- **Measured Local Baseline:** In our CPU-based test environment (Intel 8-core / 12-thread host), Whisper Base consumed an average of **840.1 MB RAM** with a clean inference latency of **4.416s** per utterance (`RTF: 1.149`).

### 1.6 Strengths
- **Acoustic Robustness:** Extensive weakly supervised training across heterogeneous web audio imparts high resilience to non-standard accents, colloquial speech, and recording equipment variation [1].
- **All-in-One Capabilities:** Natively handles language identification, text normalization, and timestamp estimation without requiring external acoustic or language models [1, 2].
- **Standardized Ecosystem:** Widespread library support across Python, Hugging Face Transformers, and major machine learning platforms [2, 3].

### 1.7 Weaknesses
- **Autoregressive Hallucination:** Under acoustic degradation, low signal levels, or extended silence, the autoregressive decoder can enter repetitive loops or hallucinate plausible text phrases not present in the acoustic signal [1, 2].
- **Language / Script Ambiguity:** On lower-resource languages or unprompted dialectal speech, the language identification head can oscillate between related languages or orthographies. In unprompted Hindi recognition, Whisper Base can output Romanized "Hinglish" or Perso-Arabic (Urdu) scripts instead of standard Devanagari [2].
- **Compute Inefficiency on CPU:** PyTorch's default execution graph incurs substantial latency overhead on CPU hardware, often failing to sustain real-time transcription (`RTF > 1.0`) without GPU acceleration [6].

### 1.8 Relevance to Noisy Real-World Audio
OpenAI's published research demonstrates that training on diverse, noisy web audio yields superior zero-shot transfer compared to models trained exclusively on clean laboratory corpora like LibriSpeech [1]. However, official documentation notes that severe signal-to-noise ratio (SNR) degradation remains a trigger for decoding failure modes, including phrase repetition and insertion of phantom tokens [1, 2].

### 1.9 Relevance to the Benchmark
Whisper Base serves as our primary end-to-end sequence-to-sequence baseline. Benchmarking this model on clean and noisy audio directly validates how an autoregressive web-scale Transformer handles real-world acoustic degradation and provides a baseline against which optimized runtimes can be evaluated.

---

## 2. Faster-Whisper Base

### 2.1 Model Overview
Faster-Whisper is an optimized re-implementation of OpenAI's Whisper model built on **CTranslate2**, a specialized C++ inference engine for Transformer models developed by SYSTRAN [4, 6]. 
- **Important Technical Clarification:** Faster-Whisper is **not a separately trained ASR model family** [4].
- It utilizes the exact neural weights converted directly from OpenAI's official checkpoints [4].
- Faster-Whisper Base runs the 74M parameter Whisper Base architecture on an optimized, highly efficient execution runtime [4, 6].

### 2.2 Architecture
Faster-Whisper retains the identical architectural topology of OpenAI Whisper Base (80-channel log-Mel front-end, 2-layer convolutional downsampler, 6-layer Transformer encoder, 6-layer Transformer decoder, 51,865-token BPE tokenizer) [1, 4]. The architectural distinction lies in runtime graph execution and memory layout [6]:
- **CTranslate2 Execution Graph:** Replaces PyTorch's dynamic graph with a custom C++ runtime designed specifically for encoder-decoder Transformer inference [6].
- **Layer & Kernel Fusion:** Fuses consecutive memory-bound operations, such as Layer Normalization, linear projections, and activation functions (GELU), into single optimized compute kernels [6].
- **KV-Cache Optimization:** Implements pre-allocated, contiguous memory buffers for Key-Value attention states during autoregressive decoding, preventing memory fragmentation and dynamic allocation overheads [6].
- **Embedded Audio Decoding via PyAV:** Directly integrates PyAV (in-memory C bindings to `libavformat` and `libavcodec`) to decode audio waveforms in memory, bypassing the need to fork external `ffmpeg` subprocesses [4].
- **Integrated Voice Activity Detection (VAD):** Embeds an optional Silero VAD pre-filter to detect and eliminate silence or non-speech segments before Transformer encoding, mitigating hallucination tendencies [4].

### 2.3 Training Data
Faster-Whisper uses no independent training data. The model weights are directly converted from OpenAI's official Whisper Base release, derived from the same 680,000 hours of weakly supervised web audio (438k hours English, 126k hours multilingual, 117k hours translation) [1, 4].

### 2.4 Licensing
Faster-Whisper is distributed under the **MIT License** [5]. The underlying CTranslate2 runtime engine is also licensed under the **MIT License** [6]. Both components allow unrestricted commercial and academic use, redistribution, and modification.

### 2.5 Hardware / Deployment Requirements
- **Cross-Platform Acceleration:** Native acceleration on x86-64 and ARM CPUs via Intel oneDNN / MKL and OpenBLAS; native GPU support via cuBLAS and NVIDIA TensorRT [6].
- **Quantization Capabilities:** Supports multiple precision profiles: FP16, INT16, INT8, and INT8_float16 [4, 6]. On CPU platforms, 8-bit integer quantization (INT8) leverages vector instructions (AVX2, AVX-512, VNNI) to achieve high throughput with negligible accuracy divergence [6].
- **Lightweight Footprint:** Eliminates system-level external binary requirements (PyAV bundles required decoders) and reduces binary disk requirements to ~145 MB for INT8 weights [4].
- **Published Reference Benchmarks:** SYSTRAN reported reference benchmarks demonstrating up to **4× speedup** and **70% memory reduction** on standard GPU test suites for large Whisper checkpoints compared to vanilla PyTorch [4].
- **Measured Local Baseline:** In our CPU-based test environment using INT8 quantization, Faster-Whisper Base achieved an average clean latency of **2.336s** (`RTF: 0.569`, a **1.89× speedup** over PyTorch Whisper) and an average noisy latency of **3.502s** (`RTF: 0.842`), with peak resident memory of **619.7 MB** (**26.2% lower** than PyTorch Whisper).

### 2.6 Strengths
- **Sub-Real-Time CPU Latency:** Achieves real-time execution (`RTF < 1.0`) on standard commodity CPUs without requiring dedicated GPU hardware [4, 6].
- **Reduced Memory Bandwidth:** Lower memory consumption allows higher concurrent stream concurrency on edge gateways or microservice nodes [4].
- **Robust In-Memory Pipeline:** Avoids OS process spawning bottlenecks through native PyAV audio handling [4].

### 2.7 Weaknesses
- **Inherited Model Limitations:** Inherits all fundamental acoustic, vocabulary, and training limitations of the underlying Whisper checkpoint, including susceptibility to language code confusion on unprompted multilingual audio [4].
- **Quantization Nuances:** While INT8 quantization preserves accuracy on most datasets, extreme quantization can occasionally alter decoding trajectories in low-confidence acoustic segments [6].

### 2.8 Relevance to Noisy Real-World Audio
Faster-Whisper preserves the acoustic representations and weights of Whisper Base, maintaining its documented resilience to background acoustic noise [1, 4]. Furthermore, its integration of Silero VAD enables early filtering of acoustic noise during pauses, suppressing silence hallucinations. The reduced latency prevents computational backlog when processing degraded audio streams.

### 2.9 Relevance to the Benchmark
Faster-Whisper Base provides a direct empirical comparison between an optimized inference runtime (CTranslate2 INT8) and standard PyTorch on the identical model weights. This allows our benchmark to isolate runtime engineering efficiency from model architectural differences.

---

## 3. Wav2Vec 2.0 Base-960h

### 3.1 Model Overview
Wav2Vec 2.0 is a framework for self-supervised learning of speech representations introduced by Baevski et al. (Meta AI / FAIR, 2020) [7]. The specific evaluated checkpoint, `facebook/wav2vec2-base-960h`, is a **94.4 million parameter** acoustic model pretrained on unlabeled speech and fine-tuned for end-to-end speech recognition using Connectionist Temporal Classification (CTC) [7, 8, 9, 11].

### 3.2 Architecture
Wav2Vec 2.0 employs a non-autoregressive, acoustic-only encoder topology operating directly on raw audio waveforms [7].
- **Raw Audio Input:** Operates strictly on raw single-channel PCM audio sampled at **16 kHz**; it does not compute handcrafted spectrograms [7, 8].
- **Convolutional Feature Encoder:** A multi-layer temporal convolutional network with 7 blocks (kernel sizes 10, 3, 3, 3, 3, 2, 2 and strides 5, 2, 2, 2, 2, 2, 2) downsamples the 16 kHz waveform into latent feature representations $Z$ representing 25 ms of audio every 20 ms (a cumulative temporal stride of 320) [7].
- **Transformer Context Network:** A 12-layer Transformer encoder stack (hidden size 768, 8 attention heads, intermediate dimension 3072, totaling ~94.4M parameters) processes the local acoustic representations to generate contextualized features [7, 9].
- **CTC Output Head & Vocabulary:** A linear projection maps contextualized vectors directly into emission probabilities over a **32-token character vocabulary** [7, 8, 9]. Greedy CTC decoding collapses consecutive identical tokens and removes blank tokens [11].
- **Character Alphabet Scope:** The vocabulary contains only 32 tokens: uppercase English letters `A-Z`, apostrophe `'`, space delimiter `|`, padding, and blank tokens [8, 9]. It contains no non-Latin or Devanagari script characters.

### 3.3 Training Data
- **Self-Supervised Pretraining:** Pretrained on **960 hours** of unannotated English speech from the LibriSpeech corpus (`train-clean-100`, `train-clean-360`, and `train-other-500`) using a contrastive loss over quantized latent representations [7, 8].
- **Supervised Fine-Tuning:** Fine-tuned on the same 960 hours of transcribed LibriSpeech audio using supervised CTC loss [7, 8].
- **Published Paper Benchmarks:** On LibriSpeech, Baevski et al. reported **3.4% WER** on `test-clean` and **8.6% WER** on `test-other` without a language model (and **1.8% / 4.7%** with an external 4-gram LM) [7].

### 3.4 Licensing
The Fairseq implementation and `facebook/wav2vec2-base-960h` weights are released under the **Apache License 2.0** [8, 10]. This license permits commercial deployment, modification, and integration into proprietary pipelines.

### 3.5 Hardware / Deployment Requirements
- **Non-Autoregressive Feedforward:** Computes all output token logits simultaneously in a single forward pass through the convolutional encoder and Transformer stack, requiring $O(1)$ forward passes regardless of sequence length [7, 11].
- **Compute & Memory Efficiency:** Highly efficient on CPU hardware. The absence of sequential decoding loops eliminates key-value caching overheads [7].
- **Strict Sampling Constraint:** Audio must be resampled to exactly 16 kHz prior to model ingestion [7, 8].
- **Measured Local Baseline:** In our CPU-based test environment, Wav2Vec2 Base-960h demonstrated an ultra-fast clean latency of **0.376s** (`RTF: 0.083`, running **12.0× faster than real-time**) and noisy latency of **0.368s** (`RTF: 0.081`), with peak memory of **962.9 MB**.

### 3.6 Strengths
- **Deterministic Ultra-Low Latency:** Non-autoregressive CTC execution provides predictable, near-instantaneous inference suitable for low-latency streaming pipelines [7, 11].
- **Immunity to Hallucination Loops:** Because CTC directly maps acoustic frames to character emissions without an autoregressive feedback loop, Wav2Vec2 cannot enter infinite token repetition loops or hallucinate text during silence [7, 11].
- **Direct Waveform Feature Extraction:** Deep convolutional front-end learns optimal filter representations directly from raw audio without information loss from fixed spectrogram filterbanks [7].

### 3.7 Weaknesses
- **Lack of Internal Language Model:** CTC enforces conditional independence between output labels given the acoustic input, making the model prone to phonetic spelling errors and homophone confusion unless paired with an external $n$-gram or neural language model [7, 11].
- **Strict English Alphabet Limitation:** The `base-960h` checkpoint contains only English characters `[A-Z, ']`, making it fundamentally incapable of generating non-Latin scripts (e.g., Devanagari) [8, 9].
- **Acoustic Domain Sensitivity:** Trained on audiobook speech recorded in controlled environments; lacks the broad acoustic noise augmentation present in weakly supervised web corpora [7].

### 3.8 Relevance to Noisy Real-World Audio
Published research indicates that while Wav2Vec 2.0 exhibits strong acoustic modeling on clean speech, its error rate increases significantly on more challenging acoustic conditions (e.g., from 3.4% on `test-clean` to 8.6% on `test-other` in LibriSpeech) [7]. Without an integrated language model or noise-augmented pretraining, acoustic distortion directly disrupts the convolutional feature encodings, leading to increased character substitutions and deletions.

### 3.9 Relevance to the Benchmark
Wav2Vec2 Base-960h provides a benchmark comparison against the sequence-to-sequence Transformer paradigm. It empirically illustrates the trade-off between ultra-low non-autoregressive latency (RTF: 0.083) and acoustic domain/vocabulary boundaries, specifically demonstrating how 10 dB additive noise impacts an acoustic CTC architecture relative to weakly supervised encoder-decoder models.

---

## 4. Noisy-Audio Considerations

### 4.1 Theoretical Robustness in Authoritative Literature
The authoritative literature highlights distinct design decisions governing how ASR architectures handle acoustic degradation:

1. **Background Noise Resilience:**
   - *OpenAI Whisper [1]:* Trained on 680,000 hours of uncurated web audio containing natural ambient noise, variable room acoustics, reverberation, and diverse recording hardware. The authors demonstrated that this diverse pretraining yields strong out-of-the-box robustness across noisy benchmarks without explicit noise-specific data augmentation.
   - *Faster-Whisper [4, 6]:* Shares the identical acoustic parameters as Whisper, retaining its baseline resilience while introducing Silero VAD to suppress silence-triggered hallucinations.
   - *Wav2Vec 2.0 Base-960h [7]:* Pretrained and fine-tuned on the LibriSpeech corpus, which consists of clean and semi-clean read audiobooks. While self-supervised masking enforces acoustic feature robustness, the lack of broad noise-augmented training makes the acoustic encoder vulnerable to severe additive noise.

2. **Accent & Speaker Variability:**
   - *Whisper / Faster-Whisper [1, 2]:* The multilingual and cross-accent web corpus spans thousands of speakers and regional dialects, conferring resilience to speaker idiosyncrasies.
   - *Wav2Vec 2.0 Base-960h [7, 8]:* Trained primarily on North American read English audiobooks, leading to higher sensitivity when exposed to non-native accents or colloquial speech patterns.

3. **Domain Vocabulary & Script Constraints:**
   - *Whisper / Faster-Whisper [1, 3]:* Employs a 51,865-token BPE subword vocabulary supporting multiple scripts and languages, allowing the model to handle diverse vocabulary when properly prompted.
   - *Wav2Vec 2.0 Base-960h [8, 9]:* Restricted to a 32-token uppercase English character set. It lacks word-level lexical priors and cannot physically output Devanagari characters, causing out-of-domain acoustic inputs to be mapped phonetically to Latin characters.

4. **Speaker Rate & Dynamic Range:**
   - Whisper's 30-second cross-attention context can track conversational flow across variable speaking rates [1], whereas Wav2Vec2's frame-wise CTC alignment relies strictly on temporal convolutional downsampling [7, 11].

### 4.2 Local Empirical Noise Benchmark (Synthetic 10 dB AWGN)
Our experimental benchmark evaluates the three systems under a controlled, reproducible noise scenario:
- **Noise Model:** Additive White Gaussian Noise (AWGN) calibrated to **10 dB Signal-to-Noise Ratio (SNR)**.
- **Dataset:** 100 paired clean and noisy utterances sampled from Mozilla Common Voice Hindi (`hi`).
- **Empirical Findings:**
  - Whisper Base and Faster-Whisper Base demonstrated high noise resilience, experiencing modest error degradation ($\Delta\text{WER} = +0.0233$ and $+0.0200$, respectively).
  - Wav2Vec 2.0 Base-960h experienced severe degradation ($\Delta\text{WER} = +0.1099$), a noise vulnerability **5.5× greater** than the Whisper architectures.
- **Important Methodological Note:** Our noisy benchmark measures resilience against **controlled synthetic additive noise (AWGN)**. It does **not** claim to replicate the full complexity of production contact-center audio, which typically incorporates non-stationary environmental babble, dynamic reverberation, packet jitter, telephony bandpass filtering (G.711 / AMR), and overlapping speaker cross-talk.

---

## 5. Relationship Between Research and Experiment

The relationship between the published literature, our controlled empirical benchmark, and technical interpretation is structured as follows:

$$\text{Published Research / Model Cards} + \text{Controlled Empirical Benchmark} = \text{Scenario-Specific Technical Analysis}$$

```
+------------------------------------+      +------------------------------------+
|       Published Literature         |      |    Controlled Empirical Benchmark  |
|  - Radford et al. (2022) [1]       |      |  - 100 clean & 100 noisy samples   |
|  - Baevski et al. (2020) [7]       |  +   |  - Fixed CPU hardware testbed      |
|  - CTranslate2 & SYSTRAN [4, 6]    |      |  - 10 dB AWGN synthetic noise      |
|  - Official Model Cards [2, 8]     |      |  - Standardized WER & RTF metrics  |
+------------------------------------+      +------------------------------------+
                                      |
                                      v
                    +------------------------------------+
                    | Scenario-Specific Engineering Link |
                    |  - Disentangle runtime vs model    |
                    |  - Explain script & vocabulary gap |
                    |  - Measure noise degradation delta |
                    +------------------------------------+
```

### Clarifying the Boundary: Evidence vs. Results vs. Interpretation

| Dimension | Published Evidence (Prior Art) | Our Measured Benchmark (Empirical) | Technical Interpretation |
| :--- | :--- | :--- | :--- |
| **Whisper Base Accuracy** | Radford et al. report 4.2% WER on LibriSpeech clean and competitive multilingual scores [1]. | Measured Clean WER: **1.1018**; Noisy WER: **1.1251** on Common Voice Hindi. | Baseline WER > 1.0 is driven by script/orthography divergence (outputting Hinglish or Urdu tokens against Devanagari references), not acoustic failure. |
| **Faster-Whisper Runtime** | SYSTRAN reports up to 4× speedup on GPU benchmarks with large models [4]. | Measured Clean Latency: **2.336s** (`RTF: 0.569`); **1.89× speedup** over PyTorch on CPU. | Confirms CTranslate2 INT8 quantization delivers real-time performance on CPU without accuracy penalty. |
| **Wav2Vec2 Performance** | Baevski et al. report 3.4% LibriSpeech clean WER [7]. | Measured Clean Latency: **0.376s** (`RTF: 0.083`); $\Delta\text{WER}: \mathbf{+0.1099}$ under noise. | Non-autoregressive CTC provides extreme latency advantages but exhibits severe noise sensitivity and vocabulary mismatch on non-English audio. |

*Note: Published LibriSpeech WER values are not directly comparable to our local benchmark, which evaluates cross-lingual generalization and synthetic noise degradation on Common Voice Hindi speech.*

---

## 6. Comparative Research Table

The following table summarizes the architectural and operational properties of the three benchmarked ASR systems based strictly on authoritative documentation:

| Property | Whisper Base | Faster-Whisper Base | Wav2Vec2 Base-960h |
|---|---|---|---|
| **Model type** | End-to-end multi-task speech recognition & translation model | Optimized inference runtime for Whisper models via CTranslate2 | Self-supervised acoustic representation model with CTC ASR head |
| **Architecture** | Sequence-to-sequence Transformer (encoder-decoder) | Sequence-to-sequence Transformer with fused C++ runtime kernels | Non-autoregressive temporal CNN feature encoder + Transformer encoder |
| **Parameters** | ~74 Million (39M encoder, 35M decoder) | ~74 Million (identical weights, INT8/FP16 quantized) | ~94.4 Million (convolutional encoder + Transformer) |
| **Training data** | 680,000 hours of weakly supervised web audio (438k English, 126k multilingual, 117k translation) | Converted from OpenAI Whisper Base (same 680,000 hours) | 960 hours of English read speech from LibriSpeech (clean & other) |
| **License** | MIT License | MIT License (wrapper & CTranslate2 engine) | Apache License 2.0 |
| **Language scope** | Multilingual (96 languages including Hindi, Urdu, English) | Multilingual (identical to Whisper Base checkpoint) | Monolingual English (32-token character alphabet: `[A-Z, ']`) |
| **Main inference runtime** | PyTorch (`torch.nn`) | CTranslate2 (C++ custom Transformer inference engine) | PyTorch / Hugging Face Transformers (`transformers`) |
| **Quantization support** | FP32 (CPU default), FP16 (CUDA GPU default) | INT8, INT8_float16, INT16, FP16 (optimized for CPU and GPU) | FP32, FP16, dynamic INT8 quantization via PyTorch |
| **Major strengths** | Zero-shot robustness across accents and acoustic environments; unified multi-task capabilities | High throughput; real-time CPU factor (`RTF < 1.0`); low memory footprint; embedded PyAV decoding | Ultra-low deterministic latency; non-autoregressive single-pass forward execution; immune to repetition loops |
| **Major limitations** | Susceptible to autoregressive hallucination loops; high CPU latency; language ambiguity on unprompted dialects | Inherits underlying Whisper checkpoint limitations; requires model conversion step | English-only character set; lacks autoregressive language model; vulnerable to acoustic noise |
