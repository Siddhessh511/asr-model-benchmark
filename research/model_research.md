# Primary Model Research & Architectural Analysis

This document provides a rigorous architectural, algorithmic, and operational analysis of the three Automatic Speech Recognition (ASR) systems evaluated in this comparative benchmark:
1. **OpenAI Whisper (`base`)**
2. **Faster-Whisper (`base` runtime via CTranslate2)**
3. **Wav2Vec 2.0 (`facebook/wav2vec2-base-960h`)**

All analysis strictly distinguishes between **[Published Facts]** (from primary peer-reviewed literature, official repositories, and model cards), **[Our Experiment]** (empirically measured during this benchmark), and **[Our Technical Interpretation]**.

---

## 1. OpenAI Whisper (`base`)

### 1.1 Architecture & Tokenization
- **[Published Fact]** Proposed by Radford et al. (OpenAI, 2022) in *"Robust Speech Recognition via Large-Scale Weak Supervision"*.
- **[Published Fact]** Whisper employs an encoder-decoder sequence-to-sequence Transformer architecture.
  - **Acoustic Front-End:** Raw audio is resampled to 16 kHz and converted to an 80-channel log-magnitude Mel-spectrogram computed over 25ms windows with a 10ms hop size.
  - **Encoder:** Two convolutional layers with a filter width of 3 and stride 2 compress the acoustic feature sequence along the temporal dimension, followed by sinusoidal positional encodings and a stack of Transformer encoder blocks (6 layers, 8 heads, hidden dimension 512 for `base`).
  - **Decoder:** Transformer decoder stack (6 layers, 8 heads, hidden dimension 512) utilizing causal self-attention and cross-attention over encoder activations.
  - **Tokenization:** Multilingual Byte-Pair Encoding (BPE) with a 51,865-token vocabulary, shared across text transcription, translation, voice activity detection (VAD), and timestamp estimation.
- **[Published Fact]** Parameter count for `base` is approximately **74 million parameters** (39M encoder, 35M decoder).

### 1.2 Training Data & Methodology
- **[Published Fact]** Trained on **680,000 hours** of weakly supervised internet audio collected from public web sources:
  - 438,000 hours of English speech.
  - 126,000 hours of multilingual speech across 96 languages (including Hindi, Urdu, Bengali, Marathi, etc.).
  - 117,000 hours of speech-to-English translation data.
- **[Published Fact]** Trained across multiple multitask prompt tokens: `<|startoftranscript|>`, `<|language|>`, `<|transcribe|>`, `<|translate|>`, and `<|notimestamps|>`.

### 1.3 Licensing & Governance
- **[Published Fact]** Open-source under the **MIT License**. Permissive for commercial and non-commercial deployment without proprietary royalty constraints.

### 1.4 Hardware Requirements & Resource Footprint
- **[Published Fact]** Official OpenAI implementation runs on standard PyTorch (`torch.nn`).
- **[Our Experiment]** On an Intel 8-core / 12-thread host CPU:
  - Peak resident memory (RSS): **844.0 MB**.
  - Average clean inference time: **4.416s** per utterance (`RTF: 1.149`).
  - Requires host-level installation of FFmpeg for audio decoding subprocesses.

### 1.5 Known Strengths
- **[Published Fact]** Exceptional zero-shot domain robustness across diverse conversational styles, accents, and recording equipment compared to models trained on clean single-corpus datasets.
- **[Published Fact]** Built-in multilingual transcription and inverse text normalization (ITN) natively handling numerals, dates, and punctuation.

### 1.6 Known Weaknesses & Failure Modes
- **[Published Fact]** Autoregressive decoders are susceptible to **hallucination loops**, silent audio hallucination, and phrase repetition under degraded signal-to-noise ratios.
- **[Our Experiment]** On low-resource languages or unprompted dialectal speech, the `base` model's language identification head frequently alternates between closely related orthographies. In our Hindi evaluation, Whisper `base` frequently outputted Romanized "Hinglish" or Perso-Arabic (Urdu) tokens instead of Devanagari script, resulting in high word-level token substitution when evaluated against standard Devanagari references.

### 1.7 Deployment Characteristics
- High computational complexity due to $O(N)$ autoregressive token generation steps in the decoder. Standard PyTorch execution lacks fused quantization kernels on CPU by default.

---

## 2. Faster-Whisper (`base`)

### 2.1 Technical Clarification: Implementation vs. Model Family
- **[Published Fact]** Faster-Whisper is **not an independently trained ASR model family**. It is a reimplementation of OpenAI's Whisper architecture using **CTranslate2**, a custom C++ inference engine designed by Systran for Transformer models.
- **[Published Fact]** Weights are bit-for-bit converted from the official OpenAI Whisper checkpoints, ensuring identical parameter semantics while altering the runtime execution graph.

### 2.2 Inference Acceleration Mechanics
- **[Published Fact]** Implements 8-bit integer quantization (INT8) and 16-bit floating point (FP16) on both CPU (via Intel MKL / oneDNN and OpenBLAS) and GPU (via cuBLAS and TensorRT).
- **[Published Fact]** Incorporates key optimization techniques:
  - Fused layer normalization and activation kernels.
  - Efficient KV-cache memory allocation reducing tensor reallocation overhead.
  - Native integration with PyAV (libavformat/libavcodec) for fast in-memory audio decoding without spawning external `ffmpeg` subprocesses.
  - Built-in Voice Activity Detection (VAD) via Silero VAD to suppress silence hallucination.

### 2.3 Licensing
- **[Published Fact]** Open-source under the **MIT License**.

### 2.4 Hardware Requirements & Measured Benchmark
- **[Our Experiment]** On the same Intel 8-core CPU hardware:
  - Peak resident memory (RSS): **619.2 MB** (**26.6% lower** than OpenAI Whisper).
  - Average clean inference time: **2.336s** per utterance (`RTF: 0.569`, running **1.89× faster** than PyTorch Whisper).
  - Noisy inference time: **3.502s** per utterance (`RTF: 0.842`).
  - Disk footprint: ~145 MB (INT8 model binary).

### 2.5 Known Strengths
- Enables real-time transcription on commodity CPU hardware (`RTF < 1.0`).
- Significantly lower memory bandwidth pressure, allowing higher concurrent stream density per node.

### 2.6 Known Weaknesses
- Inherits the underlying parameter limitations of the original Whisper checkpoint. In the case of `base`, language selection ambiguities for Hindi speech remain identical to vanilla Whisper unless guided by explicit prompt tokens.

---

## 3. Wav2Vec 2.0 (`facebook/wav2vec2-base-960h`)

### 3.1 Architecture & Tokenization
- **[Published Fact]** Proposed by Baevski et al. (Meta AI, 2020) in *"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"*.
- **[Published Fact]** Acoustic-only, non-autoregressive architecture:
  - **Feature Encoder:** Multi-layer temporal convolutional network (7 conv blocks) operating directly on raw 16 kHz audio waveforms with temporal strides, outputting latent feature vectors $Z$ representing 25ms of audio every 20ms.
  - **Context Network:** 12-layer Transformer encoder (hidden size 768, 8 attention heads, 95 million parameters) computing contextualized representations $C$.
  - **Prediction Head:** Linear projection layer mapping Transformer output states to character logits, trained via Connectionist Temporal Classification (CTC) loss.
  - **Tokenization / Vocabulary:** Character-level alphabet consisting of 32 tokens: the 26 uppercase English letters `A-Z`, apostrophe `'`, blank token, unknown `<unk>`, and word delimiter `|`.

### 3.2 Training Data & Methodology
- **[Published Fact]** Pretrained via self-supervised contrastive learning on **960 hours** of unannotated English audio from the LibriSpeech corpus (`train-clean-100`, `train-clean-360`, `train-other-500`).
- **[Published Fact]** Fine-tuned for end-to-end speech recognition using supervised CTC loss on the 960 hours of transcribed LibriSpeech audio.

### 3.3 Licensing
- **[Published Fact]** Open-source under the **Apache 2.0 License**. Permissive for enterprise production.

### 3.4 Hardware Requirements & Measured Benchmark
- **[Our Experiment]** On host CPU:
  - Peak resident memory (RSS): **940.0 MB**.
  - Average clean inference time: **0.376s** (`RTF: 0.083`, **12.0× faster than real-time**).
  - Average noisy inference time: **0.368s** (`RTF: 0.081`).
  - Model disk size: ~378 MB.

### 3.5 Known Strengths
- Ultra-low latency and deterministic execution. Non-autoregressive forward pass generates all token alignments in parallel in a single forward pass without sequential decoding loops.

### 3.6 Known Weaknesses & Cross-Lingual Failure Mode
- **[Published Fact]** No native internal language model. Lacks word-level or phrase-level autoregressive conditioning, making it sensitive to acoustic noise and homophones unless coupled with an external $n$-gram or neural language model.
- **[Our Experiment]** **Vocabulary Script Limitation:** Because `facebook/wav2vec2-base-960h` has an English character vocabulary, it cannot physically generate Devanagari Hindi orthography. Its acoustic CNN encoder maps Hindi acoustic phones to English characters (e.g. transcribing *"voh bahut"* as `BAHERBAHUD`).
- **[Our Experiment]** **Noise Sensitivity:** Under 10 dB SNR noise, Wav2Vec2 suffered a **+0.1099 (+10.99%) WER degradation**, the largest degradation among all three evaluated architectures.

---

## 4. Architectural Synthesis

| Dimension | OpenAI Whisper (`base`) | Faster-Whisper (`base`) | Wav2Vec 2.0 (`base-960h`) |
| :--- | :--- | :--- | :--- |
| **Model Class** | Seq2Seq Encoder-Decoder | Seq2Seq Encoder-Decoder | Acoustic CTC Encoder |
| **Decoding Paradigm** | Autoregressive (token-by-token) | Autoregressive (INT8 C++ engine) | Non-autoregressive (single pass) |
| **Input Feature** | 80-channel log-Mel spectrogram | 80-channel log-Mel spectrogram | Raw 16 kHz PCM waveform |
| **Parameter Count** | ~74 Million | ~74 Million (INT8 quantized) | ~95 Million |
| **Pretraining Scale** | 680,000 hours (weakly supervised) | 680,000 hours (OpenAI checkpoint) | 960 hours (self-supervised) |
| **Clean Latency (CPU)** | 4.416s (`RTF: 1.149`) | 2.336s (`RTF: 0.569`) | 0.376s (`RTF: 0.083`) |
| **Noisy Latency (CPU)** | 4.775s (`RTF: 1.131`) | 3.502s (`RTF: 0.842`) | 0.368s (`RTF: 0.081`) |
| **Noise Robustness** | High ($\Delta\text{WER}: +0.0233$) | High ($\Delta\text{WER}: +0.0200$) | Vulnerable ($\Delta\text{WER}: +0.1099$) |
| **License** | MIT | MIT | Apache 2.0 |
