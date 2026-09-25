# Authoritative References & Source Documentation

This document catalogs the authoritative primary literature, official model cards, software repositories, and documentation consulted for the architectural analysis in `model_research.md`.

---

## 1. OpenAI Whisper

### [1] Primary Research Paper
- **Title:** Robust Speech Recognition via Large-Scale Weak Supervision
- **Authors:** Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever
- **Year:** 2022
- **Source Type:** Peer-Reviewed Pre-print / Research Paper
- **Official URL:** [https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)
- **Key Evidence Extracted:** Sequence-to-sequence Transformer architecture, 80-channel log-Mel front-end, 680,000-hour weakly supervised corpus breakdown (438k English, 126k multilingual, 117k translation), 30-second chunking mechanism, multi-task conditioning tokens, zero-shot robustness characteristics, and documented hallucination failure modes.

### [2] Official Model Card
- **Title:** Whisper Model Card
- **Authors:** OpenAI
- **Year:** 2022
- **Source Type:** Official Model Card
- **Official URL:** [https://github.com/openai/whisper/blob/main/model-card.md](https://github.com/openai/whisper/blob/main/model-card.md)
- **Key Evidence Extracted:** Intended use cases, parameter counts across variants (Base: 74M parameters), language coverage across 96 languages, uneven accuracy across low-resource dialects, and bias considerations.

### [3] Official Repository & Software Implementation
- **Title:** Whisper: Robust Speech Recognition via Large-Scale Weak Supervision (Codebase)
- **Authors:** OpenAI
- **Year:** 2022 (maintained through 2026)
- **Source Type:** Official Open-Source Code Repository & Documentation
- **Official URL:** [https://github.com/openai/whisper](https://github.com/openai/whisper)
- **Key Evidence Extracted:** PyTorch implementation details, FP32 CPU fallback behaviors, requirement of FFmpeg on system PATH, tokenizer vocabulary structure (51,865 multilingual BPE tokens), and MIT License specification.

---

## 2. Faster-Whisper & CTranslate2

### [4] Faster-Whisper Official Repository
- **Title:** Faster Whisper: Faster Whisper transcription with CTranslate2
- **Authors:** SYSTRAN / Guillaume Klein
- **Year:** 2023 (maintained through 2026)
- **Source Type:** Official Open-Source Code Repository
- **Official URL:** [https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Key Evidence Extracted:** Clarification as an optimized inference runtime (not an independently trained model), CTranslate2 integration, memory footprint reductions, PyAV in-memory decoding integration, and Silero VAD filtering support.

### [5] Faster-Whisper License
- **Title:** Faster-Whisper MIT License
- **Authors:** SYSTRAN
- **Year:** 2023
- **Source Type:** Official Software License File
- **Official URL:** [https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE)
- **Key Evidence Extracted:** Confirmation of MIT License permissions for commercial and non-commercial runtime deployment.

### [6] CTranslate2 Architecture & Documentation
- **Title:** CTranslate2: Fast Inference Engine for Transformer Models
- **Authors:** OpenNMT / Guillaume Klein, Dakun Zhang, Vincent Nguyen, Jean Senellart
- **Year:** 2020 (v4.x documentation)
- **Source Type:** Official Software Documentation & Technical Report
- **Official URL:** [https://opennmt.net/CTranslate2/](https://opennmt.net/CTranslate2/)
- **Key Evidence Extracted:** INT8 and FP16 quantization kernels (Intel MKL/oneDNN and OpenBLAS on CPU, cuBLAS/TensorRT on GPU), fused multi-head attention kernels, optimized key-value (KV) cache memory reuse, parallel worker execution model, and published reference speedups.

---

## 3. Wav2Vec 2.0

### [7] Primary Research Paper
- **Title:** wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
- **Authors:** Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli
- **Year:** 2020
- **Source Type:** Peer-Reviewed Research Paper (NeurIPS 2020)
- **Official URL:** [https://arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477)
- **Key Evidence Extracted:** Self-supervised pretraining via contrastive loss over quantized latent representations, temporal convolutional feature encoder (7 layers, stride 5/2/2/2/2/2/2), 12-layer Transformer context network, Connectionist Temporal Classification (CTC) fine-tuning objective, and LibriSpeech 960h evaluation metrics.

### [8] Official Model Card
- **Title:** Wav2Vec2-Base-960h Model Card
- **Authors:** Meta AI (formerly Facebook AI Research) / Hugging Face
- **Year:** 2020
- **Source Type:** Official Model Card
- **Official URL:** [https://huggingface.co/facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- **Key Evidence Extracted:** Target language (English), training corpus (LibriSpeech 960 hours clean + other), 16 kHz sampling requirement, character-level alphabet vocabulary, and intended speech-to-text usage.

### [9] Official Model Configuration & Architecture Specification
- **Title:** Wav2Vec2Config & Pretrained Weights
- **Authors:** Meta AI / Hugging Face
- **Year:** 2020
- **Source Type:** Official Architecture Configuration (`config.json`)
- **Official URL:** [https://huggingface.co/facebook/wav2vec2-base-960h/blob/main/config.json](https://huggingface.co/facebook/wav2vec2-base-960h/blob/main/config.json)
- **Key Evidence Extracted:** Parameter verification: 94,396,448 parameters (~94.4M), hidden size 768, 8 attention heads, 12 hidden layers, intermediate dimension 3072, vocabulary size 32 tokens (`[A-Z, ', \|, <pad>, <s>, </s>, <unk>]`).

### [10] Official Model License
- **Title:** Fairseq / Wav2Vec 2.0 Apache 2.0 License
- **Authors:** Meta AI
- **Year:** 2020
- **Source Type:** Official Software License File
- **Official URL:** [https://github.com/facebookresearch/fairseq/blob/main/LICENSE](https://github.com/facebookresearch/fairseq/blob/main/LICENSE)
- **Key Evidence Extracted:** Permissive commercial use, modification, and distribution under Apache License 2.0.

---

## 4. Auxiliary Theoretical & Corpus Citations

### [11] CTC Loss Formulation
- **Title:** Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
- **Authors:** Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber
- **Year:** 2006
- **Source Type:** Peer-Reviewed Research Paper (ICML 2006)
- **Official URL:** [https://dl.acm.org/doi/10.1145/1143844.1143891](https://dl.acm.org/doi/10.1145/1143844.1143891)
- **Key Evidence Extracted:** Mathematical formulation of non-autoregressive sequence alignment using dynamic programming over character emission probabilities and blank tokens.

### [12] Evaluation Corpus Specification
- **Title:** Common Voice: A Massively-Multilingual Speech Corpus
- **Authors:** Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis Tyers, Gregor Weber
- **Year:** 2020
- **Source Type:** Peer-Reviewed Paper (LREC 2020)
- **Official URL:** [https://aclanthology.org/2020.lrec-1.520/](https://aclanthology.org/2020.lrec-1.520/)
- **Key Evidence Extracted:** Creative Commons CC0 licensing, crowdsourced recording methodology, acoustic diversity, and validation voting mechanics.
