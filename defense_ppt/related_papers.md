# Top 10 Related Papers for Thesis

Surveyed ~120 papers across gradient compression, pipeline parallelism, distributed RL, and gradient staleness. These 10 are the most relevant to your thesis contributions (gradient accumulation + percentile sparsification + buffer decay for pipeline-parallel RL). Papers already in your references.bib are excluded.

---

## 1. Sparse Communication for Distributed Gradient Descent
**Aji & Heafield, EMNLP 2017**

- Drops 99% of gradient values by absolute magnitude and transmits only sparse matrices, with untransmitted residuals accumulated locally (called "memory gradient")
- This is the closest prior work to your accumulate-grads technique — the key difference is they use a fixed top-k cutoff while you use a percentile-based threshold
- Achieves up to 49% wall-clock speedup on MNIST and 22% on neural machine translation without accuracy loss
- Applied to data-parallel SGD for NLP, not pipeline parallelism or RL — your work extends this idea to a fundamentally different communication pattern (activation gradients between pipeline stages)
- The residual accumulation in their "memory gradient" is equivalent to your accumulation buffer with λ=1.0 (no decay), which your results show suffers from staleness

## 2. Scalable Distributed DNN Training Using Commodity GPU Cloud Computing
**Strom, Interspeech 2015**

- Introduces threshold-based gradient compression: only gradients exceeding a fixed absolute threshold are transmitted, with residuals accumulated locally
- Predates DGC and is one of the earliest works on gradient sparsification with error accumulation — achieved 1000x communication reduction for speech model training
- Uses a fixed absolute threshold rather than a percentile-based adaptive threshold like your method — percentile-based adapts naturally as gradient magnitudes change during training
- Demonstrates that aggressive compression (sending <0.1% of gradients) is viable without accuracy loss, though on supervised learning tasks rather than RL
- Your work can be seen as combining Strom's threshold idea with percentile-based adaptivity and adding buffer decay to handle staleness in the RL setting

## 3. Sparsified SGD with Memory
**Stich, Cordonnier & Jaggi, NeurIPS 2018**

- First theoretical proof that top-k sparsified SGD with error compensation (memory) converges at the same rate as vanilla SGD
- Provides the mathematical foundation for why your accumulation buffer approach works — unsent gradients stored in the buffer are guaranteed not to lose information asymptotically
- Shows that the "memory" (accumulation buffer) is essential: without it, sparsified SGD can diverge
- Proves convergence for both convex and non-convex objectives, which covers the neural network optimization in your thesis
- Your buffer decay (λ=0.99) modifies this theoretical framework by deliberately forgetting old residuals — trading the "lossless" guarantee for fresher gradient signal, which your experiments show is beneficial in RL

## 4. Error Feedback Fixes SignSGD and Other Gradient Compression Schemes
**Karimireddy, Rebjock, Stich & Jaggi, ICML 2019**

- Proves that error feedback (accumulating compression residuals) restores SGD-level convergence for any biased compression operator, including top-k sparsification
- Directly relevant because your accumulation buffer IS error feedback — gradients below the percentile threshold are the "compression error" that gets fed back into the next round
- Shows that without error feedback, biased compressors like signSGD and top-k can fail to converge — this explains why naive gradient dropping without accumulation would fail
- Introduces the EF-SGD framework that unifies many compression approaches under one convergence theory
- Your buffer decay is a deliberate departure from pure error feedback: you discount old errors rather than preserving them exactly, which the theory says should hurt convergence but your experiments show helps in the non-stationary RL optimization landscape

## 5. QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding
**Alistarh, Grubic, Li, Tomioka & Vojnovic, NeurIPS 2017**

- Proposes stochastic gradient quantization with tunable precision — each gradient component is randomly rounded to one of a set of discrete levels
- Represents the quantization approach to gradient compression, complementary to your sparsification approach — quantization reduces bits per value while sparsification reduces the number of values sent
- Provides provable convergence guarantees with a smooth tradeoff between compression ratio and convergence speed
- Achieves 1.8x speedup training ResNet-152 on 16 GPUs with 4-bit quantization
- Could potentially be combined with your percentile-based sparsification: quantize the top 10% of gradients before sending, for even greater compression

## 6. signSGD: Compressed Optimisation for Non-Convex Problems
**Bernstein, Wang, Azizzadenesheli & Anandkumar, ICML 2018**

- Extreme 1-bit compression: transmits only the sign (+1 or -1) of each gradient component, achieving 32x compression
- Uses majority vote aggregation across workers, which provides natural Byzantine fault tolerance (up to 50% adversarial workers)
- Represents the opposite end of the compression spectrum from your approach — signSGD compresses all dimensions equally, while your method sends some dimensions at full precision and drops others entirely
- Proves convergence under the assumption that gradient noise has bounded variance, though subsequent work (Karimireddy et al.) showed it needs error feedback to converge reliably
- Relevant as a comparison point: your 90th percentile sparsification sends 10% of values at full precision, while signSGD sends 100% of values at 1-bit precision — both achieve significant compression but through fundamentally different mechanisms

## 7. PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization
**Vogels, Karimireddy & Jaggi, NeurIPS 2019**

- Low-rank gradient compression using power iteration: approximates the gradient matrix with a rank-r factorization, sending two small matrices instead of the full gradient
- The only gradient compression method that consistently achieves wall-clock speedups over optimized SGD with state-of-the-art communication backends (NCCL)
- Represents a third compression paradigm (low-rank) distinct from both sparsification (your approach) and quantization (QSGD/signSGD)
- Compatible with all-reduce aggregation, making it practical for data-parallel training — your pipeline-parallel setting uses point-to-point communication, where sparsification may be more natural
- Relevant as a potential alternative or complement to your approach: low-rank compression could be applied to the activation gradients in pipeline parallelism

## 8. Asynchronous Stochastic Gradient Descent with Delay Compensation (DC-ASGD)
**Zheng, Meng, Wang, Chen, Yu, Ma & Liu, ICML 2017**

- Compensates for gradient staleness in asynchronous SGD using Taylor expansion of the gradient function and an approximate Hessian diagonal
- Directly relevant to your staleness problem: they correct stale gradients mathematically, while you mitigate staleness through exponential decay of the accumulation buffer — both address the same fundamental issue
- Adopted in production systems (Microsoft CNTK, Apache MXNet, PaddlePaddle), demonstrating practical viability of staleness correction
- Their approach requires computing an approximate Hessian which adds computation overhead; your decay approach is simpler (just multiply by λ) but less theoretically grounded
- Shows that gradient staleness is a well-recognized problem in distributed training that degrades convergence, supporting your experimental finding that no-decay accumulation underperforms

## 9. PipeMare: Asynchronous Pipeline Parallel DNN Training
**Yang, Zhang, Li, Re, Aberger & De Sa, MLSys 2021**

- Directly addresses gradient staleness in pipeline parallelism — uses learning rate rescheduling and discrepancy correction to tolerate asynchronous weight updates between pipeline stages
- Most directly comparable to your work: both tackle the same problem (staleness in pipeline-parallel training) but with different solutions — PipeMare corrects for weight version mismatches, while you handle gradient accumulation staleness
- Achieves up to 2.7x less memory and 4.3x higher pipeline utilization compared to synchronous approaches like GPipe
- Their "discrepancy correction" adjusts gradients based on the difference between current and stale weights, similar in spirit to your buffer decay but applied to weight staleness rather than gradient accumulation staleness
- Applied to supervised learning (image classification, language modeling), not RL — your work demonstrates that staleness mitigation is equally important in the RL setting where the optimization landscape is non-stationary

## 10. Asynchronous Methods for Deep Reinforcement Learning (A3C)
**Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver & Kavukcuoglu, ICML 2016**

- Foundational distributed RL paper: multiple CPU threads run parallel environment copies and asynchronously update a shared global network, eliminating experience replay
- Introduces the actor-critic framework for distributed RL that your pipeline-parallel architecture builds upon — your Machine 0 (CNN) + Machine 1 (actor-critic) split is a pipeline-parallel version of the actor-critic structure A3C popularized
- Demonstrates that asynchronous gradient updates (inherently stale) can work in RL, though they note performance degrades with too much staleness — consistent with your finding that uncontrolled gradient accumulation hurts performance
- Replaced by IMPALA (which you already cite) for large-scale training, but remains the conceptual ancestor of most distributed on-policy RL systems
- Relevant because A3C's success with stale gradients in RL provides precedent for your work: if some staleness is tolerable in RL, then controlled staleness (via buffer decay) should also work

---

## Summary Table

| # | Paper | Year | Core Idea | Relation to Thesis |
|---|-------|------|-----------|-------------------|
| 1 | Aji & Heafield | 2017 | Top-k sparsification + residual accumulation | Closest precursor to accumulate-grads |
| 2 | Strom | 2015 | Threshold-based compression + residual | Early threshold compression, no adaptivity |
| 3 | Stich et al. | 2018 | Convergence proof for sparsified SGD + memory | Theoretical foundation for accumulation buffer |
| 4 | Karimireddy et al. | 2019 | Error feedback fixes biased compressors | Theory behind why accumulation works |
| 5 | Alistarh et al. | 2017 | Stochastic gradient quantization | Alternative compression: reduce bits |
| 6 | Bernstein et al. | 2018 | 1-bit sign compression | Alternative compression: extreme quantization |
| 7 | Vogels et al. | 2019 | Low-rank gradient compression | Alternative compression: low-rank |
| 8 | Zheng et al. | 2017 | Delay compensation via Taylor expansion | Staleness correction (mathematical approach) |
| 9 | Yang et al. | 2021 | Async pipeline with staleness correction | Pipeline parallelism + staleness (closest setting) |
| 10 | Mnih et al. | 2016 | Async distributed RL (A3C) | Foundational distributed RL with stale gradients |

## Coverage

- **Gradient sparsification with accumulation:** Papers 1, 2, 3
- **Error feedback theory:** Paper 4
- **Alternative compression approaches:** Papers 5, 6, 7
- **Gradient staleness mitigation:** Papers 8, 9
- **Distributed RL:** Paper 10
