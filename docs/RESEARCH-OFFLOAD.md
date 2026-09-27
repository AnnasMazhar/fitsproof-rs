# RESEARCH-OFFLOAD.md — five offload papers, read in full, mapped to our hardware

Lane: `research/offload` (opencode). Analysis only — no production code in this lane.
Binding spec: `OFFLOAD-ENGINE.md` (C1–C6, honesty rules). This document supplies the *why* and the
*how*, with section citations.

**Read in full, not from abstracts.** Sources read end-to-end:

| # | paper | text read |
|---|---|---|
| 1 | M2Cache | arXiv HTML v2, `https://arxiv.org/html/2410.14740v2` — §1–§8 + references (v2 has no appendix) |
| 2 | HiFC | NeurIPS 2025 PDF text, `~/.hermes/cache/web/papers.neurips.cc-fed1b62c7b.md` — main §1–§7 + Appendices A–H + checklist |
| 3 | INF2 | `~/.hermes/cache/web/arxiv.org-79cdb2e4e8.md` — §1–§10 + references (no appendix) |
| 4 | Tutti | `~/.hermes/cache/web/arxiv.org-265736fdd2.md` — §1–§5 + references (no appendix) |
| 5 | CALVO | `~/.hermes/cache/web/arxiv.org-21e6686938.md` — §1–§5 + references (no appendix) |

Section numbers below are the papers' own. Where a paper's claim is a number, it is labelled
**theirs**. No number in this document is presented as ours: every number we will own must later
come from a committed JSON under `reports/` (spec: *Baseline honesty* / *Every number*).

---

## 0. The hardware we actually have (measured, commands shown)

These are inventory facts, not performance claims. Commands were run on 2026-09-27.

**P500 (`openclaw`, this host)**

```
$ lsblk -d -o NAME,ROTA,SIZE,MODEL
sda       0 447.1G SAMSUNG MZ7LM480HCHP-00005
$ free -g   -> total 31   $ nproc -> 8
$ lspci | grep -i nvidia
03:00.0 VGA compatible controller: NVIDIA Corporation GM206GL [Quadro M2000] (rev a1)
```

- One disk: a **447 GB SATA** Samsung SSD (MZ7LM480HCHP). **No NVMe device exists on this box.**
- 31 GB DRAM, 8 cores, Quadro M2000 (4 GB, Maxwell, GM206), **no CUDA toolkit** (spec + README).
- Consequence: sequential read tops out around SATA class (~0.5 GB/s), not the ~5 GB/s NVMe class
  the KV-cache papers measure on. Tier-2 bandwidth is the number that will decide every experiment.

**papi (192.168.1.56, over SSH)**

```
$ ssh papi 'lsblk -d -o NAME,ROTA,SIZE,MODEL; nproc; free -g'
sda       1 931.5G ST1000DM003-1ER162      <- rotational HDD
sdb       0 223.6G ADATA SU630             <- SATA QLC SSD (ROTA=0)
12   /   total 15
$ ssh papi 'lscpu | grep "Model name"'  -> Intel(R) Xeon(R) CPU X5670 @ 2.93GHz
```

- 12 vCPU (Westmere X5670), 15 GB RAM, **two** storage classes: a 224 GB **SATA QLC SSD** and a
  1 TB **HDD**. Again **no NVMe, no GPU**.
- A 27B-class q4_K_M model is ≈16–18.5 GB (published GGUF sizes; **no 27B GGUF is on either disk
  yet** — verify with `fitsproof plan` once downloaded). Against a 13 GB RSS budget the weights
  **cannot be resident**: papi is necessarily an SSD-streaming machine. That is exactly C1.

**What this means structurally:** on P500 the weight tier is DRAM-resident + SSD source (the
"VRAM tier" does not exist for us — M2000 has no compute path in this engine). On papi the tier
chain is SSD → DRAM window → compute. Neither machine has the hardware any of the five papers
uses for its headline result. Everything below separates the *portable mechanism* from the
*hardware claim*.

---

## 1. M2Cache — arXiv 2410.14740 (v2, 23 Oct 2024)

**Citation:** Jie Peng, Zhang Cao, Huaizhi Qu, Zhenyu Zhang, Chang Guo, Yanyong Zhang, Zhichao Cao,
Tianlong Chen, *"Harnessing Your DRAM and SSD for Sustainable and Accessible LLM Inference with
Mixed-Precision and Multi-level Caching"*, arXiv:2410.14740v2. No venue listed in v2.

### Mechanism (sections cited)

M2Cache does two coupled things. (a) **Dynamic sparse mixed-precision inference (§5.2):** a Deja-Vu-style
low-rank predictor assigns each FFN neuron a score; active neurons are loaded from DRAM, and their
precision is chosen by score — higher score → higher precision — so a fixed memory budget buys
*all* neurons at mixed bit-widths instead of few neurons at full width. The high/low precision
ratio is not guessed: **Offline Neuron Ratio Search (§5.2, Algorithm 1, Eq. 2)** sweeps
`(r_low, r_high)` under a fixed budget and minimises an uncertainty estimate
`UQEst = −Σ Σ LLM^i_k log LLM^i_k` over wikitext generations. (b) **Three-level cache (§5.3–5.4):**
HBM keeps a *neuron-level* LRU per layer — actually an **Adjacent Token Update (ATU)** policy that
only refreshes neurons that differ between adjacent tokens, giving ~80% hit ratio with "nearly
zero" management overhead (§5.3, Fig. 6–7); DRAM keeps a **layer-aware FIFO** with a two-level
split (§5.4: *fixed area* = first n layers so a new token's decode does not re-read them,
*dynamic area* = layers ahead of the current one); SSD holds **all** FFN parameters (§5.4).
Preloading is **pattern-aware**: a preloader pulls whole *missing* layers SSD→DRAM, launched from
≥2 layers ahead, because "the one-layer neuron preloading time (SSD→DRAM) is approximately twice
as long as one layer inference time" (§5.4). The paper explicitly analyses the alternative
(neuron-level preload using the predictor) and rejects it: predictor accuracy falls to ~80% at
two layers ahead, so mispredicted neurons would have to be fetched on the critical path anyway
(§5.4).

Measured tier ratios in the paper (**theirs**, Figs. 4–5): DRAM→HBM ≈ 10× slower than HBM-resident;
SSD ≈ 8× slower than DRAM and ≈85× slower than HBM; and neuron-granularity GPU copies are ~10×
slower than DRAM copies, which is *why* their HBM cache is layer-partitioned with contiguous
per-layer units (§5.3).

### Hardware prerequisites

- RTX 3090 24 GB HBM, 64 GB DRAM, 1 TB **PCIe 3.0×4** SSD, AMD 6950x, Ubuntu 22.04 (§6.2); one
  CPU core deliberately reserved for cache management (§1).
- **CUDA streams** for DRAM↔HBM transfer and **separate I/O threads** for SSD→DRAM preloading,
  overlapped with compute (§6.1).
- A trained **Deja-Vu FFN predictor** per layer (§5.2, §6.1: "we adopt the training method from
  Deja Vu").
- Baseline: DeepSpeed ZeRO-Infinity, batch size 1, wikitext prompts 64–128 tokens (§6.3).

### Verdict: **PARTIALLY REPRODUCIBLE (software part only)**

| sub-mechanism | verdict | why |
|---|---|---|
| C1 three-tier weight residency + pattern-aware preloading + two-level DRAM cache (§5.4) | **REPRODUCIBLE** | pure software: file offsets, a prefetch thread, a DRAM window. Needs only SSD + DRAM + CPU — precisely papi. |
| C2 importance-ranked mixed precision (§5.2, Algorithm 1) | **REPRODUCIBLE (static version)** | the *ratio search* is offline CPU work over a calibration corpus. The *dynamic* part (per-token Deja-Vu predictor deciding which neurons are active) is **NOT REPRODUCIBLE (hardware/scope)** — it needs trained per-layer predictors and a GPU offload path this engine does not have. |
| neuron-level HBM cache / ATU (§5.3) | **NOT REPRODUCIBLE (hardware)** | no HBM tier, no CUDA streams (§6.1). The M2000 cannot hold the model (4 GB) and has no compute path here. |
| carbon-emission accounting (§2.2 Eq. 1, §6.4) | **NOT REPRODUCIBLE (not our goal)** | gCO₂ requires their energy model (§6.4 assumes 26 W/256 GB DRAM, 2 W SSD, 820 gCO₂/kWh). Out of scope for C1–C6. |

The paper's own limits (**§5.5.2**): "still high inference latency" from SSD reliance, and it
"can only work for small batch size scenarios" because the Deja-Vu predictor degrades at large
batch. Their ablation (**§6.4**): +MP Inference ≈ +1 tok/s; +LRU cache raises it to 4.62 tok/s;
+SSDs removes 22 GB of DRAM "without compromising inference performance". Their quality table
(**§6.5**, Table 14) is honest but soft: LLaMA-7B HumanEval 0.1280 → 0.1159, PIQA 0.7008 → 0.7122;
LLaMA-13B HumanEval 0.1707 → 0.1280 (a 25% *relative* drop the text calls "negligible").

### Falsifiable experiment (ours, proposed)

> **X2 — preload lookahead on papi.** `scripts/bench/preload.rs` runs the 27B q4_K_M model
> streamed from papi's SATA SSD with lookahead `W ∈ {0,1,2,4,8}` layers (W=0 = read-on-demand =
> sequential read baseline), fixed prompt set, 64 generated tokens, 3 repetitions.
> **Metrics:** `tok_s`, decode stall fraction (time blocked in a storage read / decode time),
> preload hit rate, SSD MB/s.
> **Pass:** some `W ≥ 2` gives `tok_s ≥ 1.15 × tok_s(W=0)` **and** stall fraction < 5% **and**
> preload hit rate ≥ 95%.
> **Fail:** mechanism is **NOT PROVEN**; we ship W=0 (plain mmap/sequential read) and say so.

> **X3 — mixed precision under a fixed budget (C2, static version).** Same prompt set, fixed
> budget B = 13 GB on papi. Compare (a) uniform int4, (b) uniform int8 (if it fits), (c) mixed
> policy: attention tensors int8 + FFN tensors int4, ratio chosen by a port of Algorithm 1's
> sweep on a wikitext calibration file.
> **Metrics:** bytes (`fit` prediction from `src/cost.rs`), quality check on a fixed prompt set
> (≥20 prompts, top-5 token agreement against the int8/fp16 reference).
> **Pass:** mixed policy quality ≥ uniform-int4 quality while both fit B, and mixed uses ≥ 15%
> fewer bytes than uniform-int8.
> **Fail:** C2 **NOT PROVEN** — we stay on GGUF's own per-tensor types.

### Their headline number (theirs, not ours)

Abstract: "significantly improves the token generation speed by up to **×10.51** … carbon
emission reduction up to **×7.67**" vs DeepSpeed ZeRO-Infinity. §1: ×7 latency reduction on
LLaMA-7B, ×14 on LLaMA-13B, 0.3835 tok/s (LLaMA-70B) and 0.312 tok/s (Falcon-40B) on a single
3090; ×1.47 from mixed-precision inference alone, ×2.15 from the multi-level cache alone.
**All on an RTX 3090 with CUDA. Not transferable to us.**

---

## 2. HiFC — NeurIPS 2025, "High-efficiency Flash-based KV Cache Swapping"

**Citation:** Inho Jeong, Sunghyeon Woo, Sol Namkung, Dongsuk Jeon (Seoul National University /
SK hynix / NAVER Cloud), *"HiFC: High-efficiency Flash-based KV Cache Swapping for Scaling LLM
Inference"*, 39th Conference on Neural Information Processing Systems (NeurIPS 2025).

### Mechanism (sections cited)

HiFC deletes host DRAM from the KV swap path. (a) **FC block allocator (§3.2):** it extends vLLM's
allocator with Flash-Cache blocks alongside GPU/CPU blocks, fixed size 32–128 tokens, allocated
with an **append policy in physical order** so evictions become sequential I/O and stale logical
addresses are never reused — measured **WAF 1.02 vs the SSD spec's 1.4** (§3.2, validated in §5.5
Table 4 and Appendix H Eq. 16, from SMART "Data Units Written": 487,595 DUs × 512 kB against
7,440 × 32 MB host writes). (b) **pSLC regions (§4, Appendix D):** the system runs *exclusively*
inside a pSLC zone (20% of the selected SSDs' capacity ≈ 200 GiB), 30,000 P/E cycles vs 3,000 for
TLC → **8× TBW** (Appendix D, Eqs. 8–10) and 6,000 TiB theoretical write budget; empirically an
8.3-year predicted lifetime at 1.98 TiB/day (Appendix D.1, Table 15). (c) **GDS path (§3.2):**
CUDA kernels move KV tensors SSD↔HBM directly via NVIDIA GPUDirect Storage, 4 KB-aligned, up to
16 I/O threads, precomputed per-layer byte offsets — **Appendix B, Eqs. 2–7**
(`block_size_bytes = H·S·D·T`, `layer_offset = l·(2·B·block_size_bytes)`, key/value offsets at
`k/v × B × block_size_bytes`), with a worked example for DS-Qwen-32B (L=64, B=3200, H=8, S=128,
D=128, T=2 → layer 63 starts at 98.44 GiB). Sustained **>4.7 GiB/s in the pSLC region** (§3.2);
Appendix E hardware: 2× A100 80 GiB, 256 GiB DDR4, **1 TB NVMe Gen4 flash cache**, GDS 1.8.1.2,
vLLM 0.6.6. (d) **Slack, not speed, is the real argument (§5.4, Appendix G):** swaps become
invisible when `T_swap ≤ T_gap` where `T_gap = ΔTPS · T_lat` (Eqs. 12–15); their worked run has a
266 ms swap absorbed by a 2.07 s pipeline gap. Block size matters: **64 tokens is the sweet spot**
(§5.6, Fig. 6b), 128/256 induce redundant swap events.

### Hardware prerequisites

GPU HBM pressure (swap is triggered by HBM exhaustion, §1/§5.4), an NVMe SSD with a pSLC
provisioning mode, NVIDIA GDS + a GDS-capable GPU, and vLLM's pipeline-aware scheduler providing
the slack (§5.4, Appendix G, Appendix E). The paper is explicit that HiFC's cost win is *DRAM
provisioning economics*: 128 GiB DDR4 vs 1 TiB pSLC SSD, **4.5× cheaper over 3 years**
(§4 Table 1; updated to **6.1×** with H2-2025 prices in Appendix C, Table 12).

### Verdict: **NOT REPRODUCIBLE (hardware)** for the paper's core; **PARTIALLY REPRODUCIBLE (software part only)** for three sub-mechanisms

- **pSLC zones (§4, App. D) → NOT REPRODUCIBLE (hardware).** No NVMe on either machine (§0), no
  pSLC provisioning interface on a SATA Samsung PM-class drive or the ADATA SU630 QLC, and no
  vendor tooling. We cannot claim endurance or the 4.7 GiB/s pSLC numbers.
- **GDS path (§3.2, App. B data path) → NOT REPRODUCIBLE (hardware).** No CUDA toolkit, no
  GDS-capable GPU, engine is CPU-only.
- **Append-only, fixed-size, 4 KB-aligned block layout (§3.2 + App. B Eqs. 2–7) → REPRODUCIBLE.**
  This is arithmetic + file layout. We need it if (and only if) we ever put KV or weight shards on
  SSD.
- **Pipeline-slack visibility criterion (App. G Eqs. 12–15) → REPRODUCIBLE.** A portable cost
  model: it tells the scheduler *when* SSD I/O is free. Directly reusable by C6.

Also portable in spirit (**§5.4/App. F**): swap only when capacity is exhausted, and let
concurrency absorb the latency. Their limitations (**§6**): short-context / latency-sensitive
workloads suffer; shared-flash contention needs tuning; SSD tuning "requires domain-specific
expertise".

### Falsifiable experiments (ours, proposed)

> **X5 — sequential vs fragmented layout on papi.** `scripts/bench/tier_rw.rs` writes the same
> 8 GB payload twice: (a) append-sequential, HiFC §3.2 style, (b) interleaved per-block as a naive
> paged layout would, then reads both back with 64-token-aligned offsets (App. B).
> **Metric:** MB/s and syscalls.
> **Pass:** sequential ≥ 1.5× fragmented read throughput (and ≥ 1.2× on P500's Samsung SATA).
> **Fail:** the layout work is **NOT PROVEN** on our hardware — we keep whatever layout the
> allocator already has and record NOT PROVEN.

> **X6 — is GDS-shaped latency hiding even needed? (App. G port).** From X2's traces compute
> `T_gap` (pipeline deficit) and `T_swap` (storage transfer time) per Eq. 13–15.
> **Pass (mechanism useful):** in ≥80% of decode steps, `T_swap > T_gap`, i.e. I/O *is* visible and
> scheduling must hide it.
> **Fail:** I/O is already hidden — then we spend no effort on a swap scheduler (a genuine "don't
> build").

### Their headline numbers (theirs, not ours)

"Throughput comparable to DRAM-based swapping" — within **1–2%** across models/datasets (§5.3
Table 3); **4.5×** lower 3-year memory-expansion cost (§4, Table 1; 6.1× updated, App. C);
**WAF 1.02** vs 1.4 spec (§5.5); **8× TBW** (App. D); **2.81×** faster session init at 100 GiB KV
(§5.8 Table 6); under heavy load, **182 vs 172 tok/s and 5.4 s vs 5.8 s latency vs DRAM swapping**
(App. F Table 19). **All on 2× A100 80 GiB + NVMe pSLC + GDS.**

---

## 3. INF2 — arXiv 2502.09921, near-storage processing

**Citation:** Hongsun Jang, Siung Noh, Changmin Shin, Jaewon Jung, Jaeyong Song, Youngsok Kim,
Jinho Lee (Seoul National University / Yonsei), *"INF2: High-Throughput Generative Inference of
Large Language Models using Near-Storage Processing"*, arXiv:2502.09921.

### Mechanism (sections cited)

INF2 attacks KV-cache I/O in *batched offload* inference. (a) **Attention-near storage (§4.1):**
self-attention runs on an FPGA inside each computational storage device (SmartSSD), so KV read
traffic uses the CSD's private internal PCIe path and only `b·h·d·2` bytes of result cross the
system interconnect instead of `s·2·b·h·d·2` — a traffic ratio of `(s+1)/2` (Eq. 3). Parallelism
is split across batch × attention heads (§4.1). (b) **Delayed KV writeback (§4.2):** instead of
writing each new K/V entry to SSD (which sits on the critical path and is smaller than a page),
host-memory buffers feed the accelerator directly and **spill in bulk after `c` decode
iterations**; `c` must be > 1 because new entries are 256 B while direct I/O's minimum is 512 B
(§6), default `c = 2` (§7.1). (c) **Cooperative X-cache (§4.3):** store the *input activation X*
rather than K and V; since `K = X·W_K`, `V = X·W_V` (Eq. 1), X is **half** the size of the KV
cache, so at a fixed host-memory budget you halve interconnect traffic again, paying an extra
QKV-projection recomputation on the GPU which they report as "almost negligible" (§4.3). The
scheduler (§6) estimates footprints, splits DRAM between CSD residency and X-cache, and runs the
delayed-writeback spill "independently from the LLM inference pipeline".

Motivation numbers (**theirs**, §3): **>80%** of total inference time is KV-cache I/O when storage
is used; RAID0 stops scaling past **3 SSDs** because the shared PCIe interconnect saturates
(Fig. 3c).

### Hardware prerequisites

16× Samsung SmartSSD (4 TB NVMe + Kintex UltraScale+ KU15P FPGA each), PCIe expansion chassis
(H3 Falcon 4109), A100/H100, Xeon Gold 6342 2×48C, 1 TB DDR4 (default 512 GB budget), Vitis/XRT
2023.1, CUDA 12.1, OpenCL 2.2 (§7.1 Table 1, §7.2). The accelerator itself occupies 47.36% LUT /
47.10% BRAM, 14.21 W per device (§7.2, Table 2). Baselines: FlexGen and DeepSpeed ZeRO-Inference
with CPU attention, batch 32, output 64, OPT/LLaMA-2 FP16 (§7.1).

### Verdict

- **Attention-near storage (§4.1) + FPGA accelerator (§5) → NOT REPRODUCIBLE (hardware).** No CSD,
  no FPGA, no PCIe expansion, no second device. HiFC Appendix A.1 makes the same point about this
  family of systems: CSD ≈ $2,526 used vs $136 for a commodity NVMe (their Table 7).
- **Delayed KV writeback (§4.2) → REPRODUCIBLE (software part only; this is the portable win).**
  Host buffer + spill threshold + chunked writes is ~100 lines against any file descriptor. It
  matters on our machines **only if** KV actually reaches SSD (long context on papi) — otherwise
  there is nothing to hide.
- **Cooperative X-cache (§4.3) → NOT REPRODUCIBLE *as a win* (wrong hardware), even though the
  code is portable.** X-cache pays off when (i) KV crosses a constrained link (PCIe) and (ii)
  there is spare main memory and GPU compute to regenerate KV (§4.3, §7.5: gains grow with memory
  budget 256 GB→1 TB, and with batch size — at batch 1 "the KV cache can fit into main memory,
  resulting in no storage traffic", §7.5). On P500 KV lives in the same DRAM the CPU already reads;
  on papi our bottleneck is *weights*, and KV at our context lengths is a fraction of the 17 GB
  weight stream. Recomputing QKV per token on a Xeon X5670 is a net loss.

### Falsifiable experiments (ours, proposed)

> **X7 — delayed writeback (C4 software part).** `scripts/bench/kv_writeback.rs` generates KV
> entries at decode rate and writes them (a) per token, (b) with `c ∈ {8, 32, 64}` (INF2 §4.2),
> using 512 B-aligned `pwrite` batches (§6 granularity rule).
> **Metrics:** decode stall milliseconds attributed to writes; achieved write MB/s; bytes written.
> **Pass:** stall time at `c = 64` ≤ 50% of `c = 1`, with write bandwidth ≥ 4× the per-token case.
> **Fail:** per-token write cost is already < 2% of decode time on this hardware → mechanism
> **NOT NEEDED** (record as NOT PROVEN / not built, with the measured percentage).

> **X8 — X-cache byte ledger (C5).** From GGUF metadata compute, per 1k tokens:
> `bytes(X) = Σ_layers hidden·dtype`, `bytes(KV) = 2·Σ_layers n_kv_heads·head_dim·dtype`, plus the
> extra FLOPs `2·(W_K + W_V)` per token. Then profile a real decode and measure what fraction of
> decode wall-time is KV reads from SSD.
> **Pass (build C5):** KV-from-storage reads ≥ 10% of decode time on papi **and** `bytes(X) ≤
> 0.6·bytes(KV)` for the model in question.
> **Fail (expected):** below that → **do not build**, ledger committed as the evidence.

### Their headline numbers (theirs, not ours)

Up to **3.46×** throughput over FlexGen (abstract, §7.5); **>2×** consistently with 16 CSDs
(§7.3: 2.67× @30B, 2.3× @175B); ablation over ANS alone: X-cache **1.07–1.19×**, delayed writeback
**1.10–1.12×**, both together **1.1–1.57×** (§7.4); >80% of time in KV I/O for the baseline
(§3). **On 16 SmartSSDs + A100/H100.**

---

## 4. Tutti — arXiv 2605.03375, SSD-backed KV cache

**Citation:** Shi Qiu, Yifan Hu, Xintao Wang, Wenhao Zhu, Jianqin Yan, Hao Chen, Kaiqiang Xu, Kai
Chen, Yiming Zhang (Xiamen University / SJTU / HKUST), *"Tutti: Making SSD-Backed KV Cache
Practical for Long-Context LLM Serving"*, arXiv:2605.03375.

### Mechanism (sections cited)

Tutti's target is not SSD bandwidth but the *control path*. Its diagnosis (§2.2): a 128 K-token
KV reload for Qwen3-32B at block 64 means ~**256 K scattered 80 KB objects**; with LMCache's
256-token chunks it is still >1,000 mostly-random chunk accesses; SSD tiers cause **70–80% GPU
bubble**, and even GDS stays >70% because "GDS … still relies on CPU intervention to initiate each
I/O" (§2.2). Three designs follow:

1. **GPU-native object abstraction (§3.1):** a GPU file is `2×L` objects (one key + one value per
   layer), aligned with the engine's KV block manager; a **Tensor-Stripe layout** follows tensor
   granularity rather than fine striping; objects are round-robin across SSDs; the pool of NVMe
   files is **pre-allocated at startup** so no metadata op runs at runtime; a precomputed **P2P
   mapping table** converts KV virtual addresses to PCI addresses, using **SGL (16 B per chunk)
   instead of PRP** — 60 GB KV on 80 GB HBM would cost 3.75 GB of PRP pages vs **15 MB** of SGL
   (§3.1). Net: CPU overhead drops from `O(layer × blocks)` to `O(layer)`.
2. **GPU io_uring / `gio_uring` (§3.2):** CPU prepares I/O control blocks ahead of time into
   lock-free SQ/CQ rings in HBM (non-cached mmap); each SQ entry holds 2048 I/O contexts; CUDA
   events order I/O against compute; **NVIDIA green contexts** partition SMs into a Compute
   Domain and an I/O Control Domain so a long I/O kernel cannot monopolise the GPU (§3.2).
3. **Slack-aware I/O scheduling (§3.3):** an offline profile table indexed by
   `(L_input, L_prefix)` records per-layer slack windows (duration + SM budget); reads are
   scheduled into slack with priority, writes are deferred and only issued in slack (flushed
   best-effort in decode). The motivation is measured and portable: **concurrent read+write drops
   total bandwidth by 60%**, reproduced with FIO using one read and one write thread at 256 MB
   granularity (§3.3, Fig. 6) — large-block R/W contend for the NVMe internal cache.

Implementation (§3.4): ~8,000 LoC C++ + ~1,500 LoC Python on vLLM's KVConnector; multi-GPU via a
local daemon owning one NVMe queue pair per GPU; Mooncake as the cluster control plane.

### Hardware prerequisites

64-core Xeon 6530, 512 GB DRAM (256 GB pinned), **2× H100 80 GB**, **4× Solidigm D7-PS1010
7.68 TB NVMe**, 14 TB SSD per GPU, 50 GB/s DRAM–HBM, SSDs peaking 29 GB/s read / 12 GB/s write
(§2.2, §4 Environments). GeminiFS (FAST'25) as the companion GPU file system (§3.1), NVIDIA green
contexts (§3.2), SGL-capable NVMe command path (§3.1, §4.2.2).

### Verdict

- **`gio_uring`, GeminiFS, P2P/SGL mapping, SM partitioning (§3.1–§3.2) → NOT REPRODUCIBLE
  (hardware).** Needs an H100-class GPU, NVMe queues, and driver-level privileges. Their own
  PRP→SGL microbenchmark (§4.2.2: 0.287 → 8.891 GB/s read, 31×) is a GPU-NVMe result, useless here.
- **Object abstraction + pre-allocated pool + `O(layer)` batching (§3.1) → REPRODUCIBLE.** It is a
  file-format and call-granularity decision: one `pread` per layer instead of per block. Directly
  applicable to weight streaming (C1) and any KV tier (C4).
- **Slack-aware scheduling (§3.3) → PARTIALLY REPRODUCIBLE (software part only).** The *idea*
  (offline profile → schedule I/O into measured idle windows; never overlap reads with writes)
  has no GPU-specific dependency. The *lookup table keys* (`L_input`, `L_prefix`, SM budget) must
  be re-derived for CPU decode (our slack = time between layer k and layer k+2 of weight loading).
- **Decoupled read/write (§3.3) → REPRODUCIBLE and testable on both machines with FIO** — this is
  the single most portable empirical claim in the five papers.

### Falsifiable experiments (ours, proposed)

> **X6b — read/write bandwidth collapse (the 60% claim, ours to re-measure).**
> `scripts/bench/ssd_rw_contention.sh` (FIO, 256 MB blocks, direct=1): (a) 1 read job, (b) 1 write
> job, (c) 1 read + 1 write concurrently, on papi's ADATA SU630 and P500's Samsung, 60 s each.
> **Metric:** aggregate MB/s of (c) ÷ (a)+(b).
> **Pass (build decoupled scheduling):** ratio ≤ 0.70 (collapse reproduced on our hardware).
> **Fail:** ratio ≥ 0.90 → contention is not a problem here; **do not build** the read/write
> scheduler; report Tutti's 60% as theirs and our ratio as ours.

> **X5b — object granularity (`O(layer)` vs `O(layer×blocks)`).** Read a 128 K-token-equivalent
> payload from SSD via (a) per-block 80 KB reads and (b) one contiguous per-layer read, `O_DIRECT`.
> **Metrics:** MB/s, syscall count.
> **Pass:** batching ≥ 1.5× throughput or ≥ 50% fewer syscalls with equal throughput.
> **Fail:** granularity doesn't matter on our block layer → NOT PROVEN, keep the simple path.

### Their headline numbers (theirs, not ours)

**TTFT −78.3%** vs GDS-enabled LMCache under strict SLO, **2×** achievable request rate, serving
cost **−27%** (abstract, §4.1, §4.3); retrieval bandwidth **25.9 GB/s** vs GDS' 11.9 GB/s
(**2.08×**, §4.2.1); PRP→SGL **31.0× read / 91.3× write** (§4.2.2); compute/I/O crossover pushed to
**98.3%** cache hit rate with bubble averaging 25 ms (§4.2.5); LMCache-SSD bubbles >70–80% of
latency (§2.2). **On H100 + enterprise NVMe.**

---

## 5. CALVO — arXiv 2603.21257, network-intensive inference

**Citation:** Weiye Wang, Chen Chen, Junxue Zhang, Zhusheng Wang, Hui Yuan, Zixuan Guan, Xiaolong
Zheng, Qizhen Weng, Yin Chen, Minyi Guo (SJTU / USTC / Huawei / TeleAI), *"CALVO: Improve Serving
Efficiency for LLM Inferences with Intense Network Demands"*, arXiv:2603.21257.

### Mechanism (sections cited)

CALVO is about **admission and scheduling**, not bytes. Two mechanisms:

1. **Dispatcher–executor decoupling (§3.1):** every stage (L3 remote → L2 local DRAM → L1 HBM →
   compute) gets its own **dispatcher–executor pair** that runs autonomously instead of being
   poked by vLLM's single compute thread (§2.3.1 identifies the centralised control as the
   resource-idle cause). The load-bearing trick: **lower-level dispatchers proactively trigger
   space allocation at the higher level** — when L3→L2 issues a transfer it simultaneously
   submits an L1 allocation request, so loading overlaps as soon as data dependencies resolve;
   when a block lands, the stage signals upward for fine-grained overlap (§3.1). Compute launches
   as soon as *its* blocks are resident rather than waiting for "loading finished" as a phase
   (§3.1).
2. **Load-inclusive cost model (§3.2):** offline profiling fits two linear functions —
   `T_load(context tokens)` (verified linear, Fig. 6) and `T_comp(query tokens)` — joined into a
   binary linear service cost. Scheduling then is textbook but only correct *with both terms*:
   **SJF on `T_load + T_comp`** minimises mean TTFT (Cheng 1985), **LSTF** with
   `LST = DDL − T_load − T_comp` maximises SLO attainment (Liu & Layland 1973). Their §2.3.2
   example: R1 (load 0.361 s, comp 0.019 s) vs R2 (load 0.199 s, comp 0.025 s) — FIFO or
   compute-only SJF picks R1 → mean TTFT **0.49 s**; load-aware SJF picks R2 → **0.41 s**. And
   §4.3: scheduling by *prefill token count only* (SJF-PT) can be **worse than FIFO**.

Context (**theirs**, §1, §2.2): with 400 Gbps RDMA, KVCache loading can exceed **90% of TTFT**;
the workload pattern is "long context, short query" (Table 1: LooGLE 28.1 K context vs 28-query
tokens).

### Hardware prerequisites

One GPU node (80 GB GPU, 128 GB DRAM) + a remote 512 GB CPU DRAM node over **RDMA 400 Gbps**
(§4.1); vLLM 0.9.1 + LMCache 0.3.1 + Mooncake Store as L3, ZeroMQ IPC, ~3.3 K LoC, an override of
vLLM's `add_request()` to bypass the FIFO queue (§4.1). Workloads: Llama-3.1-8B-Instruct and
Qwen2.5-14B-Instruct-1M on LooGLE / ICL / Code, Poisson arrivals (§4.1).

### Verdict: **REPRODUCIBLE (software part only)** — the most portable paper of the five

The hardware is a GPU cluster with 400 Gbps RDMA; we have neither. But **neither mechanism needs
that hardware**: dispatcher/executor autonomy is threads + channels, and the cost model is two
linear fits. What changes for us is the *tier names*: our "L3→L2" is **SSD→DRAM**, our "L2→L1" is
**DRAM→compute-resident window**, and the network disappears. The insight survives intact —
*price the load, and never let the compute thread own the loader* — and it is exactly C6.

Caveat to state honestly: CALVO's gains appear when **loading dominates service time** (their
>90%-of-TTFT regime, §1). On P500 weights are DRAM-resident, so loading ≈ 0 → C6 has nothing to
price. C6 is a **papi-first** feature and, later, a concurrency feature on both machines.

### Falsifiable experiments (ours, proposed)

> **X8b — linearity of load cost (CALVO Fig. 6 port).** `scripts/bench/load_fit.rs` times SSD
> reads of `n ∈ {1k…32k}` token-equivalents of model/KV data on both machines, 20 repetitions.
> **Metric:** least-squares fit of `T_load = a·n + c`; **R²**.
> **Pass:** R² ≥ 0.90 on both machines (the model the scheduler will rely on).
> **Fail:** cost model is non-linear → C6 must use measured tables, not a linear fit; the CALVO
> recipe as-is is **NOT PROVEN** here.

> **X9 — load-aware scheduling (C6).** `scripts/bench/scheduler_sim.rs` (trace-driven, then
> real): 200 synthetic requests, Poisson arrivals, contexts 1k–32k, engineered so
> `T_load ≥ 50%` of service time (papi, SSD tier). Compare (a) FIFO, (b) SJF on `T_comp` only,
> (c) SJF on `T_load + T_comp`, (d) LSTF with deadlines at 4× isolated TTFT.
> **Metrics:** mean TTFT, SLO attainment %.
> **Pass:** (c) mean TTFT ≥ 10% lower than (b), or (d) SLO attainment ≥ +10 pp vs (b).
> **Fail:** **NOT PROVEN** — keep FIFO and say so.

### Their headline numbers (theirs, not ours)

Up to **+61.67% SLO attainment** vs vLLM-LMCache (abstract, §4.2); **−81.3% mean TTFT** on ICL at
QPS 1.2 (§4.2); LSTF 73% vs EDF 58% SLO attainment (§4.3); load-aware SJF 0.41 s vs 0.49 s on the
two-request example (§2.3.2); KVCache loading **>90% of TTFT** at 400 Gbps (§1). **On GPU +
400 Gbps RDMA + distributed DRAM pool.**

---

## 6. What the engine must do (C1–C6, traced)

| # | requirement for `fitsproof-rs` | traced to | notes |
|---|---|---|---|
| **E1** | **Tiered weight residency with a prefetch invariant.** Weight tensors addressed as (SSD offset → DRAM window → compute buffer). A background loader reads *layer k+W* while layer *k* computes; W ≥ 2 (M2Cache §5.4 derives ≥2 from SSD-read ≈ 2× layer-inference time). Decode must never issue a blocking read on the critical path. | M2Cache §5.4 (preloader, fixed+dynamic two-level DRAM cache); Tutti §3.1 (`O(layer)` batching, pre-allocated pool) | This is the papi enabler; without it C1's row is "model does not run". |
| **E2** | **Measure the preload, not just build it:** hit rate, stall fraction, achieved MB/s as first-class telemetry (spec C1 evidence column). | M2Cache §5.4; HiFC App. G (visibility criterion) | Feeds X2/X6. |
| **E3** | **Static mixed-precision policy per tensor class** (attention int8, FFN int4) with an offline ratio search over a calibration corpus; a quality check that can fail. *No* per-token neuron predictor. | M2Cache §5.2 + Algorithm 1 (port the sweep, drop the Deja-Vu predictor); §6.5 quality table as the template for our check | Spec C2. |
| **E4** | **Quantised KV cache (int8) with byte accounting in the tracking allocator**, plus a KV bytes/token metric before and after. | spec C3 (KIVI); HiFC §2.2 / Tutti §2.1 for why KV dominates at long context | Needed to hold peak RSS < 13 GB on papi at long context. |
| **E5** | **If (and only if) KV or weight shards reach SSD:** append-only, fixed token-block (start at 64 tokens, HiFC §5.6), 4 KB-aligned, precomputed per-layer offsets (HiFC App. B Eqs. 2–7), delayed bulk writeback with spill threshold c ≥ 1 (INF2 §4.2, §6), reads and writes never concurrent (Tutti §3.3), layer-granular `pread` (Tutti §3.1). | HiFC §3.2/App. B; INF2 §4.2; Tutti §3.1, §3.3 | Spec C4 — gated on X5/X6b/X7 outcomes. |
| **E6** | **Loader as a first-class stage:** independent thread with its own queue, proactive destination allocation before the data arrives, no reliance on the compute loop to kick I/O. | CALVO §3.1 | Prerequisite for C6 and for E1's non-blocking invariant. |
| **E7** | **Load-inclusive scheduling cost:** `cost = a·load_tokens + b·compute_tokens`, fitted per machine; SJF/LSTF built on it; FIFO retained as the honest baseline. | CALVO §3.2 (Fig. 6 linearity), §2.3.2 | Spec C6. |
| **E8** | **Honesty gates:** llama.cpp/ollama measured on the same machine and model as a second column; every number from `reports/*.json`; anything below threshold written as **NOT PROVEN**, never as a number we did not measure. | spec *Non-negotiables* | M2Cache §6.5 is the cautionary tale: a 25% relative HumanEval drop described as "negligible". |

---

## 7. Ranked implementation plan

Effort: **S** ≤ 0.5 day · **M** 1–2 days · **L** 3–5 days (one developer, this engine).

| rank | build | why first | effort | depends on |
|---|---|---|---|---|
| **0** | **Baseline harness + honesty column** — `scripts/bench/tier_bw.rs` (seq/random read+write MB/s per tier), `scripts/bench/llama_cpp_baseline.sh` (same model, same machine, tok/s + peak RSS), JSON under `reports/`. | Everything else is a *ratio* against this. Spec forbids shipping a number without it. Also calibrates every threshold below. | **S** | — |
| **1** | **C1/E1+E2: SSD→DRAM pattern-aware preloader** (double buffer, W≥2 lookahead, fixed area = first n layers + dynamic area, layer-granular `pread`, telemetry). | papi cannot run the model at all without it: 17 GB of weights vs 13 GB budget. Highest certainty, lowest risk, directly traceable to M2Cache §5.4. | **M** | 0 |
| **2** | **C3/E4: int8 KV + allocator byte accounting.** | Second binding constraint on papi (KV grows with context while weights stream); cheapest memory win per line of code; quality gate is a check that can fail. | **S–M** | 0 |
| **3** | **C2/E3: static mixed-precision policy + offline ratio search + quality gate.** | Buys back memory without losing quality *if* X3 passes; no predictor, no GPU. Runs after E1/E3 so the budget it optimises is real. | **M** | 1, 2 |
| **4** | **C6/E6+E7: loader stage + load-inclusive cost model + scheduler.** | Only pays off when loading dominates (papi) or when requests overlap. Do it after the loaders exist — the cost model needs E1's telemetry to fit against. | **M–L** | 1 (X8b, X9) |
| **5** | **C4/E5: SSD KV tier** — append-only 4 KB-aligned layout, HiFC offsets, delayed writeback, decoupled R/W. | Only justified if a long-context papi run actually spills KV to disk *and* X5/X6b/X7 pass. Ship behind a feature flag; the layout code is reusable for weight shards either way. | **M** | 0, 2 (X5, X6b, X7) |
| **6** | **C5: X-cache** | Expected negative on this hardware (§3). Build **only** if X8's ledger passes (KV-from-storage ≥ 10% of decode time). | **M** (gated) | X8 |

Sequencing rationale in one line: *make the model run on papi (1), make it fit (2–3), make it
schedule (4), only then add a disk tier for KV (5), and never add a mechanism whose own ledger
says it cannot pay (6).*

---

## 8. Falsifiable experiment register

| id | claim under test | machine | metric | pass threshold | on fail |
|---|---|---|---|---|---|
| X0 | baseline exists | P500 + papi | llama.cpp tok/s, peak RSS, tier MB/s | recorded (no threshold) | block all other claims |
| X2 | preload lifts streamed decode (M2Cache §5.4) | papi | tok/s, stall %, hit rate | ≥1.15× vs W=0, stall <5%, hit ≥95% | **NOT PROVEN** → ship W=0 |
| X3 | mixed precision beats uniform int4 at fixed budget (M2Cache §5.2) | papi | quality @ budget B, bytes | quality ≥ uniform-int4, bytes ≤ 0.85× uniform-int8 | **NOT PROVEN** → keep GGUF types |
| X4 | int8 KV halves bytes/token with acceptable quality | P500 + papi | KV bytes/1k tok, top-5 agreement | bytes ≤ 0.55× fp16; ≥18/20 prompts agree | revert to fp16 KV, **NOT PROVEN** |
| X5 | sequential layout beats fragmented (HiFC §3.2) | P500 + papi | read MB/s | ≥1.5× (≥1.2× P500) | keep current layout |
| X6 | swap I/O is visible, not hidden (HiFC App. G) | papi | `T_swap > T_gap` fraction | ≥80% of steps visible | don't build swap scheduling |
| X6b | concurrent R/W collapses bandwidth (Tutti §3.3) | P500 + papi | (a+b)/c MB/s ratio | ≤0.70 → build decoupler | ≥0.90 → **do not build** |
| X5b | layer-batched reads beat per-block (Tutti §3.1) | papi | MB/s, syscalls | ≥1.5× or ≥50% fewer syscalls | NOT PROVEN |
| X7 | delayed writeback hides write stall (INF2 §4.2) | papi | write stall ms, MB/s | stall ≤50% of c=1, BW ≥4× | **NOT NEEDED** (report %) |
| X8 | X-cache byte ledger justifies C5 (INF2 §4.3) | papi | KV-read share of decode time | ≥10% and `bytes(X) ≤ 0.6·bytes(KV)` | **do not build** |
| X8b | load cost is linear in tokens (CALVO Fig. 6) | P500 + papi | R² of `a·n+c` | R² ≥ 0.90 | use measured tables, CALVO recipe NOT PROVEN |
| X9 | load-aware scheduling beats compute-only (CALVO §3.2/§4.3) | papi | mean TTFT, SLO % | ≥10% mean TTFT or ≥10 pp SLO | **NOT PROVEN** → keep FIFO |

Harness paths are proposals for the implementation lane; **this lane writes no code.**

---

## 9. Do not build

The list is as valuable as the plan. Each entry names the paper section we are declining.

1. **GPU Direct Storage / any GPU↔SSD path** (HiFC §3.2; Tutti §3.1–3.2). No CUDA toolkit, no
   GDS-capable GPU, engine is CPU-only (README + spec). Everything downstream of GDS — cufile
   staging, 4 KB GDS buffers, P2P DMA — inherits the exclusion.
2. **pSLC zone provisioning and SSD endurance engineering** (HiFC §4, App. D). We have SATA drives
   (§0), no NVMe pSLC control, no vendor tooling; and our write volume is *model weights written
   once*, not the 1.98 TiB/day HiFC budgets. WAF optimisation cannot pay back.
3. **Attention-near storage / computational storage** (INF2 §4.1, §5; cf. HiFC App. A.1). No CSD,
   no FPGA, no PCIe expansion chassis. HiFC's own Table 7 prices the gap ($2,526 vs $136).
4. **Deja-Vu-style dynamic neuron predictors + HBM neuron cache with ATU** (M2Cache §5.2–5.3).
   Requires trained per-layer predictors and a GPU offload path; §5.5.2 concedes the predictor
   fails at large batch anyway. Our mixed precision (E3) is deliberately the *static* subset.
5. **GPU io_uring, NVIDIA green contexts, GeminiFS, SGL/P2P mapping** (Tutti §3.1–3.2). H100-class
   GPU + NVMe queues + privileged drivers; ~8,000 LoC of GPU storage stack (§3.4) for hardware we
   do not own.
6. **Distributed L3 KV pool over the LAN** (CALVO §1, §2.2, §4.1; Mooncake). A 1 GbE hop to papi
   is ≈ 110 MB/s — *slower than either machine's local SATA SSD*. Strictly dominated; would
   violate "budget conscious" and add failure modes for negative gain.
7. **RAID0 multi-SSD scaling** (INF2 §3, Fig. 3c). One SATA port per machine, and INF2 themselves
   show interconnect saturation past 3 SSDs.
8. **Carbon-emission accounting as a product metric** (M2Cache §2.2 Eq. 1, §6.4). Requires an
   energy model we cannot measure; not in C1–C6.
9. **Sliding-window / LLM-in-a-Flash-style DRAM cache** (M2Cache §5.3 discussion). The OS page
   cache already does this for a file we read sequentially; M2Cache replaced it with layer-aware
   FIFO for a reason.
10. **Claiming "SSD swapping matches DRAM"** (HiFC abstract). Their result is A100 + NVMe pSLC +
    GDS + vLLM pipelining. On SATA + CPU the premise is absent; we will report our own ratio or
    NOT PROVEN, never their 1–2%.
11. **Per-token KV flush** (INF2 §4.2's naive baseline) — do not even implement it as a default;
    X7 exists to quantify how bad it is, and the answer feeds E5's threshold `c`.

---

## 10. Summary

**Papers read (5):** M2Cache (arXiv 2410.14740v2, §1–§8) · HiFC (NeurIPS 2025, main + App. A–H)
· INF2 (arXiv 2502.09921, §1–§10) · Tutti (arXiv 2605.03375, §1–§5) · CALVO (arXiv 2603.21257,
§1–§5).

**Verdicts:**

| paper | verdict |
|---|---|
| M2Cache | **PARTIALLY REPRODUCIBLE (software part only)** — C1 preloader + static half of C2 are ours; HBM tier, CUDA streams, Deja-Vu predictor are not |
| HiFC | **NOT REPRODUCIBLE (hardware)** — no NVMe/pSLC/GDS; block layout (App. B) and slack criterion (App. G) are the portable residue |
| INF2 | **NOT REPRODUCIBLE (hardware)** for ANS/CSD; delayed writeback **REPRODUCIBLE**, X-cache portable-but-not-worthwhile here |
| Tutti | **NOT REPRODUCIBLE (hardware)** for `gio_uring`/SGL/SM partitioning; object batching + slack-aware / decoupled R/W scheduling **PARTIALLY REPRODUCIBLE (software part only)** |
| CALVO | **REPRODUCIBLE (software part only)** — dispatcher/executor decoupling and the load-inclusive cost model need no special hardware; gains only where loading dominates (papi) |

**Top 3 mechanisms to build:**
1. **Pattern-aware SSD→DRAM layer preloading with a ≥2-layer lookahead and a two-level DRAM
   window** (M2Cache §5.4; Tutti §3.1 batching) — the only mechanism without which papi cannot
   run the model at all. Experiment X2.
2. **int8 KV cache with allocator byte accounting** (spec C3; HiFC §2.2 / Tutti §2.1 motivate why
   KV binds at long context) — cheapest path to peak RSS < 13 GB on papi. Experiment X4.
3. **Load-inclusive cost model + autonomous loader stage** (CALVO §3.1–§3.2) — turns "we stream
   from disk" from a stall into a schedulable cost, and is the only C6-compliant answer.
   Experiments X8b + X9.

**Do not build:** GDS/GPU paths · pSLC/SSD-endurance engineering · CSD/attention-near-storage ·
Deja-Vu predictors + HBM neuron cache · GPU io_uring/GeminiFS/SGL · LAN-distributed L3 KV pool ·
RAID0 scaling · carbon accounting · sliding-window DRAM cache · any inherited headline claim
("SSD ≈ DRAM").

Everything above is analysis and proposed experiments. No production code was written in this lane.

**PASS_offload-research COMPLETE** — papers: M2Cache / HiFC / INF2 / Tutti / CALVO (all read in
full, sections cited); verdicts: PARTIALLY / NOT (software residue) / NOT (software residue) /
NOT (software residue) / REPRODUCIBLE (software only); top 3 mechanisms: (1) pattern-aware
SSD→DRAM layer preloading, (2) int8 KV with byte accounting, (3) load-inclusive cost model +
autonomous loader; do-not-build: GDS, pSLC engineering, CSD/near-storage, Deja-Vu predictors +
HBM cache, GPU io_uring/GeminiFS, LAN KV pool, RAID0, carbon accounting, sliding-window cache,
inherited headline claims.
