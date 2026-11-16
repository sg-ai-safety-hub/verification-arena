# VerificationArena

*This readme is human written, everything else is AI-generated and unverified*

VerificationArena is a testbed for software-only Compute Verification approaches. It provides a platform to test and improve verification schemes. The arena is currently initialised with an implementation of Memory Exhaustion and Link Exhaustion.

## The Arena: How It Works

We focus on the goal of distinguishing between inference vs training on a GPU.

The Blue Team defines a verification scheme that provides the minimal inference overhead while ensuring the maximal training slowdown. The Red Team tries to complete the training workload while satisfying the verifier with the least slowdown possible.

Specifically:

1. We define baseline inference and training workloads
2. The Blue Team provides a verifier and passing inference implementation
3. The Red Team provides the fastest possible passing training implementation
4. We measure the inference overhead and the training slowdown compared to baselines

We operationalise this as follows:

* Environment: Qwen2.5-32B on a single H200
* Inference workload: vLLM in deterministic mode
* Training workloads:
   * Pretraining
   * RL with LoRA

## Current Initialisation: Memory + Link Exhaustion

### Current Blue Team Implementation

* Memory Exhaustion is as described in our [blog post](https://www.aisafety.sg/blog/exploring-memory-exhaustion-as-a-verification-approach)
   * It covers weights, KV cache and filler used for unused space
   * Verification is done using recomputation, Zero-Knowledge Proofs (ZKPs) are currently unimplemented
* Link Exhaustion is a new idea that exhausts the PCIe link
   * The verifier continuously streams random data through the inbound PCIe link into a small buffer on the GPU
   * This buffer is verified using the same memory challenge mechanism in Memory Exhaustion
   * Link Exhaustion relies on the observation that inference only requires sending tokens through links, hence the vast majority of the bandwidth can be exhausted via random data
* We are currently not using any sort of Compute Exhaustion and hence are not relying on any of its security assumptions

### Current Blue Team Results

* Overhead of 0.9x
* Verifies 99% of HBM (1.9GB unverified, including activations and CUDA runtime)
* Verifies 98% of usable inbound PCIe bandwidth (1GB/s unverified)

### Current Red Team Implementation and Results

* Leverage the unverified HBM and bandwidth
* Pretraining
   * Hold a small shard of weights in unverified HBM, stream activations over the unverified link bandwidth
   * Slowdown: 36x
* RL with LoRA
   * Keep adapters and KV cache in unverified memory using a small batch size
   * Slowdown: 100x
* Note that slowdown numbers exhibit random variation when renting cloud GPUs

## Rules

* Inference workload provided by Blue Team must output results that match baseline inference bit-exactly
* Training workloads provided by Red Team do not need to return any inference outputs, they only need to pass the verifier
* The Red Team is allowed to:
   * Pick the weights of the inference workload when running attacks (e.g. to be all zeros)
   * Utilise the host CPU and memory
   * Utilise sharding (i.e. submit a workload that trains only on a shard of the model weights)
* The Red Team is not allowed to interfere with the verifier (which would be hosted in a separate secure environment in reality)

## Future Work

* Iterating on both the Red and Blue Teams
* Scaling up the environment beyond a single GPU and moving to bare metal servers
* Using a separate machine for the verifier
* Experimenting with alternative environments and workloads
* Simulating adversaries with access to unknown sources of memory/bandwidth
* Implementing verification using ZKPs
* Rigorous security analysis beyond empirical experiments
* Making mini-arenas (e.g. the encoding function, the link saturation mechanism and the challenge function)
