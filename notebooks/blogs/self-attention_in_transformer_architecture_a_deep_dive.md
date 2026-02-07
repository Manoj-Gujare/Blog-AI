# Self-Attention in Transformer Architecture: A Deep Dive

## Introduction: Why Self-Attention Matters

Before transformers revolutionized natural language processing, recurrent neural networks (RNNs) dominated sequence modeling tasks. RNNs process input tokens sequentially—each hidden state depends on the previous one, creating an inherent bottleneck. This sequential nature prevents parallelization during training, making RNNs slow on modern hardware. Worse, as sequences grow longer, gradients must backpropagate through many timesteps, leading to vanishing or exploding gradients that make learning long-range dependencies nearly impossible.

Self-attention fundamentally changes this paradigm. Instead of processing tokens one at a time, self-attention allows every token in a sequence to directly interact with every other token in parallel. When processing the word "bank" in the sentence "The bank can guarantee deposits will eventually cover future tuition costs," self-attention lets the model simultaneously consider relationships with "deposits," "guarantee," and "tuition" to determine whether "bank" refers to a financial institution or a riverbank—without waiting for sequential processing to propagate context.

**The core mechanism is elegant**: for each token, the model computes attention scores representing how much focus to place on every other token when building that token's representation. A word doesn't just look at its immediate neighbors; it can attend strongly to relevant words anywhere in the sequence, regardless of distance. This parallel, all-to-all comparison eliminates the sequential bottleneck while naturally capturing long-range dependencies.

This attention mechanism relies on three learned transformations—**queries, keys, and values**—that work together to determine which tokens are relevant to each other and how to combine their information. In the following sections, we'll unpack exactly how these components enable self-attention's remarkable effectiveness.

## The Query-Key-Value Mechanism

At the heart of self-attention lies an elegant mechanism inspired by information retrieval systems: the query-key-value (QKV) paradigm. Understanding this triplet is essential to grasping how transformers decide which parts of the input to focus on.

> **[IMAGE GENERATION FAILED]** The Query-Key-Value attention mechanism: Input embeddings (X) are projected into three separate spaces (Q, K, V). Queries and keys compute relevance scores via dot product, normalized by softmax, then used to weight and aggregate values.
>
> **Alt:** Query-Key-Value mechanism flow diagram showing how input embeddings are transformed into Q, K, V matrices through linear projections, then combined via dot product, softmax, and weighted aggregation
>
> **Prompt:** Technical diagram showing the Query-Key-Value attention mechanism flow. Left side: input embedding matrix X. Three parallel arrows labeled W_Q, W_K, W_V point to three matrices Q, K, V. Center: Q and K^T matrices multiplying to create attention scores matrix. Right side: softmax normalization arrow, then final multiplication with V matrix producing output. Use clean lines, matrix blocks in different colors (blue for Q, green for K, orange for V), mathematical notation labels, white background, technical illustration style.
>
> **Error:** [Errno 2] No such file or directory: 'images\\images\\qkv_mechanism_flow.png'


**Queries, Keys, and Values: An Intuitive Framework**

Think of the attention mechanism as a soft database lookup:

- **Queries (Q)**: Represent "what information am I looking for?" Each token generates a query vector that encodes what it needs from other tokens in the sequence.
- **Keys (K)**: Represent "what information do I contain?" Each token advertises its content through a key vector that can be matched against queries.
- **Values (V)**: Represent "the actual information to retrieve." Once we determine relevance, we extract the actual content from value vectors.

This separation allows the model to decouple content matching (query-key interaction) from content retrieval (value extraction).

**Linear Transformations: Creating Q, K, V Spaces**

Starting from input embeddings with dimension `d_model`, we project each token into three separate representation spaces using learned weight matrices:

```
Q = X · W_Q    where W_Q has shape (d_model, d_k)
K = X · W_K    where W_K has shape (d_model, d_k)
V = X · W_V    where W_V has shape (d_model, d_v)
```

These linear transformations are crucial—they allow the model to learn task-specific notions of "relevance" and "content." The dimension `d_k` (typically `d_model / num_heads`) controls the expressiveness of our matching space.

**Measuring Relevance: The Dot-Product Similarity**

To determine how much attention one token should pay to another, we compute the dot product between a query and all keys:

```
score_ij = q_i · k_j^T
```

This dot product serves as a similarity measure: larger values indicate higher relevance. Geometrically, it captures both magnitude and directional alignment between query and key vectors. The scaling factor `1/√d_k` is applied to prevent the dot products from growing too large, which would push softmax into regions with vanishing gradients.

**Normalization: Softmax for Probability Distributions**

Raw attention scores need normalization to create a meaningful distribution:

```
attention_weights_i = softmax(scores_i / √d_k)
```

Softmax ensures that all attention weights for a given query sum to 1.0, creating a probability distribution over the sequence. This normalization is differentiable and creates the "soft" in soft attention—instead of selecting a single token, we get a weighted blend.

**Weighted Aggregation: Computing the Output**

The final step combines values according to attention weights:

```
output_i = Σ(attention_weight_ij · v_j)
```

Each output token is a weighted sum of all value vectors, where weights reflect how relevant each position was to the query. This allows information to flow from relevant contexts into each token's representation, enabling the model to build context-aware embeddings.

## Implementing Scaled Dot-Product Attention from Scratch

The scaled dot-product attention mechanism can be expressed as a series of matrix operations that transform queries, keys, and values into contextualized representations. Let's build this step-by-step in NumPy to understand each component.

> **[IMAGE GENERATION FAILED]** Scaled dot-product attention computation pipeline: (1) Compute QK^T similarity scores, (2) Scale by 1/√d_k, (3) Apply optional mask, (4) Softmax normalization, (5) Multiply by values to get output.
>
> **Alt:** Step-by-step visualization of scaled dot-product attention computation showing matrix operations from QK^T through scaling, masking, softmax, to final output
>
> **Prompt:** Technical flowchart diagram showing 5 sequential steps of attention computation. Step 1: Two matrices Q and K^T with multiplication symbol creating scores matrix. Step 2: Division by sqrt(d_k) with mathematical notation. Step 3: Optional mask overlay (shown as triangular pattern for causal mask). Step 4: Softmax function with probability distribution visualization. Step 5: Multiplication with V matrix producing final output matrix. Each step connected by arrows, clean technical style, use color coding for different matrices, white background, annotated with dimension labels.
>
> **Error:** [Errno 2] No such file or directory: 'images\\images\\attention_computation_steps.png'


**The Core Formula**

The attention operation follows this sequence:

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: Query matrix of shape (batch_size, seq_len, d_k)
        K: Key matrix of shape (batch_size, seq_len, d_k)
        V: Value matrix of shape (batch_size, seq_len, d_v)
        mask: Optional mask of shape (batch_size, seq_len, seq_len)
    
    Returns:
        output: Attention output of shape (batch_size, seq_len, d_v)
        attention_weights: Attention scores of shape (batch_size, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    # Step 1: Compute attention scores (QK^T)
    scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch, seq_len, seq_len)
    
    # Step 2: Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Step 4: Apply softmax for numerical stability
    scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-9)
    
    # Step 5: Multiply by values
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights
```

**Why Scale by sqrt(d_k)?**

The scaling factor prevents the dot products from growing too large as dimensionality increases. Without scaling, for high-dimensional vectors, QK^T produces values with variance proportional to d_k. Large magnitudes push the softmax function into regions with extremely small gradients, causing vanishing gradient problems during backpropagation. Dividing by sqrt(d_k) normalizes the variance to approximately 1, keeping the softmax input in a range where gradients remain healthy.

**Masking Strategies**

Two common masking patterns are essential:

```python
def create_padding_mask(seq_len, pad_positions):
    """Mask out padding tokens"""
    mask = np.ones((1, seq_len, seq_len))
    for pos in pad_positions:
        mask[:, :, pos] = 0
        mask[:, pos, :] = 0
    return mask

def create_causal_mask(seq_len):
    """Prevent attending to future positions (autoregressive)"""
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask.reshape(1, seq_len, seq_len)
```

**Verification with a Toy Example**

```python
# Setup: 4 tokens, 8-dimensional embeddings
batch_size, seq_len, d_model = 1, 4, 8
np.random.seed(42)

Q = np.random.randn(batch_size, seq_len, d_model)
K = np.random.randn(batch_size, seq_len, d_model)
V = np.random.randn(batch_size, seq_len, d_model)

# Test without mask
output, weights = scaled_dot_product_attention(Q, K, V)
assert output.shape == (batch_size, seq_len, d_model)
assert weights.shape == (batch_size, seq_len, seq_len)
assert np.allclose(weights.sum(axis=-1), 1.0)  # Verify softmax normalization

# Test with causal mask
causal_mask = create_causal_mask(seq_len)
output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, causal_mask)
assert np.allclose(np.triu(weights_masked[0], k=1), 0)  # Upper triangle should be zero

print("✓ All shape and masking validations passed")
```

The epsilon term (1e-9) in the softmax denominator prevents division by zero in edge cases where all attention scores might be masked out.

## Multi-Head Attention: Parallel Representation Subspaces

Single-head attention has a fundamental limitation: it must choose a single weighted combination of values at each position. But language contains multiple types of relationships simultaneously. A token might need to attend to its syntactic parent, semantic related words, and positional neighbors—all at once. Multi-head attention solves this by running multiple attention operations in parallel, each learning to capture different relationship types.

> **[IMAGE GENERATION FAILED]** Multi-head attention architecture: Input is split into h parallel heads, each with separate Q, K, V projections. Each head independently computes attention, then outputs are concatenated and projected through W_O to produce the final representation.
>
> **Alt:** Multi-head attention architecture diagram showing parallel attention heads with separate Q, K, V projections, independent attention computations, concatenation, and final output projection
>
> **Prompt:** Technical architecture diagram of multi-head attention. Top: single input embedding matrix X. Multiple parallel branches (show 4 heads) each containing: small W_Q, W_K, W_V projection boxes, followed by attention computation block (simplified as scaled dot-product symbol), producing head outputs. Bottom: all head outputs concatenating (shown as stacked matrices merging), followed by final W_O projection matrix producing single output. Use different colors for each head (blue, green, orange, purple), arrows showing data flow, dimension labels (d_model, d_k, etc.), clean technical diagram style, white background.
>
> **Error:** Image generation failed: litellm.APIError: APIError: OpenrouterException - {"error":{"message":"Insufficient credits. This account never purchased credits. Make sure your key is on the correct account or org, and if so, purchase more at https://openrouter.ai/settings/credits","code":402}}


### Motivation: Learning Diverse Relationships

Different attention heads naturally specialize during training. One head might learn syntactic dependencies (verbs attending to their subjects), another captures coreference (pronouns to their antecedents), while a third tracks positional patterns (attending to adjacent tokens). This specialization happens without explicit supervision—the backpropagation signal encourages different heads to capture complementary information.

### Architecture: Splitting and Projecting

Instead of computing attention with full `d_model` dimensions, we split the representation into `h` heads, each with dimension `d_k = d_model / h`. Each head gets its own learned projection matrices `W^Q_i`, `W^K_i`, `W^V_i`:

```python
class MultiHeadAttention:
    def __init__(self, d_model=512, h=8):
        self.h = h
        self.d_k = d_model // h
        
        # Separate projections for each head
        self.W_Q = np.random.randn(h, d_model, self.d_k)
        self.W_K = np.random.randn(h, d_model, self.d_k)
        self.W_V = np.random.randn(h, d_model, self.d_k)
        self.W_O = np.random.randn(h * self.d_k, d_model)
    
    def forward(self, X):
        batch_size, seq_len, d_model = X.shape
        
        # Project and reshape for all heads: (batch, h, seq_len, d_k)
        Q = np.stack([X @ self.W_Q[i] for i in range(self.h)], axis=1)
        K = np.stack([X @ self.W_K[i] for i in range(self.h)], axis=1)
        V = np.stack([X @ self.W_V[i] for i in range(self.h)], axis=1)
        
        # Compute attention per head
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        attn = softmax(scores, axis=-1)
        head_outputs = attn @ V  # (batch, h, seq_len, d_k)
        
        # Concatenate heads and project
        concat = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return concat @ self.W_O
```

### Final Projection: Mixing Head Information

After computing attention independently, all head outputs are concatenated and passed through a final linear projection `W_O`. This projection learns to mix information from different heads, allowing the model to combine syntactic and semantic signals into a unified representation.

### Representational Capacity Comparison

With 8 heads of 64 dimensions each (totaling 512), the model maintains the same parameter count as single-head attention but gains flexibility. Consider the sentence "The bank by the river overflowed." A single head must average attention between "bank" (financial) and "river" (geographical). Multi-head attention allows one head to focus on "river" for disambiguation while another attends to "overflowed" for syntactic structure—capturing both relationships simultaneously.

## Positional Encoding and Self-Attention

Self-attention has a fundamental limitation: it's completely **permutation-invariant**. If you shuffle the order of tokens in a sequence, the attention mechanism produces identical outputs—just in shuffled order. This happens because the attention computation treats the input as an unordered set. When calculating attention scores, each query attends to all keys based purely on content similarity, with no awareness of whether a token appears at position 0 or position 500. For language understanding, this is catastrophic. The meaning of "The cat sat on the mat" differs entirely from "The mat sat on the cat."

To inject positional information, we add **positional encodings** to the input embeddings before they enter the attention layers. The two dominant approaches are:

**Absolute positional encodings** use fixed patterns to represent positions. The original Transformer paper introduced sinusoidal encodings with alternating sine and cosine functions at different frequencies:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

These have useful properties: the encoding for position `pos + k` can be represented as a linear function of the encoding at `pos`, potentially helping the model learn relative positions. However, they're fixed and can't adapt during training.

**Learned positional embeddings** are trainable parameters, one vector per position up to a maximum sequence length. During training, the model optimizes these vectors alongside other weights. This approach is simpler and often performs comparably to sinusoidal encodings, though it can't generalize to sequences longer than those seen during training.

**Relative positional encodings** represent a more sophisticated approach. Instead of encoding absolute positions, they modify the attention computation itself to incorporate the relative distance between tokens. For example, T5 uses learned relative position biases added directly to attention logits. This provides two advantages: the model learns what relative distances matter (adjacent tokens vs. distant context), and it naturally handles variable-length sequences without position limits.

Without positional information, attention patterns become purely content-driven, often collapsing into trivial distributions. With positions encoded, the model can learn structured patterns—attending to previous tokens for autoregressive prediction, or focusing on nearby context for local dependencies—while still leveraging the content-based flexibility that makes self-attention powerful.

## Computational Complexity and Memory Bottlenecks

### Time Complexity Analysis

Self-attention's computational cost is **O(n²d)** where n is the sequence length and d is the model dimension. Let's derive this:

For each attention head, we compute three operations:

1. **QK^T multiplication**: (n × d) @ (d × n) → (n × n) matrix, costing O(n²d)
2. **Softmax**: Applied to the (n × n) attention matrix, costing O(n²)
3. **Attention-weighted values**: (n × n) @ (n × d) → (n × d), costing O(n²d)

The dominant terms are steps 1 and 3, each O(n²d). Since we perform these operations for h attention heads, the total complexity is **O(n²dh)**. However, since hd_head = d (where d_head is the dimension per head), we can express this as O(n²d), which captures the quadratic scaling with sequence length.

### Memory Requirements

The primary memory bottleneck is storing attention weight matrices. For a single layer:

**Memory = batch_size × num_heads × seq_len × seq_len × 4 bytes** (assuming float32)

Let's calculate concrete examples:

| Sequence Length | Batch Size | Heads | Memory per Layer |
|----------------|------------|-------|------------------|
| 512 | 8 | 12 | 8 × 12 × 512 × 512 × 4 = ~100 MB |
| 2048 | 8 | 12 | 8 × 12 × 2048 × 2048 × 4 = ~1.6 GB |
| 8192 | 8 | 12 | 8 × 12 × 8192 × 8192 × 4 = ~25.8 GB |

Notice the **quadratic explosion**: doubling sequence length quadruples memory. At 8192 tokens, a single layer's attention matrices consume more memory than a 16GB GPU can hold.

### Why Long Sequences Are Prohibitively Expensive

The O(n²) memory scaling creates a hard wall for long-context processing. A 12-layer transformer at sequence length 8192 would theoretically need **~310 GB** just for attention matrices—ignoring activations, gradients, and model parameters. This explains why models like GPT-3 originally capped at 2048 tokens and why efficient attention variants (Linformer, Performer, FlashAttention) became necessary.

### Practical Batching Guidelines

For **16GB GPU memory** (e.g., V100, RTX 4080):
- Sequence length 512: batch size 16–32 is safe
- Sequence length 2048: batch size 4–8 maximum
- Sequence length 4096+: batch size 1–2, or requires gradient checkpointing

For **40GB GPU memory** (e.g., A100):
- Sequence length 2048: batch size 16–32
- Sequence length 4096: batch size 4–8
- Sequence length 8192: batch size 1–2 with optimization tricks

Always account for activations (2–3× attention memory) and gradients during training (another 2× multiplier). A conservative rule: **keep total memory under 80% of GPU capacity** to avoid OOM errors from memory fragmentation.

## Common Failure Modes and Debugging Strategies

Self-attention implementations fail in predictable ways. Understanding these patterns helps you diagnose issues quickly and build robust systems.

**Shape Mismatches in Multi-Head Attention**

The most common bug occurs during head reshaping. When splitting `(batch_size, seq_len, d_model)` into multiple heads, you need `(batch_size, num_heads, seq_len, d_k)`. A typical mistake:

```python
# Wrong: flattens batch and heads together
x = x.view(batch_size * num_heads, seq_len, d_k)

# Correct: preserves batch dimension
x = x.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
```

Always verify intermediate tensor shapes after each reshape operation. The batch dimension should remain first, and `seq_len` should stay accessible for the attention computation.

**Attention Collapse**

When all attention weights converge to uniform distributions (each token attends equally to all positions), your model learns nothing about relationships. This manifests as attention matrices that look like constant grids after softmax.

Check two culprits: First, examine your weight initialization. Xavier or He initialization for projection matrices prevents extreme pre-softmax logits. Second, verify your softmax temperature. The standard `1/sqrt(d_k)` scaling prevents saturation—without it, large dot products push softmax into regions where gradients vanish.

Visualize attention weights during training. Healthy attention patterns show structure: nearby tokens receiving higher weights in lower layers, specific token relationships in upper layers.

**Masking Leaks in Causal Attention**

Causal models must never attend to future positions. A subtle bug: applying the mask after softmax instead of before. The correct pattern sets future positions to `-inf` before softmax, ensuring zero attention weights:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
attention_weights = F.softmax(scores, dim=-1)
```

Test by feeding a sequence where token predictions depend on future context—your model should fail unless the mask works correctly.

**Gradient Flow Validation**

Plot gradient magnitudes across layers during training. Attention layers should show consistent gradient scales. Exploding gradients indicate missing normalization; vanishing gradients suggest poor initialization or too many stacked layers without residual connections.

**Edge Case Testing**

Always test: single-token sequences (attention should be identity), all-padding batches (masked positions should contribute zero), and sequences at your maximum length (memory usage and numerical stability). These edge cases reveal boundary condition bugs that only surface in production.

## Self-Attention Variants and Optimization Techniques

While standard self-attention provides powerful modeling capabilities, its O(n²d) time complexity and O(n²) memory requirements become prohibitive for long sequences. Several attention variants address these limitations through different trade-offs.

**Sparse Attention Patterns**

Sparse attention restricts which positions can attend to each other, dramatically reducing computational costs. Local window attention limits each token to attending only to neighbors within a fixed window size w, reducing complexity to O(nwd). Strided attention creates a pattern where tokens attend to every k-th position, useful for capturing regular patterns. Dilated attention combines multiple stride patterns at different rates, similar to dilated convolutions. These approaches typically achieve O(n√n) or O(n log n) complexity by carefully structuring the attention mask. The Longformer and BigBird architectures combine local windows with global attention on special tokens and random attention for a balance between efficiency and modeling power.

**Linear Attention Approximations**

Linear attention methods reformulate the attention mechanism to avoid explicitly computing the full n×n attention matrix. By using kernel methods, these approaches approximate softmax attention as:

Attention(Q, K, V) ≈ φ(Q)(φ(K)ᵀV)

where φ is a feature map. This reordering allows computing φ(K)ᵀV first (complexity O(nd²)), then multiplying by φ(Q), achieving overall O(nd²) complexity regardless of sequence length. The Performer and Linear Transformer use random feature approximations of the softmax kernel, trading some accuracy for guaranteed linear scaling.

**Flash Attention and Memory Optimization**

Flash Attention optimizes standard attention without approximation by restructuring computations to maximize GPU memory hierarchy efficiency. It tiles the attention computation into blocks that fit in fast SRAM, avoiding repeated reads from slower HBM memory. This achieves 2-4x speedups and enables longer sequences by reducing memory usage from O(n²) to O(n), particularly valuable for training on modern GPUs with limited memory bandwidth.

**Cross-Attention vs. Self-Attention**

Cross-attention differs from self-attention by computing queries from one sequence while keys and values come from another. In encoder-decoder architectures, the decoder uses cross-attention to attend to encoder outputs, enabling translation and conditional generation tasks. Each decoder layer typically contains both self-attention (for previous decoder positions) and cross-attention (for encoder context).

**Choosing the Right Attention Mechanism**

For sequences under 2K tokens with sufficient GPU memory, standard self-attention remains the default choice, offering maximum accuracy. Between 2K-16K tokens, Flash Attention provides the best balance, maintaining full attention with better efficiency. Beyond 16K tokens or under tight memory constraints, sparse patterns or linear approximations become necessary—use local windows for tasks with strong locality bias, and linear attention when global context remains important but approximate attention suffices.

## Putting It Together: Self-Attention in Production

Once you understand self-attention mechanics, the next step is deploying it in real systems. Start by choosing between pre-built implementations and custom code. For most production use cases, **PyTorch's `nn.MultiheadAttention`** provides a battle-tested, optimized starting point that handles weight initialization, masking, and gradient flow correctly. Hugging Face's Transformers library offers model-specific attention layers with pre-trained weights, saving weeks of training time. Reserve custom implementations for research or when you need non-standard variants like sparse attention or custom masking patterns.

**Monitor attention weight distributions** during training to catch anomalies early. Log the mean, standard deviation, and entropy of attention weights after the softmax operation. Healthy attention patterns show gradual specialization—early training has high entropy (uniform attention), while later stages develop sharper, focused patterns. Sudden spikes in entropy or collapsed attention (all weights focus on one token) signal learning problems or data issues.

**Profile memory and compute** bottlenecks before scaling to longer sequences. Use PyTorch Profiler to identify which operations dominate wall-clock time—typically the scaled dot-product and softmax. For GPU workloads, NVIDIA Nsight Systems reveals kernel launch overhead and memory transfer bottlenecks. The quadratic memory scaling of attention (O(n²) for sequence length n) makes profiling essential when moving from 512-token to 2048-token sequences.

**Apply mixed-precision training** to reduce memory footprint by 50% and increase throughput by 2-3×. Use `torch.cuda.amp.autocast()` to run matrix multiplications in FP16 or BF16 while keeping critical operations like softmax and layer normalization in FP32. BF16 is preferable on Ampere+ GPUs because its wider dynamic range handles the exponential operations in attention without loss scaling.

**Monitor for numerical instability** by adding checks for NaN or Inf values in attention scores after scaling but before softmax. These typically arise from:
- Extremely large logits causing softmax overflow
- Division by zero in attention normalization
- Gradient accumulation errors in mixed precision

Add gradient clipping and consider using `torch.nn.functional.scaled_dot_product_attention` with Flash Attention kernels, which implement numerically stable algorithms that fuse operations and reduce memory bandwidth.