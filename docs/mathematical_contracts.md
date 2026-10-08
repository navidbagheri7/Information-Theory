# Mathematical and measurement contracts

All reported information quantities use **bits**. Natural logarithms are used internally by optimizers; `beta_nats` multiplies distortion in the objective `I_nats + beta_nats * D`. Array axes and conventions are specified below. Observed finite-sample quantities are never silently clipped to a theoretical answer.

## Information algebra

`information.py` accepts normalized, finite, nonnegative PMFs. It checks normalization rather than silently repairing it. Zero-probability entropy terms contribute zero. KL divergence is infinite when the first distribution assigns mass outside the second's support. Joint PMFs have axes `(X,Y)`; conditional-MI arrays have axes `(X,Y,Z)` and compute `I(X;Y|Z)`. A deterministic function reduces entropy, with equality precisely when the function is invertible on the source's positive-probability support.

Differential entropy is not an ordinary discrete entropy and can be negative or change under a coordinate rescaling. Gaussian mutual information is invariant to invertible changes of coordinates. This repository does not compare discretized entropy to differential entropy as though they were the same quantity.

## Sources and asymptotics

Markov matrices have rows indexed by the current state and columns by the next state. A unique stationary distribution is required. The entropy rate is the stationary average row entropy, per output symbol. Subsampling changes the time unit: a subsampled output corresponds to two original time steps. Reordered and anchored-sum examples preserve entropy rate but need not remain first-order Markov.

Typicality experiments draw exact multinomial counts of IID blocks, sufficient for the tested definitions. They do not approximate an IID block by unrelated continuous samples. Strong typicality uses absolute frequency tolerance `epsilon/alphabet_size`; weak typicality uses per-symbol surprisal tolerance. The uniform source provides a counterexample to weak implying strong typicality.

Rare-event experiments store natural-log probabilities and importance weights. Zero naive hits do not imply zero event probability. Importance-sampling uncertainty is a Monte Carlo standard error, separate from rigorous method-of-types inequalities. Neyman–Pearson thresholds include boundary randomization to meet the false-alarm budget; Stein's exponent is a limiting result, not an equality at each finite blocklength.

## Certified optimization

For channel capacity, the Blahut–Arimoto iteration returns

`I(q,W) <= C(W) <= max_x KL(W[x,:] || qW)`.

The reported gap is in bits. For rate–distortion, the returned stochastic reconstruction kernel supplies a feasible primal value. Scaling `a_x=1/Z_x` to satisfy every dual reproduction constraint supplies a lower bound at the achieved distortion. A small parameter change alone is not a certificate. Rate–distortion trace bounds refer to each iteration’s attained distortion, which changes along the trace; they are not bounds on one fixed-D value throughout optimization. Infinite distortion entries are forbidden transitions even at beta zero.

The 17-symbol Gaussian example is a **finite discretization** with Gaussian CDF bin masses and squared center distances. It is not the continuous Gaussian source. It uses a declared convex reproduction-marginal warm start if ordinary BA is slow; convergence still requires the primal/dual gap. Its tolerance is 1e-6 bits, versus 1e-8 for the other configured discrete sources. The asymptotic continuous Gaussian curve is shown separately as a limit.

Discrete information bottleneck uses self-consistent updates and multiple random starts. Its residual certifies a fixed point, not a global optimum. Restarts and objective traces remain available, including unsuccessful convergence.

## Actual lossless coding

Every method has a decoder and byte-exact round-trip checks. Complete rates count everything in the saved file. The owned `ITL1` frame has a 17-byte magic/method/length/CRC32 header. Huffman and Shannon formats additionally transmit 256 uint32 symbol counts (1024 bytes); this two-pass model can dominate small messages. Singleton alphabets need zero payload bits because the frame supplies the decoded length.

Arithmetic coding uses a 32-bit integer interval, E1/E2/E3 renormalization, unit symbol pseudocounts and count rescaling at 32768. Order zero has one context; order one uses previous-byte contexts. Four physical lookahead bytes and padding count in the file. LZ77 transmits a uint16 window, then either a zero tag plus an eight-bit literal, or a one tag, `ceil(log2(window))` bits for `distance-1`, and an Elias-gamma match length. Matches have length 3–258; at most 256 candidates are searched. This explicit format is not a guessed reconstruction of the class-only LZ convention. DEFLATE and Zstandard use their actual standard-library streams.

Byte entropy on text or DNA is an empirical marginal statistic, not a certified entropy rate. In the binary synthetic sources, one source byte stores one binary symbol, so bits/source-byte are bits/binary-symbol. Container payload columns include byte padding; unavailable standard-codec internal payload splits are explicitly labelled unavailable.

## Noisy channel codes

Uniform 64-bit messages are independent experimental units. Noise is real AWGN, modulation is BPSK with symbol energy one. With actual code rate `r=k/n`, variance is `1/(2*r*Eb/N0)`. All six convolutional termination bits count in `n`. The comparisons therefore use a common energy per message bit, while code lengths and decoding costs differ.

Repetition uses summed soft LLRs. Hamming (7,4) uses syndrome or exhaustive 16-word soft ML per block. Convolutional coding uses constraint length seven and generators (133,171) in octal, with exact terminated soft Viterbi decoding. Owned short IRA-LDPC codes use balanced sparse A and dual-diagonal parity structure, decoded with sum-product BP or normalized min-sum (factor 0.8), maximum 50 iterations. A zero syndrome is not evidence of a correct message. Polar uses a BEC-designed Bhattacharyya construction and exact successive cancellation; it is neither AWGN-optimal construction, SCL decoding, nor a full 5G implementation.

BER divides total decoded bit errors by transmitted message bits. BLER counts messages with at least one error. Wilson intervals use independent message trials; decoded bits within a message are correlated. Zero observed errors still have a nonzero confidence upper bound. Binary-input BPSK information and unconstrained Gaussian-input capacity are distinguished. Finite-length normal approximations are labelled approximations and retain negative values where the approximation breaks down.

## Estimation and learned objectives

Classical estimators use identical independent sample pairs for a seed: Gaussian plug-in assumes a Gaussian joint law, sparse histograms estimate discretized MI with finite-cell bias, and KSG uses strict Chebyshev neighbor balls with explicit rejection of joint ties. High-dimensional histogram and KSG failures are expected phenomena, not automatically implementation bugs.

MINE-style DV uses a bounded neural critic and EMA gradient normalization; InfoNCE uses separable embedding critics. These are explicit educational implementations, not claims of byte-for-byte reproduction of published training recipes. Gaussian train/validation/test draws have disjoint fixed seeds. Checkpoints are selected using validation only; known MI is used only for final assessment. Models export as NumPy arrays, loaded with `allow_pickle=False`. Population variational bounds do **not** certify finite empirical estimates. InfoNCE has ceiling `log2(batch_size)`; test batches are disjoint and their individual objective estimates are saved for uncertainty calculation. Negative or excessive empirical estimates remain visible.

## Quantization and source–channel separation

Uniform quantizers use known prior parameters. Lloyd–Max centers fit independent training data; test data determine held-out distortion only. Complete index streams transmit their Huffman frequency table and all float64 reconstruction centers. Gaussian R(D) is an asymptotic lower limit, not the measured finite scalar-code rate.

The joint source/channel example uses Gaussian source mean one and variance three. Four/eight quantization bits require seven/fourteen Hamming-coded channel uses. Repeated analog transmission uses the same number of channel uses and expected power one per use. The mean is removed before transmitting; failing to center it would violate the declared power budget. Digital soft ML, source-prior MAP, and exact source-prior posterior mean have different objectives. The analog estimate is the Gaussian MMSE estimate. Quantization-only distortion, observed analog power and clipping fraction are shown. The ideal separation distortion assumes asymptotic coding and Gaussian channel inputs; it is not achieved by these short BPSK codes.

## Image codec measurement

All codecs receive the exact same RGB8 images. Traditional codecs are JPEG 4:4:4, WebP, AVIF 4:4:4 and irreversible JPEG2000. Learned codecs use official MSE-trained CompressAI factorized, scale-hyperprior, mean-scale-hyperprior and Cheng context/residual checkpoints, qualities 1/3/5. Numeric quality settings across codec families are not equivalent; compare actual rate and distortion.

All reported image rates come from actual encoded bytes. Learned estimated likelihood bpp, entropy-stream payload bpp and complete `ITIC1` container bpp are separate. The container records architecture, quality, checkpoint SHA256, original/padded/latent dimensions and stream lengths. Padding affects rate; original pixel count is the denominator. Shared model weights are not amortized into each image's rate. The decoder checks model identity.

PSNR, SSIM and MS-SSIM use the exact rounded decoded RGB8 pixels displayed in the gallery. Three measured coding repetitions must reproduce bytes and pixels; encode/decode medians and individual timing arrays are saved. Model download/loading and likelihood estimation are outside the timed encoding interval. These are four credited photographs resized to at most 384 pixels per side, a small demonstration set with **no benchmark or state-of-the-art claim**. No pretrained weights are redistributed.
