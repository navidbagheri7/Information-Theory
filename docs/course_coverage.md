# Course coverage and corrected statements

The supplied sources were reviewed locally: Cover and Thomas, *Elements of Information Theory*, first edition (1991); Stanford EE376A notes supplied as a 75-page January 2016 PDF; Michèle Wigger's four Mn908/MICAS exercise sheets from September–October 2024. Chapter numbers below refer to the supplied first edition. Original PDFs are preserved under ignored `legacy/originals`; their hashes are in [the manifest](../legacy/manifest.json). This guide contains new explanations and implementation mappings, rather than redistributing course materials.

A theoretical exercise may be best answered by a proof plus a numerical example. A finite simulation illustrates a theorem; it does not replace its proof.

## Sheet 1 — entropy and dependence

| Problem | Result and implementation |
|---|---|
| 1, joint PMF | `foundations/joint_entropy`: H(X)=1, H(X,Y)=1.5, H(X|Y)=0.6887218755, I(X;Y)=0.3112781245 bits. Joint PMF and all quantities are saved. |
| 2, functions | Chain rule gives H(X,f(X))=H(X) and also H(f(X))+H(X|f(X)). Nonnegative conditional entropy proves the inequality. Equality requires f to be one-to-one on positive-probability source symbols; invertibility elsewhere is irrelevant. See `information.py` and the information notebook. |
| 3, parity | Exact enumeration for n=2,3,10. Pair MI is one bit for n=2 and zero for n>2. I(X1;X2|X3) is one bit for n=3 and zero for n>3. Both I(X1,...,Xn-1;Xn) and I(X1;Xn|X2,...,Xn-1) are one bit: the missing parity bit is determined when the other bits are known. |
| 4, conditioning | If X=Y=Z is a fair bit, MI drops from one to zero after conditioning. If X and Y are independent fair bits and Z=X XOR Y, MI rises from zero to one. Conditioning is not monotone for mutual information. The n=3 parity case implements the second example. |
| 5.1, mixing | Averaging two masses is the midpoint of a PMF and its transposition. Entropy concavity and permutation invariance imply entropy cannot decrease. The configured uniform-mixture family illustrates repeated mass equalization. |
| 5.2, fixed masses | Let s=1-p1-pm. The maximum puts s/(m-2) on every remaining symbol; the minimum puts all s on one remaining symbol, with zeros elsewhere. Concavity proves the minimum occurs at a simplex vertex; Jensen proves the maximum. Both feasible extremizers are saved. |

## Sheet 2 — compression and sources

| Problem | Result and implementation |
|---|---|
| 1, prefix codes | Owned Huffman, Shannon and exact exhaustive feasible-length oracle. The specified code has expected length 2.5 bits and is optimal. It is not a greedy Huffman tree for these weights: the smallest 1/24 leaf is paired with 1/12 instead of a remaining 1/16 leaf. Optimal codes need not be obtainable from the particular greedy merge rule. |
| 2, nearly one-bit redundancy | A two-symbol source `(1-delta,delta)` has optimal integer expected length one. Its entropy tends to zero as positive delta tends to zero, so redundancy approaches one bit. Do not use delta=0: a singleton framed alphabet may have an empty codeword. |
| 3, weighted costs | Put a_i=p_i*c_i, A=sum a_i. The real-valued optimum is l_i=log2(A/a_i), cost A*H(a/A). Huffman on weights a_i minimizes integer weighted cost. Rounding the relaxed lengths upward is feasible and gives cost strictly below the relaxed cost plus A for a nontrivial supported alphabet. The weighted and ordinary cases save weights, lengths, codewords and bounds. |
| 4, general stationary processes | All three transformed processes are stationary. Overlapping pairs Y have entropy rate H_X per pair; nonoverlapping pairs Z have rate 2H_X per pair. For V=X_(2i), H_X <= H_V <= 2H_X, per subsampled output. One inequality follows because the even sequence is a function of a 2n-block; the other follows by splitting the whole block into odd/even subsequences and using stationarity and subadditivity. Entropy rates of V are not automatically computable from a one-step transition matrix for a general process. |
| 5, Markov example | Stationary law `(3/7,4/7)`, rate 0.9649839289 bits/output. Subsampling is Markov with P squared, rate 0.9846718866 bits/subsampled output. Adjacent swaps are not first-order Markov: the saved conditional-MI witness is positive. They preserve rate because block permutations are invertible and boundary effects vanish. Anchored sums are invertible via X_k=Z_k-X_(k-1), preserve rate, but have a positive non-Markov witness. See `sources.transformation_examples`. |
| 6, window LZ | `lossless.py` supplies an explicit bounded-window LZ77 encoder/decoder and real rate measurements. The sheets do not specify the class token/integer framing completely. The supplied encoded bitstring therefore has no defensible unique decoded answer here. Exact class-format reproduction requires those missing conventions; this limitation is stated, not filled with an invented result. The two input strings can be round-tripped under the documented owned format. |

## Sheet 3 — typicality and channels

| Problem | Result and implementation |
|---|---|
| 1.1, strong implies weak | On supported symbols, absolute surprisal deviation is bounded by `(epsilon/alphabet_size)*sum_a abs(log2 Q(a))`, so it tends to zero with epsilon. The sheet's global codomain `[0,1]` is too restrictive for arbitrary Q and all epsilon: for Q=(0.1,0.9), a sequence with frequencies (0.6,0.4) is strongly 1-typical under the sheet's convention, but its surprisal deviation is 0.5*log2(9)=1.58496 bits. The theorem is the local implication for sufficiently small epsilon; use the displayed linear bound with a nonnegative real codomain. Zero-mass symbols are excluded. |
| 1.2, reverse implication | A uniform source gives constant surprisal for every sequence. Even an all-identical-symbol sequence is weakly typical with tolerance zero and need not be strongly typical. Exact multinomial-count simulations compare both definitions. |
| 2, course channels | W1 rows `(1,0,0)`, `(1/2,1/2,0)`, `(0,0,1)`; C=log2(9/4), maximizing input `(1/3,2/9,4/9)`. W2 rows `(1,0,0)`, `(1/2,1/4,1/4)`, `(0,1/2,1/2)`; C=1. Distinguishing input 0 from input 2 achieves one bit. W2=W1*T where T leaves output 0 unchanged and splits both other symbols uniformly between outputs 1/2. Data processing bounds C2 by C1; direct reduction to a binary observation establishes C2 exactly. Numerical solvers retain certified bounds. |
| 3, Z channel | For crossover 1/2, capacity log2(5/4), maximizing input `(0.6,0.4)`. General crossover curves are also evaluated. |
| 4, symmetry | BSC is symmetric. The configured 2x4 weakly symmetric example has row permutations and equal column sums, but columns are not all permutations. Row entropies are equal, so H(Y|X) is constant; uniform input makes output uniform and achieves `log2(|Y|)-H(first row)`, the universal upper bound. |
| 5, strictness | The statements without 'strict' are valid: MI is concave in input PMF and convex in channel law, and capacity is convex in channel law as a supremum of convex functions. The blanket strict claims are false. Any input-independent channel has zero MI for all input PMFs. Two distinct input-independent channels and their mixture have zero MI and capacity, giving equality in convexity. The explicit counterexample is saved and tested. |

## Sheet 4 — side information and distortion

| Problem | Result and implementation |
|---|---|
| 1, jammer | Unknown state gives effective crossover `q*epsilon+(1-q)/2` and capacity `1-Hb(effective crossover)`. Decoder-known state gives `q*(1-Hb(epsilon))`. With both sides informed, communicate only on free positions using a capacity-approaching BSC code; their asymptotic frequency q gives the same rate. The unknown/known discrete channels are evaluated numerically. |
| 2, symbol-error converse | Block error tending to zero implies average symbol error tends to zero, since each symbol error is contained in the block-error event. Chain rule, dropping conditioning, symbolwise Fano and concavity of Hb give `H(U^K|Z^K)/K <= Hb(Pb)+Pb*log2(a-1)`. For IID source symbols and one channel use per source symbol, data processing and capacity imply `H(V) <= C + Hb(Pb)+Pb*log2(a-1)`. Thus H(V)>C excludes Pb tending to zero. For stationary sources replace H(V) by entropy rate; unequal source/channel use ratios change the capacity term accordingly. Actual code comparisons show both BER and BLER. |
| 3, infinite distortion | Only errors 1→0 are permitted. If their joint mass is D, the allowed reconstruction kernel yields `R(D)=Hb(1/2-D)-(1/2)*Hb(2D)` for 0<=D<=1/2, and zero above 1/2. The certified solver uses exactly zero forbidden-transition probabilities and matches this analytic function. |
| 4, row offsets | For every reconstruction kernel, `E[d']=E[d]-sum p_i*w_i`, while MI is unchanged. The feasible sets are identical after shifting the constraint, hence `R'(D)=R(D+E[w])`. Choosing w_i as the row minimum sets each row's minimum to zero when it is finite. No simulated optimization is needed to prove this identity. |

## Beyond the sheets

| Source concept | Practical implementation |
|---|---|
| Cover–Thomas chapters 2–5 | Information algebra, IID AEP, stationary sources, prefix/arithmetic/LZ coding, real code and genomic corpora. |
| Chapters 8–10 | Certified finite-channel capacity; actual Hamming, convolutional, Polar and LDPC codes; BPSK information; Gaussian water filling and normal approximations with explicit limits. |
| Chapter 12 and Stanford types notes | Exact rare-event probabilities, tilted importance sampling, method-of-types bounds, randomized Neyman–Pearson tests and Stein convergence. |
| Chapter 13 and Stanford separation notes | Certified discrete rate–distortion, quantized Gaussian sources, real scalar streams, equal-use analog/digital comparison, traditional/learned image bitstreams. |
| Further statistical learning | KSG/Gaussian/histogram MI estimation; held-out MINE-style DV and InfoNCE; discrete information bottleneck with restart diagnostics. |

Gambling, Kolmogorov complexity and general multiuser network regions are not implemented as if a short simulation established their theorems. They are honest future extensions. The practical scope concentrates on reproducible compression, communication, estimation and optimization.
