# Sources, attribution and scope

The primary theoretical reference is Thomas M. Cover and Joy A. Thomas, *Elements of Information Theory*, first edition, Wiley, 1991. The supplied textbook and original course PDFs remain private. [Course coverage](course_coverage.md) identifies the supplied edition, handouts and corrected statements. [Stanford EE376A lecture notes](https://web.stanford.edu/class/ee376a/files/lecture_notes.pdf) provide a publicly accessible related teaching resource; the downloaded course PDF supplied by the owner is a different January 2016 revision.

Primary method sources:

- Blahut, “Computation of channel capacity and rate-distortion functions,” IEEE Transactions on Information Theory, 1972; Arimoto, “An algorithm for computing the capacity of arbitrary discrete memoryless channels,” 1972. The implemented certificates are explained in the mathematical contracts and checked against analytic and generic-solver oracles.
- [Arıkan, channel polarization](https://arxiv.org/abs/0807.3917): the owned Polar example uses BEC construction and successive cancellation, with explicit finite-profile limits.
- Kraskov, Stögbauer and Grassberger, “Estimating mutual information,” Physical Review E, 2004: continuous KSG estimator with strict neighbor conventions.
- [Belghazi et al., Mutual Information Neural Estimation](https://proceedings.mlr.press/v80/belghazi18a.html): the MINE-style implementation declares its bounded critic and EMA normalization.
- [Poole et al., On Variational Bounds of Mutual Information](https://proceedings.mlr.press/v97/poole19a.html): population bounds, finite estimation tradeoffs and contrastive limits motivate the measured comparisons.
- Tishby, Pereira and Bialek, “The information bottleneck method,” 1999: self-consistent discrete updates and relevance/compression tradeoffs; the implemented solver claims local stationary solutions only.
- [CompressAI source](https://github.com/InterDigitalInc/CompressAI) and [official training documentation](https://interdigitalinc.github.io/CompressAI/tutorials/tutorial_train.html): official pretrained image-codec implementations and weights. Model URLs and SHA256 hashes are in [model provenance](model_provenance.json). Ballé factorized/hyperprior, Minnen mean-scale and Cheng context/residual models are compared with real bitstreams; these experiments are not a new image-codec training or a current state-of-the-art benchmark.

The four image sources and licenses follow the [scikit-image data documentation](https://scikit-image.org/docs/stable/api/skimage.data): coffee by Rachel Michetti (CC0), Chelsea by Stefan van der Walt (CC0), the NASA Eileen Collins astronaut photograph (public domain), and the SpaceX rocket image (public domain). Exact dimensions, preprocessing and hashes are in [the image manifest](../assets/images/manifest.json). All reconstructions are derived from these credited inputs.

The genomic fixture uses NCBI RefSeq E. coli chromosome NC_000913.3, assembly GCF_000005845.2, first 200,000 bases. The upstream URL, checksum and full-sequence/excerpt hashes are in [the corpus manifest](../assets/corpora/manifest.json). The other corpus consists of the owner's MIT scientific Python code, with component hashes. Bundling short factual sequence/code fixtures keeps core examples offline and reproducible.

The MIT license applies to this repository's original code and explanations. It does not relicense original course materials, external model checkpoints or third-party packages. External dependencies retain their own licenses; pretrained weights are downloaded from their official sources and are not redistributed.
