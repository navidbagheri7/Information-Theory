# Measured reference results

The configured protocol contains 569 cases and 1117 measured runs. All tables link to recorded evidence. See [measurement contracts](mathematical_contracts.md) for units, assumptions, uncertainty and limits, and [the explorer](explorer.html) for field-specific comparisons.

| Field | Recorded runs |
|---|---:|
| Information bottleneck | 81 |
| Channel capacity | 51 |
| Noisy communication | 120 |
| Information estimation | 324 |
| Entropy and dependence | 21 |
| Real image compression | 96 |
| Lossless compression | 154 |
| Learned information estimators | 54 |
| Scalar quantization | 48 |
| Rate and distortion | 90 |
| Source–channel transmission | 30 |
| Sources and typicality | 25 |
| Rare events and hypothesis tests | 23 |

## Selected findings

These numerical examples show practical behavior; they are not universal method rankings.

- At 0 dB and seven channel uses per Gaussian source symbol, analog MSE is 0.37481; digital Hamming soft-ML MSE is 3.84462, and source-prior posterior-mean MSE is 1.04249. The matched-use power budget makes this a meaningful finite-code comparison.
- For 16 dimensions and correlation 0.95, Gaussian MI is 26.86763 bits. InfoNCE estimates range from 7.99554 to 7.99699 bits across three seeds, approaching the 8-bit batch ceiling rather than the truth. MINE-style estimates and strong Gaussian/KSG baselines are available alongside this result.
- At binomial blocklength 500, naive simulation observes 0 tail hits in 10000 trials while the exact natural-log probability is -79.78167. Importance sampling estimates the same rare probability with reported relative uncertainty.

## Image codec examples

Four attributed photographs, complete actual containers, RGB8 decoded-pixel metrics, CPU timing medians. Pretrained model weights are shared external inputs and not amortized per image. This small resized set supports a demonstration, not a benchmark or SOTA claim.

| Coffee codec | Point | Complete bpp | PSNR dB | MS-SSIM | Decode seconds |
|---|---:|---:|---:|---:|---:|
| avif | 35 | 0.37817 | 29.585 | 0.96446 | 0.00644 |
| avif | 60 | 1.05558 | 34.415 | 0.98825 | 0.00801 |
| avif | 85 | 2.13761 | 38.580 | 0.99481 | 0.01247 |
| bmshj2018-factorized | 1 | 0.15999 | 26.539 | 0.91642 | 0.13094 |
| bmshj2018-factorized | 3 | 0.30843 | 28.456 | 0.95507 | 0.14885 |
| bmshj2018-factorized | 5 | 0.65348 | 30.851 | 0.97883 | 0.14642 |
| bmshj2018-hyperprior | 1 | 0.16764 | 26.799 | 0.91979 | 0.15888 |
| bmshj2018-hyperprior | 3 | 0.35124 | 29.391 | 0.96006 | 0.11761 |
| bmshj2018-hyperprior | 5 | 0.71712 | 32.222 | 0.98280 | 0.11789 |
| cheng2020-anchor | 1 | 0.13932 | 27.713 | 0.92808 | 1.47211 |
| cheng2020-anchor | 3 | 0.27799 | 29.887 | 0.96172 | 1.48661 |
| cheng2020-anchor | 5 | 0.61719 | 33.434 | 0.98405 | 2.26499 |
| jpeg | 35 | 0.95955 | 29.538 | 0.96814 | 0.00071 |
| jpeg | 60 | 1.45036 | 31.408 | 0.98169 | 0.00095 |
| jpeg | 85 | 2.68734 | 35.105 | 0.99251 | 0.00117 |
| jpeg2000 | 20 | 1.19979 | 30.846 | 0.96281 | 0.01160 |
| jpeg2000 | 10 | 2.39811 | 34.947 | 0.98443 | 0.01623 |
| jpeg2000 | 5 | 4.79883 | 41.314 | 0.99641 | 0.02493 |
| mbt2018-mean | 1 | 0.15592 | 26.808 | 0.92029 | 0.12452 |
| mbt2018-mean | 3 | 0.33105 | 29.587 | 0.96042 | 0.12407 |
| mbt2018-mean | 5 | 0.72298 | 33.431 | 0.98368 | 0.22090 |
| webp | 35 | 0.62061 | 30.022 | 0.96284 | 0.00150 |
| webp | 60 | 0.91976 | 31.847 | 0.97517 | 0.00198 |
| webp | 85 | 1.72363 | 34.884 | 0.98733 | 0.00198 |

All scalar metrics are in [comparison.csv](comparison.csv), full row/asset associations in [results.json](results.json), and independent validation in [reference_validation.json](reference_validation.json). Exact numeric quality settings differ between codec families; compare rate and decoded quality.
