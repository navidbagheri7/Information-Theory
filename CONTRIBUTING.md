# Contributing

Changes should improve a scientific result, its explanation, or its reproducibility. Keep new material in English. Use explicit seeded configurations and cite primary method references. Add an independent oracle when changing numerical algorithms: a closed-form case, a generic solver, exhaustive likelihoods, or a separately computed statistic is stronger evidence than a test that repeats the implementation.

Install with `python -m pip install -e '.[dev,neural,images,notebooks]'`, then run `ruff check src scripts tests`, `ruff format --check src scripts tests`, and `pytest`. The small protocol is `information-lab run --config configs/quick.json --output results/smoke`.

Do not silently replace recorded results after changing an algorithm. Use a fresh output directory, record source and input hashes, rerun affected protocols, and regenerate the report and explorer. Preserve negative findings, zero-hit uncertainty and unsuccessful convergence. Finite neural estimates must not be called certified bounds; an IB fixed point must not be called globally optimal. Keep original course PDFs, external checkpoints, credentials and caches outside Git.

Changes to codec measurement must use actual encoder/decoder bytes and compare the precise RGB8 pixels displayed in the gallery. Include container overhead, code termination and the relevant energy/time-unit convention. A new explorer field must display real applicable records and pass the browser control audit, including mobile layout and link/image checks.

Public publishing is a separate manual owner action. No development or preview script pushes a repository or deploys a website.
