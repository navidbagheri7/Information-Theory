"""Create explanatory notebooks with computations and measured reference evidence."""

import argparse
import subprocess
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ruff", default="ruff", help="Ruff executable used to format learning code")
FORMATTER = parser.parse_args().ruff
SETUP = """from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/information_lab').is_dir() and (p/'pyproject.toml').exists())
sys.path.insert(0, str(ROOT/'src'))

def query(family, **wanted):
    seed = wanted.pop('seed', 41)
    records = []
    for path in sorted((ROOT/'results/reference'/family).glob('*/run.json')):
        record = json.loads(path.read_text())
        if record['seed'] == seed and all(record['config'].get(k) == v for k,v in wanted.items()):
            records.append((record, path.parent))
    if not records:
        raise ValueError('No applicable recorded evidence')
    return records

def arrays(path):
    with np.load(path/'data.npz', allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}

def measured(record):
    display(Markdown('**Recorded configuration**'))
    print(json.dumps(record['config'], indent=2))
    print(json.dumps(record['metrics'], indent=2))

def preview(record):
    key = record['config']['id']+'_s'+str(record['seed'])
    display(Image(filename=str(ROOT/'assets/experiments'/record['config']['family']/(key+'.png'))))
"""


def write(name, title, lead, cells):
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# "
            + title
            + "\n\n"
            + lead
            + "\n\n[Interactive explorer](../docs/explorer.html) · [Measurement contracts](../docs/mathematical_contracts.md) · [Course proofs](../docs/course_coverage.md)"
        ),
        nbf.v4.new_code_cell(SETUP),
    ]
    for kind, text in cells:
        notebook.cells.append(
            nbf.v4.new_code_cell(text) if kind == "code" else nbf.v4.new_markdown_cell(text)
        )
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    for cell in notebook.cells:
        if cell.cell_type == "code":
            result = subprocess.run(
                [FORMATTER, "format", "--stdin-filename", "lesson.py"],
                input=cell.source,
                text=True,
                capture_output=True,
                check=True,
            )
            cell.source = result.stdout.strip()
    path = ROOT / "notebooks" / (name + ".ipynb")
    nbf.write(notebook, path)
    print("Created", path.name)


write(
    "00_entropy_and_dependence",
    "Entropy, functions and hidden dependence",
    "Exact finite distributions expose the chain rule, parity synergy and the effect of conditioning. These are proofs with numerical illustrations, not inference from a fitted model.",
    [
        (
            "code",
            """from information_lab.information import entropy, mutual_information, conditional_entropy, conditional_mutual_information
joint=np.array([[.25,.25],[0,.5]])
print({'H(X)':entropy(joint.sum(1)), 'H(X,Y)':entropy(joint), 'H(X|Y)':conditional_entropy(joint), 'I(X;Y)':mutual_information(joint)})
assert np.isclose(entropy(joint), entropy(joint.sum(0))+conditional_entropy(joint))""",
        ),
        (
            "markdown",
            "A deterministic function cannot increase entropy. Equality requires invertibility on the supported source symbols. Merging two source labels gives a concrete strict example.",
        ),
        (
            "code",
            """p=np.array([.4,.3,.2,.1]); labels=np.array([0,0,1,1]); function_p=np.bincount(labels,weights=p)
print('Before:',entropy(p),'bits; after merging labels:',entropy(function_p),'bits')""",
        ),
        (
            "code",
            """from information_lab.sources import parity_law, project_law
x,p=parity_law(3)
xyz=project_law(x,p,[2,2,2])
print('Pairwise MI:',mutual_information(xyz.sum(2)))
print('Conditional MI given parity bit:',conditional_mutual_information(xyz))
identical=np.zeros((2,2,2));identical[0,0,0]=identical[1,1,1]=.5
print('X=Y=Z: MI',mutual_information(identical.sum(2)),'conditional MI',conditional_mutual_information(identical))""",
        ),
        (
            "code",
            """record,path=query('foundations',task='parity',variables=10)[0]
measured(record);preview(record)""",
        ),
        (
            "markdown",
            "For n=10, every pair is independent, yet the first nine bits determine the last. This distinction is useful when pairwise diagnostics miss a higher-order constraint. The supplied strict concavity/convexity statements have explicit equality counterexamples in the course guide.",
        ),
    ],
)
write(
    "01_sources_and_typicality",
    "Sources, entropy rates and typicality",
    "Time units matter: one subsampled output spans two original transitions. Invertible transforms can preserve entropy rate while destroying a first-order Markov description.",
    [
        (
            "code",
            """from information_lab.sources import stationary, entropy_rate, transformation_examples
transition=np.array([[1/3,2/3],[.5,.5]])
print('Stationary law:',stationary(transition))
print('Original bits/output:',entropy_rate(transition))
print('Subsampled bits/output:',entropy_rate(transition@transition))
print({k:v for k,v in transformation_examples().items() if 'conditional_mi' in k})""",
        ),
        (
            "markdown",
            "Weak typicality controls surprisal; strong typicality controls every empirical frequency. Under a uniform source, all sequences have identical surprisal, including very unbalanced sequences.",
        ),
        (
            "code",
            """from information_lab.sources import typicality_counts
uniform=typicality_counts([.25]*4,16,10000,41,.1,.2)
skewed=typicality_counts([.6,.25,.1,.05],16,10000,41,.1,.2)
print('Uniform weak/strong fractions:',uniform['weak'].mean(),uniform['strong'].mean())
print('Skewed weak/strong fractions:',skewed['weak'].mean(),skewed['strong'].mean())""",
        ),
        (
            "code",
            """record,path=query('sources',task='typicality',probability=[.25]*4,blocklength=16)[0]
measured(record);preview(record)""",
        ),
        (
            "code",
            """records=query('sources',task='typicality',probability=[.6,.25,.1,.05])
records=sorted(records,key=lambda rp:rp[0]['config']['blocklength'])
for metric in ['weak_fraction','strong_fraction']:
    plt.plot([r['config']['blocklength'] for r,_ in records],[r['metrics'][metric] for r,_ in records],'o-',label=metric)
plt.xscale('log');plt.ylim(0,1.05);plt.xlabel('IID blocklength');plt.ylabel('Observed typical fraction');plt.legend();plt.show()""",
        ),
    ],
)
write(
    "02_real_lossless_streams",
    "Real lossless streams and model costs",
    "A compressor must supply a decoder. Whole-file accounting exposes when a statistically good model loses on a short input because its description costs too much.",
    [
        (
            "code",
            """from information_lab.lossless import compress, decompress
methods=['huffman','shannon','arithmetic0','arithmetic1','lz77','deflate','zstd']
raw=b'ABABABABABABABABABAB'*100
for method in methods:
    encoded=compress(raw,method)
    assert decompress(encoded,method)==raw
    print(method, len(encoded),'bytes,',8*len(encoded)/len(raw),'bits/source byte')""",
        ),
        (
            "markdown",
            "The sheet’s window-LZ token convention is incomplete. The following round trips use this repository’s explicitly documented format. ASCII 0/1 bytes each represent one binary source symbol; headers count. They do not claim to decode the unidentified class bitstring.",
        ),
        (
            "code",
            """for raw in [b'1100000001111100111111110000', b'00000111011100111']:
    encoded=compress(raw,'lz77',window=4)
    assert decompress(encoded,'lz77')==raw
    print('Source symbols',len(raw),'complete bits/symbol',8*len(encoded)/len(raw))""",
        ),
        (
            "code",
            """record,path=query('lossless',task='markov_binary',max_bytes=65536,method='arithmetic1')[0]
measured(record);preview(record)""",
        ),
        (
            "code",
            """display(Image(filename=str(ROOT/'assets/previews/lossless_comparison.png')))
print('Corpus provenance:')
print(json.dumps(json.loads((ROOT/'assets/corpora/manifest.json').read_text()),indent=2))""",
        ),
        (
            "markdown",
            "A dependent Markov source can be compressed below its marginal binary entropy. Real-corpus byte entropy is a descriptive statistic, not a certified source entropy rate. The MIT code and versioned genomic excerpt have explicit origin hashes.",
        ),
    ],
)
write(
    "03_capacity_and_allocation",
    "Certified capacity and Gaussian allocation",
    "A stopping tolerance should concern the quantity of interest. The capacity solver returns a provable interval, rather than trusting a small change in parameters.",
    [
        (
            "code",
            """from information_lab.channels import bsc, bec, z_channel, capacity, water_filling, bsc_normal_approximation
from information_lab.information import binary_entropy
for channel,truth in [(bsc(.1),1-binary_entropy(.1)),(bec(.25),.75),(z_channel(.5),np.log2(5/4))]:
    result=capacity(channel)
    print(result['lower_bits'],truth,result['upper_bits'],'input',result['input'])
    assert result['lower_bits']-1e-9<=truth<=result['upper_bits']+1e-9""",
        ),
        (
            "code",
            """allocation=water_filling([.5,1,2,4],4)
print(allocation)
assert np.isclose(allocation['power'].sum(),4)
print('Short-block approximation:',bsc_normal_approximation(.1,32,.001))""",
        ),
        (
            "markdown",
            "The noise/interference state can itself be useful information. Decoder-known jammer state and unknown state have different capacities; the supplied course channels also have exact checks and a degradation relation.",
        ),
        (
            "code",
            """record,path=query('channels',task='course_channel1')[0]
measured(record);preview(record)
print('Second course channel:',query('channels',task='course_channel2')[0][0]['metrics'])""",
        ),
        (
            "code",
            """record,path=query('channels',task='water_filling',total_power=4)[0]
preview(record)""",
        ),
    ],
)
write(
    "04_noisy_message_coding",
    "Noisy messages, soft decoding and modern code families",
    "Eight actual short-code methods use independent 64-bit messages and a common Eb/N0 convention. Every encoder/decoder is tested against noiseless cases and independent algebra or likelihood oracles.",
    [
        (
            "code",
            """from information_lab.coding import hamming_encode, hamming_decode
message=np.array([[1,0,1,1]],dtype=np.uint8)
word=hamming_encode(message); corrupted=word.copy();corrupted[0,2]^=1
restored=hamming_decode(10*(1-2*corrupted.astype(float)))
print('Message',message,'coded',word,'one flipped bit',corrupted,'decoded',restored)
assert np.array_equal(restored,message)""",
        ),
        (
            "markdown",
            "Soft LLRs retain reliability that hard decisions discard. Hamming soft ML, terminated convolutional/Viterbi, owned short IRA-LDPC sum-product/min-sum and BEC-designed Polar SC have explicit construction limits. This is not a 5G profile or a guarantee that a syndrome-valid word is correct.",
        ),
        (
            "code",
            """for record,path in query('coding',ebn0_db=3):
    m=record['metrics']
    print(record['config']['method'],'rate',m['actual_rate'],'BER',m['bit_error_rate'],'BLER',m['block_error_rate'],'95% message interval',m['block_error_interval'])""",
        ),
        (
            "code",
            """record,path=query('coding',method='ldpc_bp',ebn0_db=1)[0]
measured(record);preview(record)""",
        ),
        ("code", """display(Image(filename=str(ROOT/'assets/previews/coding_comparison.png')))"""),
        (
            "markdown",
            "The pooled plot counts independent messages across seeds. A downward marker means zero observed message errors with a nonzero Wilson confidence upper bound. It is not a fabricated small error probability.",
        ),
    ],
)
write(
    "05_rate_distortion_certificates",
    "Rate–distortion kernels and forbidden transitions",
    "Rate–distortion optimization has a feasible reconstruction kernel and a separately feasible dual lower bound. A plot of iterations alone would not establish optimality.",
    [
        (
            "code",
            """from information_lab.rate_distortion import solve, bernoulli_hamming, asymmetric_forbidden
source=np.array([.5,.5]);cost=np.array([[0,1],[1,0]])
result=solve(source,cost,2)
truth=bernoulli_hamming(.5,result['distortion'])
print('Distortion',result['distortion'],'rate interval',result['lower_rate_bits'],result['upper_rate_bits'],'analytic',truth)
assert result['lower_rate_bits']-1e-8<=truth<=result['upper_rate_bits']+1e-8""",
        ),
        (
            "code",
            """forbidden=np.array([[0,np.inf],[1,0]])
result=solve(source,forbidden,2)
print('Kernel',result['conditional'],'analytic rate',asymmetric_forbidden(result['distortion']))
assert result['conditional'][0,1]==0""",
        ),
        (
            "code",
            """records=query('rate_distortion',task='asymmetric_forbidden')
records=sorted(records,key=lambda rp:rp[0]['metrics']['distortion'])
plt.plot([r['metrics']['distortion'] for r,_ in records],[r['metrics']['upper_rate_bits'] for r,_ in records],'o-',label='Feasible rate')
plt.plot([r['metrics']['distortion'] for r,_ in records],[r['metrics']['analytic_rate_bits'] for r,_ in records],'--',label='Analytic')
plt.xlabel('Expected distortion');plt.ylabel('bits/source symbol');plt.legend();plt.show()""",
        ),
        (
            "code",
            """record,path=query('rate_distortion',task='discretized_gaussian',beta=.7)[0]
measured(record);preview(record)""",
        ),
        (
            "markdown",
            "The Gaussian case has 17 finite source/reconstruction values and Gaussian CDF bin masses. It is not the continuous Gaussian law. A declared convex marginal warm start addresses slow BA convergence; its certificate remains the same dual gap. Row distortion offsets change the feasible distortion threshold, not mutual information.",
        ),
    ],
)
write(
    "06_quantization_and_separation",
    "Quantization and matched-use source/channel transmission",
    "Finite digital coding is not automatically better than analog transmission. A meaningful comparison fixes channel uses and power, centers the nonzero source mean, and distinguishes ML from prior-aware estimation.",
    [
        (
            "code",
            """from information_lab.quantization import lloyd_max, encode, decode
rng=np.random.default_rng(41);training=rng.normal(1,np.sqrt(3),4096);test=rng.normal(1,np.sqrt(3),1024)
centers,fit=lloyd_max(training,16,1,3)
stream=encode(test,centers);restored=decode(stream)
print('Held-out distortion',np.mean((test-restored)**2),'complete bits/sample',8*len(stream)/len(test),'training converged',fit['converged'])""",
        ),
        (
            "code",
            """record,path=query('quantization',method='lloyd_max',levels=16)[0]
measured(record);preview(record)""",
        ),
        (
            "markdown",
            "Four quantization bits use seven Hamming-coded channel uses. Analog repetition gets the same seven uses at expected power one per use. The known Gaussian prior supplies MAP bin probabilities and exact posterior conditional means; these are not fitted to the test source.",
        ),
        (
            "code",
            """record,path=query('source_channel',quantizer_bits=4,snr_db=0)[0]
measured(record);preview(record)""",
        ),
        (
            "code",
            """display(Image(filename=str(ROOT/'assets/previews/source_channel_comparison.png')))""",
        ),
        (
            "markdown",
            "The ideal separation distortion is an asymptotic Gaussian-input limit. Short BPSK/Hamming codes do not achieve it. Analog observed power, clipping, quantization-only distortion and receiver-specific MSE are saved, so performance cannot be improved by silently changing the budget.",
        ),
    ],
)
write(
    "07_rare_events_and_testing",
    "Rare events, importance sampling and exact hypothesis tests",
    "A simulation with no observed rare events needs a better estimator, not a claim that the event is impossible. Log probabilities preserve precision and exact binomial answers provide independent checks.",
    [
        (
            "code",
            """from information_lab.statistics import rare_event, neyman_pearson
result=rare_event(500,.1,.3,10000,41)
print('Naive hits',result['naive_hits'],'exact log probability',result['log_exact_probability'],'IS log probability',result['log_is_probability'],'relative standard error',result['relative_standard_error'])
assert result['type_lower_log_probability']<=result['log_exact_probability']<=result['type_union_upper_log_probability']""",
        ),
        (
            "code",
            """record,path=query('statistics',task='rare_events',blocklength=500)[0]
measured(record);preview(record)""",
        ),
        (
            "markdown",
            "Neyman–Pearson testing randomizes at the binomial threshold to attain the false-alarm budget exactly. Stein’s exponent concerns asymptotic decay of missed detections, rather than an exact finite-block equality.",
        ),
        (
            "code",
            """result=neyman_pearson(64,.1,.3,.05)
print(result)
assert np.isclose(result['false_alarm'],.05)""",
        ),
        (
            "code",
            """records=sorted(query('statistics',task='hypothesis_test'),key=lambda rp:rp[0]['config']['blocklength'])
plt.plot([r['config']['blocklength'] for r,_ in records],[r['metrics']['miss_exponent_bits'] for r,_ in records],'o-',label='Finite miss exponent')
plt.axhline(records[0][0]['metrics']['stein_limit_bits'],ls='--',label='Stein limit')
plt.xscale('log');plt.xlabel('Blocklength');plt.ylabel('bits/source symbol');plt.legend();plt.show()""",
        ),
    ],
)
write(
    "08_classical_information_estimation",
    "Finite-sample information estimation",
    "Known Gaussian truth permits controlled measurement of estimation bias. A correct parametric assumption can be powerful; sparse discretization and nearest-neighbor methods have different high-dimensional limits.",
    [
        (
            "code",
            """from information_lab.estimators import gaussian_pairs, gaussian_truth, gaussian_plugin, ksg, binned_plugin
x,y=gaussian_pairs(1024,4,.7,300041)
print('Known MI',gaussian_truth(4,.7),'Gaussian',gaussian_plugin(x,y),'KSG',ksg(x,y,5),'Binned',binned_plugin(x,y,np.linspace(-3,3,9)))""",
        ),
        (
            "code",
            """for method in ['gaussian_plugin','ksg','binned_plugin']:
    record,path=query('estimation',method=method,dimension=16,correlation=0,samples=4096)[0]
    print(method,record['metrics'])""",
        ),
        (
            "markdown",
            "The independent 16-dimensional example is a stress test: a positive sparse-histogram estimate can appear despite zero true MI. Estimates are left visible rather than clipped or hidden.",
        ),
        (
            "code",
            """record,path=query('estimation',method='ksg',dimension=4,correlation=.7,samples=4096)[0]
measured(record);preview(record)""",
        ),
        ("code", """display(Image(filename=str(ROOT/'assets/previews/mi_comparison.png')))"""),
        (
            "markdown",
            "All three classical estimators use the same held-out pair draws for a seed. KSG uses strict Chebyshev neighbor conventions; the reference audit independently recomputes every estimate with brute-force pairwise distances.",
        ),
    ],
)
write(
    "09_neural_information_bounds",
    "Learned information objectives and contrastive saturation",
    "These models are genuinely trained with separate train/validation/test draws. Population variational bounds do not certify a finite empirical estimate; batch saturation, critic misspecification and variance are measured outcomes.",
    [
        (
            "code",
            """import torch
from information_lab.neural import load, evaluate
torch.set_num_threads(1)
record,path=query('neural',method='mine_dv',dimension=4,correlation=.7)[0]
model,metadata=load(path/'model.npz',device='cpu');data=arrays(path)
result=evaluate(model,torch.tensor(data['test_x'],dtype=torch.float32),torch.tensor(data['test_y'],dtype=torch.float32),record['config']['batch_size'])
print('Portable CPU inference:',result['estimate_bits'],'recorded GPU estimate:',record['metrics']['estimate_bits'])
assert np.isclose(result['estimate_bits'],record['metrics']['estimate_bits'],atol=3e-5)""",
        ),
        (
            "code",
            """measured(record);preview(record)
print('Checkpoint selected on validation:',metadata['selected_step'])""",
        ),
        (
            "markdown",
            "An InfoNCE batch of 256 has ceiling eight bits. At dimension 16 and correlation 0.95, Gaussian MI is approximately 26.87 bits. A nearly perfect eight-bit objective therefore remains a severely underestimated MI value.",
        ),
        (
            "code",
            """for seed in [41,42,43]:
    record,path=query('neural',method='infonce',dimension=16,correlation=.95,seed=seed)[0]
    print(seed,record['metrics'])
assert record['metrics']['estimate_bits']<=np.log2(record['config']['batch_size'])+1e-7""",
        ),
        ("code", """display(Image(filename=str(ROOT/'assets/previews/mi_comparison.png')))"""),
        (
            "markdown",
            "NPZ states use `allow_pickle=False`; model inference does not deserialize arbitrary Python objects. The audit reloads every saved critic on CPU and compares its held-out objective. The correct Gaussian-family baseline is intentionally included, rather than assuming neural estimators must win.",
        ),
    ],
)
write(
    "10_information_bottleneck",
    "Relevance-preserving representations",
    "Discrete bottleneck optimization exposes the tension between information retained about inputs and information retained about relevant labels. Multiple starts address local solutions without pretending to certify a global optimum.",
    [
        (
            "code",
            """from information_lab.bottleneck import multistart
pyx=np.array([.05,.15,.2,.35,.65,.8,.85,.95]);joint=np.c_[1-pyx,pyx]/8
fit=multistart(joint,3,4,[41,141,241,341,441])
selected=fit['selected']
print('I(X;Z)',selected['i_input_bits'],'I(Y;Z)',selected['i_label_bits'],'objective',selected['objective_bits'])
print('Restart objectives',[r['objective_bits'] for r in fit['restarts']])""",
        ),
        (
            "code",
            """plt.imshow(selected['conditional'],aspect='auto',vmin=0,cmap='Blues');plt.colorbar(label='Probability');plt.xlabel('Representation state Z');plt.ylabel('Input state X');plt.show()""",
        ),
        (
            "markdown",
            "The self-consistent objective should decrease, and relevance cannot exceed the source-label MI by data processing. A low residual establishes a stationary local solution, not the best possible representation.",
        ),
        (
            "code",
            """assert np.max(np.diff(selected['trace'][:,2]))<=1e-8
record,path=query('bottleneck',clusters=3,beta=4)[0]
measured(record);preview(record)""",
        ),
        (
            "code",
            """display(Image(filename=str(ROOT/'assets/previews/bottleneck_comparison.png')))""",
        ),
    ],
)
write(
    "11_actual_image_compression",
    "Actual image bytes: traditional and learned codecs",
    "Four attributed photographs are compressed and decoded by JPEG, WebP, AVIF, JPEG2000, and four official pretrained learned architectures. Actual container size and precise displayed RGB8 reconstructions ground the comparison.",
    [
        (
            "code",
            """from PIL import Image as PILImage
from information_lab.images import encode, decode, metrics
original=np.array(PILImage.open(ROOT/'assets/images/originals/coffee.png').convert('RGB'))
encoded,details=encode(original,'jpeg',60,ROOT/'.local/unused-image-cache')
restored=decode(encoded,'jpeg',ROOT/'.local/unused-image-cache')
print('Actual JPEG bytes',len(encoded),'complete bpp',details['container_bpp'],metrics(original,restored))""",
        ),
        (
            "markdown",
            "The learned examples use official pretrained weights, not new training on four pictures. Shared model weights are external and excluded from per-image bpp. Learned likelihood estimates, entropy payload and complete container costs are separate. Numeric quality points across codec families are not equivalent.",
        ),
        (
            "code",
            """record,path=query('images',image='coffee',method='cheng2020-anchor',point=3)[0]
measured(record)
print('Original and actual decoded reconstruction:')
display(Image(filename=str(ROOT/'assets/images/originals/coffee.png')))
display(Image(filename=str(path/'decoded.png')))""",
        ),
        (
            "code",
            """data=arrays(path);shown=np.array(PILImage.open(path/'decoded.png').convert('RGB'))
assert np.array_equal(shown,data['decoded'])
assert np.isclose(record['metrics']['container_bpp'],8*(path/'image.bin').stat().st_size/np.prod(shown.shape[:2]))
print('Gallery pixels exactly match saved numerical pixels and whole-file rate.')""",
        ),
        ("code", """display(Image(filename=str(ROOT/'assets/previews/image_comparison.png')))"""),
        (
            "markdown",
            "This small resized set demonstrates practical rate/quality tradeoffs; it does not establish a standard benchmark or current SOTA. The reference audit re-decodes all 96 stored streams using the cached official checkpoints. The notebook itself is offline: JPEG is encoded locally and learned reconstructions are inspected from the measured evidence.",
        ),
    ],
)
(ROOT / "notebooks/README.md").write_text(
    "# Executed learning notebooks\n\nRead the notebooks in numerical order or choose a topic. Each combines original English explanations, small computations and actual measured reference evidence. Install the notebook, neural and image extras and run `python scripts/execute_notebooks.py` to execute all notebooks with the current Python interpreter. Full training and image-model downloads use the separate explicit CLI protocol; opening a notebook does not start hidden training or fetch weights.\n"
)
