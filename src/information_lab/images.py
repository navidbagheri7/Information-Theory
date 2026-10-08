"""Natural-image compression with real portable streams and decoded-pixel metrics."""

import hashlib
import io
import json
import os
import re
import struct
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

LEARNED = ["bmshj2018-factorized", "bmshj2018-hyperprior", "mbt2018-mean", "cheng2020-anchor"]
IMAGE_CREDITS = {
    "astronaut": "NASA, Eileen Collins photograph; public domain/no known restrictions",
    "coffee": "Rachel Michetti, Pikolo Espresso Bar photograph; CC0",
    "chelsea": "Stefan van der Walt, Chelsea photograph; CC0",
    "rocket": "SpaceX, launch photograph; public domain",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare_images(data_root, max_side=384):
    """Four explicitly credited demonstrations, not a standard large benchmark."""
    data_root = Path(data_root)
    os.environ["SKIMAGE_DATADIR"] = str(data_root / "skimage_cache")
    from skimage import data

    out = data_root / "images"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, credit in IMAGE_CREDITS.items():
        array = getattr(data, name)()
        image = Image.fromarray(array).convert("RGB")
        original_size = image.size
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                tuple(round(v * scale) for v in image.size), Image.Resampling.LANCZOS
            )
        path = out / (name + ".png")
        image.save(path)
        records.append(
            {
                "name": name,
                "credit": credit,
                "file": path.name,
                "file_sha256": digest(path.read_bytes()),
                "pixels_sha256": digest(np.array(image).tobytes()),
                "size": list(image.size),
                "original_size": list(original_size),
                "source": "https://scikit-image.org/docs/stable/api/skimage.data",
                "preprocessing": "RGB8; preserve aspect ratio, cap maximum side using Lanczos",
            }
        )
    (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


@lru_cache(maxsize=4)
def load_model(architecture, quality, data_root, device="cpu", allow_download=False):
    if architecture not in LEARNED:
        raise ValueError("Unknown learned codec")
    os.environ["TORCH_HOME"] = str(Path(data_root) / "models")
    import compressai.zoo as zoo
    import torch
    from compressai.zoo.image import model_urls

    torch.set_num_threads(1)
    url = model_urls[architecture]["mse"][quality]
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(url).path).name
    if not path.exists() and not allow_download:
        raise FileNotFoundError(
            "Prepare the explicitly requested pretrained models before offline inference: "
            + str(path)
        )
    expected = re.search(r"-([0-9a-f]{8})\.pth", path.name)
    if path.exists() and expected and not digest(path.read_bytes()).startswith(expected.group(1)):
        raise ValueError("Official checkpoint hash prefix mismatch")
    model = (
        getattr(zoo, architecture.replace("-", "_"))(quality=quality, metric="mse", pretrained=True)
        .eval()
        .to(device)
    )
    model.update()
    sha = digest(path.read_bytes())
    if expected and not sha.startswith(expected.group(1)):
        raise ValueError("Downloaded checkpoint hash prefix mismatch")
    return model, {
        "architecture": architecture,
        "quality": quality,
        "metric": "mse",
        "url": url,
        "checkpoint_sha256": sha,
    }


def pack_stream(strings, metadata):
    metadata = {**metadata, "stream_groups": [len(group) for group in strings]}
    description = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    out = bytearray(b"ITIC1" + struct.pack(">I", len(description)) + description)
    for group in strings:
        for stream in group:
            out.extend(struct.pack(">I", len(stream)))
            out.extend(stream)
    return bytes(out)


def unpack_stream(encoded):
    if encoded[:5] != b"ITIC1" or len(encoded) < 9:
        raise ValueError("Invalid learned-image container")
    size = struct.unpack_from(">I", encoded, 5)[0]
    metadata = json.loads(encoded[9 : 9 + size])
    position, strings = 9 + size, []
    for count in metadata["stream_groups"]:
        group = []
        for _ in range(count):
            if position + 4 > len(encoded):
                raise ValueError("Truncated stream header")
            length = struct.unpack_from(">I", encoded, position)[0]
            position += 4
            if position + length > len(encoded):
                raise ValueError("Truncated entropy stream")
            group.append(encoded[position : position + length])
            position += length
        strings.append(group)
    if position != len(encoded):
        raise ValueError("Unexpected trailing container data")
    return strings, metadata


def _sync(device):
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def encode(array, method, point, data_root, device="cpu"):
    array = np.asarray(array)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[2] != 3
        or min(array.shape[:2]) < 1
    ):
        raise ValueError("RGB8 image required")
    h, w = array.shape[:2]
    extra = {}
    if method in LEARNED:
        import torch
        from torch.nn import functional as functional

        model, model_record = load_model(method, int(point), str(data_root), device)
        x = (
            torch.tensor(array.transpose(2, 0, 1).copy(), dtype=torch.float32, device=device)[None]
            / 255
        )
        padded = functional.pad(x, (0, (-w) % 64, 0, (-h) % 64))
        with torch.no_grad():
            likelihood = model(padded)["likelihoods"]
            estimated = sum(float(-v.log2().sum().cpu()) for v in likelihood.values()) / (h * w)
            _sync(device)
            started = time.perf_counter()
            result = model.compress(padded)
            _sync(device)
            seconds = time.perf_counter() - started
        metadata = {
            **model_record,
            "image_shape": [h, w],
            "latent_shape": list(result["shape"]),
            "padded_shape": list(padded.shape[-2:]),
        }
        encoded = pack_stream(result["strings"], metadata)
        extra = {
            "estimated_bpp": estimated,
            "payload_bpp": sum(len(s) * 8 for group in result["strings"] for s in group) / (h * w),
            "model": model_record,
        }
    else:
        output = io.BytesIO()
        image = Image.fromarray(array)
        started = time.perf_counter()
        if method == "jpeg":
            image.save(output, format="JPEG", quality=int(point), subsampling=0, optimize=True)
        elif method == "webp":
            image.save(output, format="WEBP", quality=int(point), method=6, lossless=False)
        elif method == "avif":
            image.save(output, format="AVIF", quality=int(point), speed=6, subsampling="4:4:4")
        elif method == "jpeg2000":
            image.save(
                output,
                format="JPEG2000",
                quality_mode="rates",
                quality_layers=[float(point)],
                irreversible=True,
            )
        else:
            raise ValueError("Unknown image codec")
        seconds = time.perf_counter() - started
        encoded = output.getvalue()
        extra = {"payload_bpp": None, "estimated_bpp": None}
    return encoded, {
        **extra,
        "encode_seconds": seconds,
        "container_bpp": len(encoded) * 8 / (h * w),
    }


def decode(encoded, method, data_root, device="cpu"):
    if method not in LEARNED:
        with Image.open(io.BytesIO(encoded)) as image:
            return np.array(image.convert("RGB"))
    import torch

    strings, metadata = unpack_stream(encoded)
    if metadata["architecture"] != method:
        raise ValueError("Container architecture and requested decoder disagree")
    model, record = load_model(method, metadata["quality"], str(data_root), device)
    if record["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("Decoder checkpoint differs from the encoded model")
    with torch.no_grad():
        result = model.decompress(strings, tuple(metadata["latent_shape"]))["x_hat"]
    h, w = metadata["image_shape"]
    return (
        result[0, :, :h, :w]
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def metrics(original, restored):
    from skimage.metrics import structural_similarity

    original, restored = np.asarray(original), np.asarray(restored)
    if original.shape != restored.shape:
        raise ValueError("Original and decoded image dimensions disagree")
    mse = float(np.mean((original.astype(float) - restored.astype(float)) ** 2))
    result = {
        "mse_rgb8": mse,
        "psnr_db": float(10 * np.log10(255**2 / mse)) if mse > 0 else None,
        "lossless_pixels": mse == 0,
    }
    result["ssim_rgb"] = float(
        structural_similarity(original, restored, channel_axis=2, data_range=255)
    )
    if min(original.shape[:2]) > 160:
        import torch
        from pytorch_msssim import ms_ssim

        a = torch.tensor(original.transpose(2, 0, 1).copy(), dtype=torch.float32)[None] / 255
        b = torch.tensor(restored.transpose(2, 0, 1).copy(), dtype=torch.float32)[None] / 255
        result["ms_ssim_rgb"] = float(ms_ssim(a, b, data_range=1).item())
    else:
        result["ms_ssim_rgb"] = None
    return result


def experiment(settings, data_root, output, device="cpu"):
    if settings.get("timing_repeats", 3) < 1:
        raise ValueError("At least one measured encode/decode repetition required")
    path = Path(data_root) / "images" / (settings["image"] + ".png")
    original = np.array(Image.open(path).convert("RGB"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times = []
    reference = None
    for _ in range(settings.get("timing_repeats", 3)):
        encoded, details = encode(
            original, settings["method"], settings["point"], data_root, device
        )
        _sync(device)
        started = time.perf_counter()
        restored = decode(encoded, settings["method"], data_root, device)
        _sync(device)
        decode_seconds = time.perf_counter() - started
        if reference is not None and (
            encoded != reference[0] or not np.array_equal(restored, reference[1])
        ):
            raise RuntimeError("Codec repetitions are not byte/pixel reproducible")
        reference = (encoded, restored)
        times.append([details["encode_seconds"], decode_seconds])
    stream_path = output / "image.bin"
    stream_path.write_bytes(encoded)
    Image.fromarray(restored).save(output / "decoded.png")
    details.update(metrics(original, restored))
    details.update(
        {
            "encode_seconds": float(np.median(np.array(times)[:, 0])),
            "decode_seconds": float(np.median(np.array(times)[:, 1])),
            "encoded_bytes": len(encoded),
            "encoded_sha256": digest(encoded),
            "source_sha256": digest(original.tobytes()),
            "decoded_sha256": digest(restored.tobytes()),
            "image_shape": list(original.shape),
            "device": device,
            "scope": "Actual independently decodable stream, complete container rate, metrics on the same RGB8 decoded pixels shown in the gallery. Shared pretrained model weights are external and not amortized into per-image rates. Four licensed resized demonstrations, not a standard benchmark/SOTA claim.",
        }
    )
    return details, {"original": original, "decoded": restored, "timings": np.array(times)}
