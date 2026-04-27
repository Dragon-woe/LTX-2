"""FID comparison between baseline and multi-card generated videos.

Usage:
    python scripts/fid_compare.py \
        --baseline output_1card.mp4 \
        --compare output_2card.mp4 output_4card.mp4 output_8card.mp4
"""

import argparse
import sys

import numpy as np
import torch
from scipy import linalg


def extract_frames(video_path: str, max_frames: int = 64) -> list[np.ndarray]:
    """Extract frames from a video file as numpy arrays (H, W, 3) uint8."""
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python is required: pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def frames_to_tensor(frames: list[np.ndarray], size: int = 299) -> torch.Tensor:
    """Resize frames and convert to (N, 3, size, size) float tensor."""
    import cv2
    tensors = []
    for f in frames:
        resized = cv2.resize(f, (size, size))
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensors.append(t)
    return torch.stack(tensors)


def get_inception_features(frames: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """Extract InceptionV3 features from a batch of frames."""
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    frames = frames.to(device)
    with torch.no_grad():
        features = model(frames)
    return features.cpu().numpy()


def compute_fid(feats1: np.ndarray, feats2: np.ndarray) -> float:
    """Compute Fréchet Inception Distance between two sets of features."""
    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def main():
    parser = argparse.ArgumentParser(description="FID comparison between videos")
    parser.add_argument("--baseline", required=True, help="Baseline video (single-card)")
    parser.add_argument("--compare", nargs="+", required=True, help="Videos to compare")
    parser.add_argument("--device", default="cpu", help="Device for Inception (cpu/cuda/npu)")
    parser.add_argument("--max-frames", type=int, default=64, help="Max frames to extract")
    args = parser.parse_args()

    print(f"Baseline: {args.baseline}")
    baseline_frames = extract_frames(args.baseline, args.max_frames)
    if len(baseline_frames) < 2:
        print(f"[ERROR] Baseline has {len(baseline_frames)} frames, need at least 2", file=sys.stderr)
        sys.exit(1)

    baseline_tensor = frames_to_tensor(baseline_frames)
    baseline_feats = get_inception_features(baseline_tensor, args.device)

    print(f"\n{'Video':<40} {'Frames':<8} {'FID':<12} {'Status'}")
    print("-" * 72)

    for video_path in args.compare:
        frames = extract_frames(video_path, args.max_frames)
        if len(frames) < 2:
            print(f"{video_path:<40} {len(frames):<8} {'N/A':<12} SKIP (too few frames)")
            continue

        tensor = frames_to_tensor(frames)
        feats = get_inception_features(tensor, args.device)
        fid = compute_fid(baseline_feats, feats)

        status = "OK" if fid < 500 else "WARNING"
        if fid > 1000:
            status = "FAIL"
        print(f"{video_path:<40} {len(frames):<8} {fid:<12.2f} {status}")


if __name__ == "__main__":
    main()
