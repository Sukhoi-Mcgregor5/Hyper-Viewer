# src/pipeline.py
import argparse, json, pathlib
import numpy as np
from PIL import Image
from src.models.rx import rx_score  # <-- new

def save_heatmap_png(arr01: np.ndarray, out_path: pathlib.Path):
    # arr01: (H,W) in [0,1] -> grayscale PNG
    img = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)

def certainty_penalty(weather_cur: dict) -> float:
    """Return [0..1] penalty (0 = worst weather, 1 = clear)."""
    tcc = float(weather_cur.get("tcc", 0.0))         # total cloud cover 0..1 (stub)
    precip = float(weather_cur.get("precip", 0.0))   # mm/hr (stub)
    # simple heuristic: more clouds/precip => lower certainty
    pen = (1.0 - 0.5*tcc) * (1.0 - min(1.0, 0.2*precip))
    return float(np.clip(pen, 0.0, 1.0))

def simple_change(prev_bhw: np.ndarray, curr_bhw: np.ndarray) -> np.ndarray:
    """Basic band-mean absolute diff, normalized to [0,1]."""
    prev_m = prev_bhw.mean(axis=0)  # (H,W)
    curr_m = curr_bhw.mean(axis=0)
    change = np.abs(curr_m - prev_m)
    # robust normalize for display
    lo, hi = np.percentile(change, 1), np.percentile(change, 99)
    if hi > lo:
        change = (change - lo) / (hi - lo)
    return np.clip(change, 0, 1).astype(np.float32)

def to_mask(score_hw: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    thr = float(np.quantile(score_hw, quantile))
    mask = (score_hw >= thr).astype(np.uint8)
    return mask, thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--weather_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--algo", choices=["simple", "rx", "both"], default="both")
    ap.add_argument("--q", type=float, default=0.99, help="Quantile for threshold")
    args = ap.parse_args()

    frames = pathlib.Path(args.frames_dir)
    weather = pathlib.Path(args.weather_dir)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load two frames
    prev = np.load(frames / "frame_00.npy")   # (B,H,W) float32
    curr = np.load(frames / "frame_01.npy")

    # Load weather (use frame_01 for 'current')
    w_cur = json.loads((weather / "frame_01.json").read_text())

    def run_algo(name: str, score_hw: np.ndarray):
        mask, thr = to_mask(score_hw, quantile=args.q)
        base_conf = float(np.clip((score_hw.max() - score_hw.mean()) /
                                  (score_hw.std() + 1e-6), 0, 1))
        weather_c = certainty_penalty(w_cur)
        final_conf = float(np.clip(base_conf * weather_c, 0, 1))

        # save
        save_heatmap_png(score_hw, out / f"change_{name}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"detections_{name}.png")
        (out / f"meta_{name}.json").write_text(json.dumps({
            "frames": ["frame_00.npy", "frame_01.npy"],
            "algo": name,
            "threshold": thr,
            "base_confidence": base_conf,
            "weather_penalty": weather_c,
            "final_confidence": final_conf,
            "weather_curr": w_cur
        }, indent=2))
        return final_conf

    finals = {}

    if args.algo in ["simple", "both"]:
        score_simple = simple_change(prev, curr)              # (H,W)
        finals["simple"] = run_algo("simple", score_simple)

    if args.algo in ["rx", "both"]:
        # run RX on the 'after' cube (curr)
        score_rx = rx_score(curr)                             # (H,W)
        finals["rx"] = run_algo("rx", score_rx)

    # For backward compatibility, copy the last-run algo to generic filenames
    # (so your old dashboard still works if needed)
    if finals:
        # pick the higher-confidence result to be the generic
        best_algo = max(finals, key=lambda k: finals[k])
        for src, dst in [
            (f"change_{best_algo}.png", "change.png"),
            (f"detections_{best_algo}.png", "detections.png"),
            (f"meta_{best_algo}.json", "meta.json"),
        ]:
            data = (out / src).read_bytes()
            (out / dst).write_bytes(data)

    print("Done. Final confidences:", finals)
    print(f"Outputs in {out}")

if __name__ == "__main__":
    main()