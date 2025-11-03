import argparse, json, pathlib
import numpy as np
from PIL import Image
from tqdm import trange

def generate_hsi_frame(shape=(256,256), bands=32, hot_line=False, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.2, scale=0.03, size=(bands, *shape)).astype(np.float32)
    wl_trend = np.linspace(0, 1, bands, dtype=np.float32)[:, None, None]
    base += 0.05 * wl_trend
    y0 = shape[0]//2
    for b in range(bands):
        base[b, y0-2:y0+2, 30:shape[1]-30] += 0.02
    if hot_line:
        thermal = slice(int(0.6*bands), bands)
        base[thermal, y0-2:y0+2, 30:shape[1]-30] += 0.12
    return np.clip(base, 0, 1)

def save_quicklook(arr, path_png):
    b = arr.shape[0]
    r, g, bl = 2, b//2, b-1
    rgb = np.stack([arr[r], arr[g], arr[bl]], axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bands", type=int, default=32)
    ap.add_argument("--shape", nargs=2, type=int, default=[256,256])
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "weather").mkdir(parents=True, exist_ok=True)

    for t in trange(args.frames, desc="Generating frames"):
        arr = generate_hsi_frame(tuple(args.shape), args.bands, hot_line=(t==args.frames-1), seed=42+t)
        np.save(out / "frames" / f"frame_{t:02d}.npy", arr)
        save_quicklook(arr, out / "frames" / f"frame_{t:02d}.png")
        w = {"tcc": 0.1 + 0.1*t, "precip": 0.0, "temp2m": 40.0, "rh2m": 0.35, "pressure": 1008.0}
        (out / "weather" / f"frame_{t:02d}.json").write_text(json.dumps(w))
    print(f"Saved synthetic series to: {out}")

if __name__ == "__main__":
    main()
