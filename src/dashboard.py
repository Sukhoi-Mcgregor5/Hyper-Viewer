import os, json, pathlib
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

st.set_page_config(page_title="Satyadrishti Dashboard", layout="wide")

st.title("🔭 Satyadrishti Dashboard")

# --- Sidebar ---
folder = st.sidebar.selectbox(
    "Choose folder to explore:",
    ["data/indian_pines_out", "data/indian_pines/frames", "data/out", "data/demo"],
    index=0
)
root = pathlib.Path(folder)

def list_images(p: pathlib.Path):
    if not p.exists():
        return []
    return sorted([f.name for f in p.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])

# --- Quick file browser (as before) ---
files = list_images(root)
if files:
    sel = st.selectbox("Select an image:", files, index=0)
    try:
        st.image(Image.open(root / sel), caption=sel, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not open {sel}: {e}")
else:
    st.info(f"No images found in {folder}")

st.markdown("---")

# === Analyst view: side-by-side + overlay (if the structure looks like a run) ===
# Expect frames in data/<name>/frames and outputs in data/<name>_out
def try_analyst_view():
    # If user is in ".../frames", derive the run name
    if folder.endswith("/frames"):
        base = pathlib.Path(folder).parent
        run_out = base.with_name(base.name + "_out")
    elif folder.endswith("_out"):
        run_out = pathlib.Path(folder)
        base = run_out.with_name(run_out.name.replace("_out", ""))
    else:
        return False

    f0 = base / "frames" / "frame_00.png"
    f1 = base / "frames" / "frame_01.png"
    det = run_out / "detections.png"
    meta = run_out / "meta.json"
    if not (f0.exists() and f1.exists() and det.exists()):
        return False

    st.subheader("Analyst View (frames + detections overlay)")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.caption("Frame 00 (before)")
        st.image(Image.open(f0), use_container_width=True)

    with col2:
        st.caption("Frame 01 (after)")
        st.image(Image.open(f1), use_container_width=True)

       # Overlay
    st.caption("Overlay detections on Frame 01")
    # discover available detection files
    det_files = []
    for tag in ["detections_rx.png", "detections_simple.png", "detections.png"]:
        p = run_out / tag
        if p.exists():
            det_files.append(tag)
    if not det_files:
        st.warning("No detection overlays found.")
        return True

    algo_choice = st.selectbox("Choose detections:", det_files, index=0)
    alpha = st.slider("Detection opacity", 0.0, 1.0, 0.5, 0.05)

    base_img = Image.open(f1).convert("RGBA")
    det_img = Image.open(run_out / algo_choice).convert("L")
    det_rgba = ImageOps.colorize(det_img, black=(0,0,0), white=(255,0,0)).convert("RGBA")

    arr = np.array(det_rgba)
    arr[..., 3] = (alpha * 255 * (arr[...,0] > 0)).astype(np.uint8)
    det_rgba = Image.fromarray(arr, mode="RGBA")

    over = Image.alpha_composite(base_img, det_rgba)
    st.image(over, use_container_width=True)

    # Pick matching meta
    meta_guess = "meta_rx.json" if "rx" in algo_choice else ("meta_simple.json" if "simple" in algo_choice else "meta.json")
    meta_path = run_out / meta_guess
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text())
            st.success(
                f"Algo: **{m.get('algo','?')}** • "
                f"Confidence: **{m.get('final_confidence','—')}** "
                f"(base={m.get('base_confidence','—')}, weather_penalty={m.get('weather_penalty','—')})"
            )
        except Exception as e:
            st.warning(f"Could not read {meta_guess}: {e}")