"""
CrimeScene - Visual Counterfactual Intelligence (VCI)
=====================================================
Academic research prototype (single file).

Pipeline (all numbers come from real CLIP inference):
  video / image folder -> OpenCV frame extraction -> CLIP image embeddings
  -> cosine similarity with CLIP text embeddings of candidate descriptions
  -> grid masking of ONE selected frame -> CLIP again -> score differences.

CLIP similarity scores are NOT crime probabilities. Masking results show model
sensitivity to image regions; they do not prove causation.

Run:  streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CrimeScene - VCI", page_icon="🎥", layout="wide")

# ----------------------------------------------------------------------------
# Dependency check (shows a clear message instead of a raw traceback)
# ----------------------------------------------------------------------------
try:
    import cv2
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
except ImportError as _e:
    st.error(
        f"Missing Python dependency: **{_e.name}**.\n\n"
        "Activate your virtual environment and run:\n\n"
        "`pip install -r requirements.txt`"
    )
    st.stop()

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
VIDEO_EXTS = {".mp4", ".avi", ".mov"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_DISPLAY_SIDE = 640      # frames are downscaled to this (aspect ratio kept)
MIN_DISPLAY_SIDE = 256      # tiny frames (e.g. 64x64 PNGs) are upscaled for display
PAD_COLOR = 127             # neutral grey used to make frames square for CLIP
MAX_SCAN_FILES = 50000

SRC_UPLOAD = "Upload a video"
SRC_DATASET = "Video from a local dataset folder"
SRC_IMAGES = "Folder of extracted frames (Kaggle PNG copy)"

DEFAULT_PROMPTS = """a person walking normally
people standing together
a person running
a physical fight between people
a person stealing an object
a person breaking into a building
a person carrying a bag
a person falling to the ground
a group of people gathering
a vehicle moving on a road
a person vandalizing property
a person pointing an object at another person
a person entering a building
a person leaving a building
an empty surveillance scene"""

DISCLAIMER = (
    "Experimental visual analysis tool. CLIP similarity scores are not calibrated "
    "probabilities, and a high score for a description does not establish that the "
    "event occurred. Masking results describe model sensitivity only, not causation. "
    "No accuracy metrics are computed. Not to be used as the sole basis for any "
    "real-world law-enforcement decision."
)


class AppError(Exception):
    """Errors with a message that is safe and useful to show to the user."""


# ----------------------------------------------------------------------------
# Styling (dark surveillance-analysis look)
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.stApp { background-color: #15171a; color: #e4e6ea; }
[data-testid="stSidebar"] { background-color: #1c1f23; border-right: 1px solid #2b2f35; }
h1, h2, h3 { color: #f0f1f3; letter-spacing: 0.3px; }
h2 { border-left: 4px solid #c0392b; padding-left: 10px; margin-top: 1.6rem; }
[data-testid="stMetric"] { background: #22252b; border: 1px solid #2f343b;
    border-radius: 6px; padding: 10px 14px; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #1d2025; }
.stButton > button[kind="primary"], .stDownloadButton > button {
    background-color: #b03a2e; color: #fff; border: 0; }
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { background-color: #c0392b; color:#fff; }
.small-note { color: #9aa0a8; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def fmt_ts(ts):
    if ts is None:
        return "n/a"
    return f"{int(ts // 60):02d}:{ts % 60:05.2f}"


def short(text, n=42):
    return text if len(text) <= n else text[: n - 1] + "…"


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def parse_prompts(text):
    seen, out = set(), []
    for line in text.splitlines():
        line = line.strip()
        if line and line.lower() not in seen:
            seen.add(line.lower())
            out.append(line)
    if not out:
        raise AppError("The candidate description list is empty. Add at least one description (one per line).")
    if len(out) > 40:
        raise AppError("Please use at most 40 candidate descriptions to keep CPU analysis responsive.")
    return out


def region_name(r, c):
    return f"row {r + 1}, col {c + 1}"


def region_bounds(h, w, r, c, rows, cols):
    return r * h // rows, (r + 1) * h // rows, c * w // cols, (c + 1) * w // cols


# ----------------------------------------------------------------------------
# CLIP loading and inference
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pretrained CLIP (the first run downloads about 600 MB)...")
def load_clip():
    device, note = "cpu", ""
    if torch.cuda.is_available():
        try:  # make sure CUDA really works, not just "is available"
            _ = torch.zeros(1, device="cuda") + 1
            torch.cuda.synchronize()
            device = "cuda"
        except Exception as e:  # noqa: BLE001
            note = f"CUDA was detected but is not usable ({e}). Using CPU instead."
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # frozen: no training
    model.to(device)
    return model, processor, device, note


def load_clip_safe():
    try:
        return load_clip()
    except Exception as e:  # noqa: BLE001
        raise AppError(
            f"CLIP could not be loaded ({type(e).__name__}: {e}).\n\n"
            "Common fixes: (1) connect to the internet for the first-time download of "
            f"`{MODEL_NAME}`; (2) free at least 2 GB of disk space; (3) check that "
            "`torch` and `transformers` installed correctly (`pip install -r requirements.txt`). "
            "The analysis was stopped and no results were produced."
        ) from e


def _features(out):
    if isinstance(out, torch.Tensor):
        return out
    for key in ("image_embeds", "text_embeds"):
        v = getattr(out, key, None)
        if v is not None:
            return v
    raise AppError("Unexpected CLIP output type. Install transformers>=4.40,<5 (see requirements.txt).")


def pad_square(rgb):
    """Pad to a square with neutral grey so CLIP sees the WHOLE frame (no centre-crop,
    no distortion). Masked regions near the edges therefore really reach the model."""
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), PAD_COLOR, dtype=np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb
    return canvas


def embed_texts(model, processor, device, prompts):
    try:
        with torch.inference_mode():
            inp = processor(text=list(prompts), return_tensors="pt", padding=True,
                            truncation=True, max_length=77)
            inp = {k: v.to(device) for k, v in inp.items()}
            f = _features(model.get_text_features(**inp))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()
    except (RuntimeError, MemoryError) as e:
        raise AppError(f"CLIP text encoding failed ({e}). Try fewer descriptions or close other programs.") from e


@st.cache_data(show_spinner=False)
def cached_text_embeddings(prompts, _model, _processor, _device):
    return embed_texts(_model, _processor, _device, prompts)


def embed_images(model, processor, device, images_rgb, batch=8, progress=None):
    outs = []
    try:
        with torch.inference_mode():
            for i in range(0, len(images_rgb), batch):
                pil = [Image.fromarray(pad_square(a)) for a in images_rgb[i:i + batch]]
                px = processor(images=pil, return_tensors="pt")["pixel_values"].to(device)
                f = _features(model.get_image_features(pixel_values=px))
                f = f / f.norm(dim=-1, keepdim=True)
                outs.append(f.cpu().numpy())
                if progress:
                    progress(min(i + batch, len(images_rgb)) / len(images_rgb))
    except (RuntimeError, MemoryError) as e:
        raise AppError(
            f"CLIP image encoding failed ({e}). This is often low memory: reduce the number of "
            "frames or the grid size, or close other programs."
        ) from e
    return np.concatenate(outs, axis=0)


# ----------------------------------------------------------------------------
# Frame extraction (OpenCV)
# ----------------------------------------------------------------------------
def resize_keep_aspect(rgb):
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m > MAX_DISPLAY_SIDE:
        s = MAX_DISPLAY_SIDE / m
        return cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    if m < MIN_DISPLAY_SIDE:
        s = MIN_DISPLAY_SIDE / m
        return cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return rgb


def sample_positions(total, n):
    """n positions spread evenly across [0, total), at the centre of each segment."""
    t = ((np.arange(n) + 0.5) / n * total).astype(int)
    return sorted(set(np.clip(t, 0, total - 1).tolist()))


def read_video_frames(path, n_frames, display_name):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise AppError(
            f"OpenCV could not open '{display_name}'. The file may be corrupted or use a codec "
            "OpenCV cannot decode. Try re-encoding it to H.264 .mp4 (e.g. with HandBrake or ffmpeg)."
        )
    raw = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if total > 0:  # fast path: seek to the target frames
            for t in sample_positions(total, n_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                ok, bgr = cap.read()
                if ok and bgr is not None:
                    raw.append((t, bgr))
        if not raw:  # fallback: frame count unknown or seeking failed -> count, then sequential read
            cap.release()
            cap = cv2.VideoCapture(path)
            total = 0
            while cap.grab():
                total += 1
            if total == 0:
                raise AppError(f"'{display_name}' has no readable frames.")
            cap.release()
            cap = cv2.VideoCapture(path)
            wanted = set(sample_positions(total, n_frames))
            for i in range(total):
                if not cap.grab():
                    break
                if i in wanted:
                    ok, bgr = cap.retrieve()
                    if ok and bgr is not None:
                        raw.append((i, bgr))
    finally:
        cap.release()
    if not raw:
        raise AppError(f"Frames could not be decoded from '{display_name}'. The video may be corrupted.")
    if not width or not height:
        height, width = raw[0][1].shape[:2]
    frames = []
    for idx, bgr in raw:
        rgb = resize_keep_aspect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frames.append({"idx": idx, "label": f"#{idx}", "ts": (idx / fps) if fps else None, "rgb": rgb})
    info = {"kind": "video", "name": display_name, "fps": fps or None,
            "total": total, "duration": (total / fps) if (fps and total) else None,
            "width": width, "height": height}
    return frames, info


def read_image_frames(paths, n_images, folder_name):
    frames, w0, h0 = [], 0, 0
    for pos in sample_positions(len(paths), n_images):
        p = paths[pos]
        try:  # np.fromfile + imdecode also works with non-ASCII Windows paths
            bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            bgr = None
        if bgr is None:
            continue
        if not w0:
            h0, w0 = bgr.shape[:2]
        rgb = resize_keep_aspect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frames.append({"idx": pos, "label": os.path.basename(p), "ts": None, "rgb": rgb})
    if not frames:
        raise AppError("None of the selected image files could be decoded.")
    info = {"kind": "images", "name": folder_name, "fps": None, "total": len(paths),
            "duration": None, "width": w0, "height": h0}
    return frames, info


# ----------------------------------------------------------------------------
# File discovery
# ----------------------------------------------------------------------------
def scan_files(root, exts):
    root = root.strip().strip('"').strip("'")
    if not root:
        raise AppError("Enter a folder path first.")
    p = Path(root).expanduser()
    if not p.exists():
        raise AppError(f"Folder not found: {p}")
    if not p.is_dir():
        raise AppError(f"This is a file, not a folder: {p}")
    found, capped = [], False
    for dp, _, fns in os.walk(p):
        for fn in fns:
            if Path(fn).suffix.lower() in exts:
                found.append(os.path.join(dp, fn))
                if len(found) >= MAX_SCAN_FILES:
                    capped = True
                    break
        if capped:
            break
    return sorted(found), capped, str(p)


# ----------------------------------------------------------------------------
# Counterfactual masking
# ----------------------------------------------------------------------------
def mask_region(rgb, r, c, rows, cols, value):
    h, w = rgb.shape[:2]
    y0, y1, x0, x1 = region_bounds(h, w, r, c, rows, cols)
    out = rgb.copy()
    out[y0:y1, x0:x1] = value
    return out


def run_counterfactual(model, processor, device, A, fpos, rows, cols, color_name, progress=None):
    value = 127 if color_name == "Neutral grey" else 0
    rgb = A["frames"][fpos]["rgb"]
    masked, labels = [], []
    for r in range(rows):
        for c in range(cols):
            masked.append(mask_region(rgb, r, c, rows, cols, value))
            labels.append(region_name(r, c))
    emb = embed_images(model, processor, device, masked, batch=8, progress=progress)
    st.session_state["cf_counter"] = st.session_state.get("cf_counter", 0) + 1
    return {
        "id": st.session_state["cf_counter"], "fpos": fpos, "rows": rows, "cols": cols,
        "color": color_name, "labels": labels, "masked_imgs": masked,
        "orig_sims": A["sims"][fpos],            # same frame, same model: no duplicate inference
        "masked_sims": emb @ A["txt_emb"].T,     # shape (regions, descriptions)
    }


def heat_overlay(rgb, drops):
    rows, cols = drops.shape
    h, w = rgb.shape[:2]
    out = rgb.astype(np.float32).copy()
    m = max(float(np.abs(drops).max()), 1e-9)
    for r in range(rows):
        for c in range(cols):
            y0, y1, x0, x1 = region_bounds(h, w, r, c, rows, cols)
            v = drops[r, c]
            a = 0.65 * abs(v) / m
            col = np.array([220, 40, 40] if v > 0 else [50, 110, 230], dtype=np.float32)
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + col * a
    out = out.astype(np.uint8)
    fs = max(0.35, w / 1500)
    for r in range(rows):
        for c in range(cols):
            y0, y1, x0, x1 = region_bounds(h, w, r, c, rows, cols)
            cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
            cv2.putText(out, f"{drops[r, c]:+.4f}", (x0 + 4, y0 + int(18 * fs / 0.4)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎥 CrimeScene")
    st.caption("Visual Counterfactual Intelligence (VCI)")
    st.caption("Academic research prototype · pretrained CLIP · no training")
    source = st.radio("Input source", [SRC_UPLOAD, SRC_DATASET, SRC_IMAGES])
    n_frames = st.slider("Frames to extract and analyse", 4, 24, 10,
                         help="Spread evenly across the video. More frames = slower on CPU.")
    prompt_text = st.text_area("Candidate descriptions (one per line, editable)",
                               DEFAULT_PROMPTS, height=330,
                               help="These are candidate descriptions, not guaranteed dataset classes.")
    st.markdown(f"<div class='small-note'>{DISCLAIMER}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Header + input selection
# ----------------------------------------------------------------------------
st.title("CrimeScene - Visual Counterfactual Intelligence")
st.caption("CLIP-based analysis of surveillance footage and a masking experiment that probes "
           "which image regions the model's scores are sensitive to.")

upload, chosen_path, image_paths, folder_label = None, None, None, ""

if source == SRC_UPLOAD:
    upload = st.file_uploader("Upload a surveillance video", type=["mp4", "avi", "mov"],
                              help="Streamlit's default upload limit is 200 MB. For larger files use the dataset-folder option.")
else:
    ext_set = VIDEO_EXTS if source == SRC_DATASET else IMAGE_EXTS
    default_hint = r"C:\Datasets\UCF-Crime" if source == SRC_DATASET else r"C:\Datasets\UCF-Crime\Train\Arson"
    root = st.text_input("Local folder path", placeholder=default_hint)
    if source == SRC_IMAGES:
        st.caption("The Kaggle copy (odins0n) contains small PNG frames, not videos. Point this at ONE class "
                   "folder (e.g. Train\\Arson). Frames are taken in file-name order; they may come from "
                   "different videos, so no timeline is implied.")
    if st.button("Scan folder"):
        try:
            files, capped, resolved = scan_files(root, ext_set)
            st.session_state["scan"] = {"source": source, "root": resolved, "files": files, "capped": capped}
        except AppError as e:
            st.session_state.pop("scan", None)
            st.error(str(e))
    scan = st.session_state.get("scan")
    if scan and scan["source"] == source:
        if not scan["files"]:
            kind = "video (.mp4/.avi/.mov)" if source == SRC_DATASET else "image (.png/.jpg/.bmp)"
            st.warning(f"No supported {kind} files were found under `{scan['root']}` (or its subfolders).")
        else:
            st.success(f"Found {len(scan['files'])} file(s) under `{scan['root']}`."
                       + (f" Scan stopped at {MAX_SCAN_FILES} files." if scan["capped"] else ""))
            if source == SRC_DATASET:
                chosen_path = st.selectbox(
                    "Select a video", scan["files"],
                    format_func=lambda p: os.path.relpath(p, scan["root"]))
            else:
                image_paths, folder_label = scan["files"], os.path.basename(scan["root"]) or scan["root"]

ready = bool(upload or chosen_path or image_paths)
run = st.button("▶ Analyse with CLIP", type="primary", disabled=not ready)

# ----------------------------------------------------------------------------
# Analysis (real frame extraction + real CLIP inference)
# ----------------------------------------------------------------------------
if run:
    st.session_state.pop("analysis", None)
    st.session_state.pop("cf", None)
    try:
        prompts = parse_prompts(prompt_text)
        model, processor, device, note = load_clip_safe()
        if note:
            st.warning(note)

        with st.spinner("Extracting frames with OpenCV..."):
            if upload is not None:
                suffix = Path(upload.name).suffix.lower() or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                try:
                    tmp.write(upload.getvalue())
                    tmp.close()
                    frames, info = read_video_frames(tmp.name, n_frames, upload.name)
                finally:
                    try:
                        os.remove(tmp.name)
                    except OSError:
                        pass
                gt_hint = None
            elif chosen_path:
                if not os.path.isfile(chosen_path):
                    raise AppError(f"Video file no longer exists: {chosen_path}")
                frames, info = read_video_frames(chosen_path, n_frames, os.path.basename(chosen_path))
                gt_hint = os.path.basename(os.path.dirname(chosen_path))
            else:
                frames, info = read_image_frames(image_paths, n_frames, folder_label)
                gt_hint = folder_label

        bar = st.progress(0.0, text="Running CLIP on extracted frames...")
        txt_emb = cached_text_embeddings(tuple(prompts), model, processor, device)
        img_emb = embed_images(model, processor, device, [f["rgb"] for f in frames],
                               progress=lambda p: bar.progress(p, text="Running CLIP on extracted frames..."))
        bar.empty()

        st.session_state["analysis"] = {
            "info": info, "frames": frames, "prompts": prompts, "txt_emb": txt_emb,
            "sims": img_emb @ txt_emb.T,                       # cosine similarity (unit vectors)
            "scale": float(model.logit_scale.exp().item()),    # CLIP's own softmax temperature
            "folder_hint": gt_hint, "device": device,
        }
    except AppError as e:
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        st.error(f"Unexpected error: {type(e).__name__}: {e}. No results were produced.")

A = st.session_state.get("analysis")
if not A:
    st.info("Choose a video (or frame folder), then press **Analyse with CLIP**. "
            "The first run downloads the pretrained CLIP weights, so it needs an internet connection.")
    with st.expander("Disclaimer", expanded=True):
        st.write(DISCLAIMER)
    st.stop()

info, frames, prompts, sims = A["info"], A["frames"], A["prompts"], A["sims"]
N = len(frames)
if parse_prompts(prompt_text) != prompts:
    st.warning("The description list in the sidebar differs from the one used for the current results. "
               "Press **Analyse with CLIP** again to apply your edits.")

# ----------------------------------------------------------------------------
# Section A - Video information
# ----------------------------------------------------------------------------
st.header("A · Source information")
c = st.columns(5)
if info["kind"] == "video":
    c[0].metric("File", short(info["name"], 22))
    c[1].metric("Duration", fmt_ts(info["duration"]) if info["duration"] else "n/a")
    c[2].metric("FPS", f"{info['fps']:.2f}" if info["fps"] else "n/a")
else:
    c[0].metric("Folder", short(info["name"], 22))
    c[1].metric("Images in folder", f"{info['total']}")
    c[2].metric("FPS / duration", "n/a")
c[3].metric("Resolution", f"{info['width']}×{info['height']}")
c[4].metric("Frames analysed", N)
st.caption(f"Device: {A['device'].upper()} · model: {MODEL_NAME} (frozen) · "
           f"{len(prompts)} candidate descriptions"
           + (f" · container folder name: “{A['folder_hint']}” (a folder name, not a verified ground-truth label)"
              if A["folder_hint"] else "")
           + ". Dataset annotation files are not read or interpreted by this app.")

# ----------------------------------------------------------------------------
# Section B - Frame analysis
# ----------------------------------------------------------------------------
st.header("B · Frame analysis")
st.caption("cos = cosine similarity between the CLIP image and text embeddings (the quantity used throughout). "
           "“Rel. weight” is only a softmax across your candidate list, a relative comparison, not a probability "
           "that the event occurred.")
for start in range(0, N, 3):
    cols = st.columns(3)
    for j, col in enumerate(cols):
        i = start + j
        if i >= N:
            break
        fr = frames[i]
        with col, st.container(border=True):
            st.image(fr["rgb"], caption=f"Frame {fr['label']} · {fmt_ts(fr['ts'])}")
            rel = softmax(A["scale"] * sims[i])
            for rank, k in enumerate(np.argsort(-sims[i])[:3], 1):
                st.markdown(f"**{rank}.** {prompts[k]}  \n`cos {sims[i, k]:.4f}` · rel. weight {rel[k] * 100:.1f}%")

with st.expander("All similarity scores (descriptions × frames)"):
    table = pd.DataFrame(sims.T, index=prompts, columns=[f"{f['label']}" for f in frames])
    st.dataframe(table.style.format("{:.4f}"))

# ----------------------------------------------------------------------------
# Section C - Counterfactual analysis
# ----------------------------------------------------------------------------
st.header("C · Counterfactual analysis (masking experiment)")
st.write("Hide one region at a time, run CLIP again, and compare scores. "
         "**Change = original similarity − masked similarity**: positive means the score dropped when the region was hidden.")
cc = st.columns([2, 1, 1, 1])
fpos = cc[0].selectbox("Frame to test", list(range(N)),
                       format_func=lambda i: f"{i + 1}. {frames[i]['label']} · {fmt_ts(frames[i]['ts'])}")
g_rows = cc[1].slider("Grid rows", 2, 5, 3)
g_cols = cc[2].slider("Grid columns", 2, 5, 3)
mask_color = cc[3].radio("Mask colour", ["Neutral grey", "Black"])

if st.button("Run counterfactual test on this frame"):
    try:
        model, processor, device, _ = load_clip_safe()
        bar = st.progress(0.0, text="Running CLIP on masked images...")
        st.session_state["cf"] = run_counterfactual(
            model, processor, device, A, fpos, g_rows, g_cols, mask_color,
            progress=lambda p: bar.progress(p, text="Running CLIP on masked images..."))
        bar.empty()
    except AppError as e:
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        st.error(f"Unexpected error: {type(e).__name__}: {e}. No counterfactual results were produced.")

cf = st.session_state.get("cf")
if cf:
    base, msims, rows, cols_n = cf["orig_sims"], cf["masked_sims"], cf["rows"], cf["cols"]
    fr = frames[cf["fpos"]]
    orig_top = int(np.argmax(base))
    st.subheader(f"Results for frame {fr['label']} · {rows}×{cols_n} grid · {cf['color'].lower()} mask")
    st.markdown(f"**Original top-ranked description:** {prompts[orig_top]} (cos {base[orig_top]:.4f})")

    ref = st.selectbox("Description to track in the heat-map and ranking", list(range(len(prompts))),
                       index=orig_top, format_func=lambda k: prompts[k], key=f"ref_{cf['id']}")
    drops = (base[ref] - msims[:, ref])  # original - masked, one value per region

    left, right = st.columns(2)
    left.image(fr["rgb"], caption="Original frame")
    right.image(heat_overlay(fr["rgb"], drops.reshape(rows, cols_n)),
                caption=f"Change in similarity to “{short(prompts[ref], 50)}” per masked region "
                        "(red = score dropped when hidden, blue = score rose)")

    # Region-wise table
    recs = []
    for ri, lab in enumerate(cf["labels"]):
        ch_all = base - msims[ri]
        kmax = int(np.argmax(np.abs(ch_all)))
        mtop = int(np.argmax(msims[ri]))
        recs.append({
            "Region": lab,
            "Top match (original)": prompts[orig_top],
            "Top match (masked)": prompts[mtop],
            "Top changed?": "yes" if mtop != orig_top else "no",
            "Tracked: original cos": float(base[ref]),
            "Tracked: masked cos": float(msims[ri, ref]),
            "Tracked: change": float(drops[ri]),
            "Largest |change| (any description)": f"{short(prompts[kmax], 34)} ({ch_all[kmax]:+.4f})",
        })
    rtab = pd.DataFrame(recs).sort_values("Tracked: change", ascending=False).reset_index(drop=True)
    st.markdown("**Regions ranked by drop in the tracked description's similarity**")
    st.dataframe(rtab.style.format({"Tracked: original cos": "{:.4f}", "Tracked: masked cos": "{:.4f}",
                                    "Tracked: change": "{:+.4f}"}))
    top3 = rtab.head(3)
    st.write("Most influential masked regions for this description (largest score drop): "
             + ", ".join(f"**{r}** ({v:+.4f})" for r, v in zip(top3["Region"], top3["Tracked: change"])) + ".")

    # Inspect one region
    st.markdown("**Inspect one masked region**")
    ri = st.selectbox("Region", list(range(len(cf["labels"]))), format_func=lambda i: cf["labels"][i],
                      index=int(np.argmax(drops)), key=f"reg_{cf['id']}")
    a, b = st.columns(2)
    a.image(fr["rgb"], caption="Original")
    b.image(cf["masked_imgs"][ri], caption=f"Masked: {cf['labels'][ri]}")
    ch = base - msims[ri]
    detail = pd.DataFrame({"Description": prompts, "Original cos": base, "Masked cos": msims[ri], "Change": ch})
    detail = detail.reindex(detail["Change"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    st.dataframe(detail.style.format({"Original cos": "{:.4f}", "Masked cos": "{:.4f}", "Change": "{:+.4f}"}))
    mtop = int(np.argmax(msims[ri]))
    kmax = int(np.argmax(np.abs(ch)))
    top_text = (f"The top-ranked description changed from “{prompts[orig_top]}” to “{prompts[mtop]}”."
                if mtop != orig_top else
                f"The top-ranked description stayed “{prompts[orig_top]}”.")
    st.info(
        f"Hiding **{cf['labels'][ri]}** changed the similarity to “{prompts[ref]}” from {base[ref]:.4f} to "
        f"{msims[ri, ref]:.4f} (change {ch[ref]:+.4f}). The largest change for any description was "
        f"“{prompts[kmax]}” ({ch[kmax]:+.4f}). {top_text} "
        "This indicates how sensitive CLIP's score is to this region; it does not show that the region contains "
        "the true cause of any event. Differences of a few thousandths may be within normal sensitivity to "
        "any occlusion, so compare regions against each other rather than reading one value in isolation."
    )

# ----------------------------------------------------------------------------
# Section D - Overall analysis
# ----------------------------------------------------------------------------
st.header("D · Overall analysis (frame-level model outputs)")
top1 = [int(np.argmax(sims[i])) for i in range(N)]
order_note = ("Frames are listed in video-time order." if info["kind"] == "video"
              else "Images are listed in file-name order (not necessarily a single video's timeline).")
st.caption(order_note + " No overall crime score is computed: averaging CLIP similarities would not be meaningful.")
chron = pd.DataFrame({
    "Frame": [f["label"] for f in frames],
    "Time": [fmt_ts(f["ts"]) for f in frames],
    "Top-ranked description": [prompts[k] for k in top1],
    "cos": [float(sims[i, k]) for i, k in enumerate(top1)],
})
st.dataframe(chron.style.format({"cos": "{:.4f}"}))

changes = [f"{frames[i - 1]['label']} → {frames[i]['label']}: “{prompts[top1[i - 1]]}” → “{prompts[top1[i]]}”"
           for i in range(1, N) if top1[i] != top1[i - 1]]
if changes:
    st.markdown("**Points where the top-ranked description changes**")
    for line in changes:
        st.markdown(f"- {line}")
else:
    st.markdown("The top-ranked description is the same in every analysed frame.")

counts = pd.Series([prompts[k] for k in top1]).value_counts()
summary = "; ".join(f"“{d}” was top-ranked in {n} of {N} frames" for d, n in counts.items())
st.markdown(f"**Cautious summary:** {summary}. These are relative rankings among the descriptions you "
            "supplied for sampled frames only. They are not detections, not evidence that any event occurred, and "
            "not comparable to dataset labels without a proper evaluation, which this prototype does not perform.")

# ----------------------------------------------------------------------------
# Export (only fields that were actually calculated)
# ----------------------------------------------------------------------------
st.header("E · Export")
rows_out = []
for i, f in enumerate(frames):
    for k, d in enumerate(prompts):
        rows_out.append({"source": info["name"], "frame": f["label"],
                         "timestamp_s": None if f["ts"] is None else round(f["ts"], 3),
                         "description": d, "clip_cosine_similarity": float(sims[i, k])})
csv1 = pd.DataFrame(rows_out).to_csv(index=False).encode("utf-8")
e1, e2 = st.columns(2)
e1.download_button("⬇ Frame similarity scores (CSV)", csv1, file_name="crimescene_frame_scores.csv", mime="text/csv")

if cf:
    f = frames[cf["fpos"]]
    rows_cf = []
    for ri, lab in enumerate(cf["labels"]):
        for k, d in enumerate(prompts):
            rows_cf.append({"source": info["name"], "frame": f["label"],
                            "timestamp_s": None if f["ts"] is None else round(f["ts"], 3),
                            "grid": f"{cf['rows']}x{cf['cols']}", "mask": cf["color"], "masked_region": lab,
                            "description": d, "original_similarity": float(cf["orig_sims"][k]),
                            "masked_similarity": float(cf["masked_sims"][ri, k]),
                            "score_change_original_minus_masked": float(cf["orig_sims"][k] - cf["masked_sims"][ri, k])})
    e2.download_button("⬇ Counterfactual results (CSV)", pd.DataFrame(rows_cf).to_csv(index=False).encode("utf-8"),
                       file_name="crimescene_counterfactual.csv", mime="text/csv")
else:
    e2.caption("Run the counterfactual test to enable that export.")

with st.expander("Disclaimer"):
    st.write(DISCLAIMER)
