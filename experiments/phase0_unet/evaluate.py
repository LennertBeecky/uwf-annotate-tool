"""Phase-0 evaluate.py — all conditions scored against the SAME manual Test GT.

Usage:
    PYTHONPATH=src python -u experiments/phase0_unet/evaluate.py \
        --checkpoints cond_A:results/cond_A/checkpoint_best.pt \
                      cond_C:results/cond_C/checkpoint_best.pt \
                      cond_G:results/cond_G/checkpoint_best.pt

Produces:
    results/comparison_table.csv   — mean Dice/sens/spec/clDice per condition
    results/<cond>/test_metrics.csv — per-image metrics for each condition
    results/scatter_A_vs_C.png     — per-image Dice, A on x, C on y
    results/scatter_A_vs_G.png     — per-image Dice, A on x, G on y
    results/stat_tests.txt         — Wilcoxon signed-rank p-values

Always evaluated against `databases/Test/{images,artery,veins}/*.png`
(manual GT), never against physics-reconstructed test labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import IMAGENET_MEAN, IMAGENET_STD, _listdir_stems  # type: ignore  # noqa: E402
from unet import SmallUNet  # type: ignore  # noqa: E402


# --- metrics ---


def dice_score(p: np.ndarray, g: np.ndarray) -> float:
    p = p.astype(bool); g = g.astype(bool)
    num = 2.0 * float((p & g).sum())
    den = float(p.sum() + g.sum())
    return num / den if den > 0 else 1.0


def sens_spec(p: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    p = p.astype(bool); g = g.astype(bool)
    tp = float((p & g).sum())
    fn = float((~p & g).sum())
    fp = float((p & ~g).sum())
    tn = float((~p & ~g).sum())
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Topology-aware centerline Dice (Shit 2021)."""
    p = pred.astype(bool); g = gt.astype(bool)
    sk_p = skeletonize(p)
    sk_g = skeletonize(g)
    if not sk_p.any() or not sk_g.any():
        return 0.0
    t_sens = float((sk_p & g).sum()) / max(sk_p.sum(), 1)
    t_prec = float((sk_g & p).sum()) / max(sk_g.sum(), 1)
    if t_sens + t_prec == 0:
        return 0.0
    return 2 * t_sens * t_prec / (t_sens + t_prec)


def count_components(m: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(m.astype(np.uint8))
    return int(n) - 1  # exclude background


# --- load & predict ---


def load_model(ckpt_path: Path, device: torch.device) -> SmallUNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SmallUNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _predict_one(model: SmallUNet, image_bgr: np.ndarray, device: torch.device,
                 resolution: int = 512) -> np.ndarray:
    """Return (2, H_orig, W_orig) binary uint8 predictions."""
    h0, w0 = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[None]
    x_t = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy()[0]   # (2, H, W)
    # Resize back to original
    out = np.zeros((2, h0, w0), dtype=np.uint8)
    for c in range(2):
        p = cv2.resize(probs[c], (w0, h0), interpolation=cv2.INTER_LINEAR)
        out[c] = (p > 0.5).astype(np.uint8)
    return out


# --- main ---


def _parse_checkpoints(entries: list[str]) -> dict[str, Path]:
    out = {}
    for e in entries:
        name, path = e.split(":", 1)
        out[name] = Path(path)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="cond_name:path entries, e.g. cond_A:results/cond_A/checkpoint_best.pt")
    p.add_argument("--db-root", default="databases", type=Path)
    p.add_argument("--output-dir", default="experiments/phase0_unet/results", type=Path)
    p.add_argument("--resolution", type=int, default=512)
    args = p.parse_args()

    device = _pick_device()
    print(f"Device: {device}")

    test_dir = args.db_root / "Test"
    imgs = _listdir_stems(test_dir / "images")
    art = _listdir_stems(test_dir / "artery")
    vn = _listdir_stems(test_dir / "veins")
    stems = sorted(set(imgs) & set(art) & set(vn))
    print(f"Test images: {len(stems)}")

    ckpts = _parse_checkpoints(args.checkpoints)
    per_image: dict[str, list[dict]] = {name: [] for name in ckpts}

    for name, path in ckpts.items():
        print(f"\n=== {name} ({path}) ===")
        model = load_model(path, device)
        for stem in stems:
            img = cv2.imread(str(imgs[stem]))
            gt_a = cv2.imread(str(art[stem]), cv2.IMREAD_GRAYSCALE) > 127
            gt_v = cv2.imread(str(vn[stem]), cv2.IMREAD_GRAYSCALE) > 127
            pred = _predict_one(model, img, device, args.resolution)
            pa, pv = pred[0].astype(bool), pred[1].astype(bool)
            pred_vessel = pa | pv
            gt_vessel = gt_a | gt_v

            d_all = dice_score(pred_vessel, gt_vessel)
            d_art = dice_score(pa, gt_a)
            d_vn = dice_score(pv, gt_v)
            sens, spec = sens_spec(pred_vessel, gt_vessel)
            cld = cldice(pred_vessel, gt_vessel)
            n_comp_a = count_components(pa)
            n_comp_v = count_components(pv)
            per_image[name].append({
                "stem": stem, "dice": d_all, "dice_a": d_art, "dice_v": d_vn,
                "sens": sens, "spec": spec, "cldice": cld,
                "n_comp_a": n_comp_a, "n_comp_v": n_comp_v,
            })

        out_cond = args.output_dir / name
        out_cond.mkdir(parents=True, exist_ok=True)
        csv_p = out_cond / "test_metrics.csv"
        with csv_p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_image[name][0].keys()))
            w.writeheader()
            for row in per_image[name]:
                w.writerow(row)
        print(f"  wrote {csv_p}  ({len(per_image[name])} rows)")

    # --- aggregated comparison ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmp_csv = args.output_dir / "comparison_table.csv"
    with cmp_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "dice", "dice_a", "dice_v", "sens", "spec",
                    "cldice", "n_comp_a_mean", "n_comp_v_mean"])
        for name, rows in per_image.items():
            arr = {k: np.array([r[k] for r in rows], dtype=float)
                   for k in ("dice", "dice_a", "dice_v", "sens", "spec", "cldice",
                             "n_comp_a", "n_comp_v")}
            w.writerow([name,
                        f"{arr['dice'].mean():.4f}",
                        f"{arr['dice_a'].mean():.4f}",
                        f"{arr['dice_v'].mean():.4f}",
                        f"{arr['sens'].mean():.4f}",
                        f"{arr['spec'].mean():.4f}",
                        f"{arr['cldice'].mean():.4f}",
                        f"{arr['n_comp_a'].mean():.1f}",
                        f"{arr['n_comp_v'].mean():.1f}"])
    print(f"\nWrote {cmp_csv}")

    # Print the table
    print("\n" + "=" * 80)
    print(f"{'Cond':<10} {'Dice':>7} {'DiceA':>7} {'DiceV':>7} {'Sens':>6} "
          f"{'Spec':>6} {'clDice':>7} {'#CompA':>7} {'#CompV':>7}")
    print("-" * 80)
    for name, rows in per_image.items():
        arr = {k: np.mean([r[k] for r in rows]) for k in
               ("dice", "dice_a", "dice_v", "sens", "spec", "cldice",
                "n_comp_a", "n_comp_v")}
        print(f"{name:<10} {arr['dice']:>7.4f} {arr['dice_a']:>7.4f} {arr['dice_v']:>7.4f} "
              f"{arr['sens']:>6.3f} {arr['spec']:>6.3f} {arr['cldice']:>7.4f} "
              f"{arr['n_comp_a']:>7.1f} {arr['n_comp_v']:>7.1f}")
    print("=" * 80)

    # --- statistical tests (paired Wilcoxon) ---
    stat_path = args.output_dir / "stat_tests.txt"
    with stat_path.open("w") as f:
        names = list(per_image.keys())
        f.write("Paired Wilcoxon signed-rank test on per-image Dice\n\n")
        print("\nPaired Wilcoxon signed-rank on per-image Dice:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = [r["dice"] for r in per_image[names[i]]]
                b = [r["dice"] for r in per_image[names[j]]]
                try:
                    stat, pv = wilcoxon(a, b)
                    line = (f"  {names[i]} vs {names[j]}:  "
                            f"median Δ = {np.median(np.array(b) - np.array(a)):+.4f}  "
                            f"W = {stat:.1f}  p = {pv:.4g}")
                except ValueError as e:
                    line = f"  {names[i]} vs {names[j]}:  wilcoxon error ({e})"
                f.write(line + "\n")
                print(line)
    print(f"Wrote {stat_path}")

    # --- scatter plots ---
    if "cond_A" in per_image:
        a_vals = {r["stem"]: r["dice"] for r in per_image["cond_A"]}
        for other in ("cond_C", "cond_G"):
            if other not in per_image:
                continue
            o_vals = {r["stem"]: r["dice"] for r in per_image[other]}
            xs, ys = [], []
            for s, x in a_vals.items():
                if s in o_vals:
                    xs.append(x); ys.append(o_vals[s])
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(xs, ys, alpha=0.6, s=18)
            lo, hi = 0.0, 1.0
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("Dice — cond_A (manual)")
            ax.set_ylabel(f"Dice — {other}")
            ax.set_title(f"Per-image Dice: {other} vs cond_A  (n={len(xs)})")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend()
            out_png = args.output_dir / f"scatter_A_vs_{other.replace('cond_', '')}.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=130)
            plt.close(fig)
            print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
