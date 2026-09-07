import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageStat


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class OursCandidate:
    id_str: str
    path: str
    run_name: str
    epoch: int
    psnr: float


def _list_method_files(method_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ext in IMG_EXTS:
        for p in glob.glob(os.path.join(method_dir, "**", f"*{ext}"), recursive=True):
            name = os.path.basename(p)
            m = re.search(r"(\d+)", name)
            if not m:
                continue
            id_str = str(int(m.group(1)))
            # Keep first hit to preserve stable selection.
            out.setdefault(id_str, p)
    return out


def _collect_ours(root_run: str, run_name: str) -> Dict[str, List[OursCandidate]]:
    out: Dict[str, List[OursCandidate]] = {}
    epoch_dirs = glob.glob(os.path.join(root_run, "epoch*"))
    for d in epoch_dirs:
        if not os.path.isdir(d):
            continue
        dn = os.path.basename(d)
        m_meta = re.search(r"epoch(\d+)_PSNR([0-9.]+)", dn)
        if not m_meta:
            continue
        epoch = int(m_meta.group(1))
        psnr = float(m_meta.group(2))
        for p in glob.glob(os.path.join(d, "*.png.png")):
            bn = os.path.basename(p)
            if "_gain" in bn:
                continue
            m_id = re.search(r"_(\d+)\.png\.png$", bn)
            if not m_id:
                continue
            id_str = str(int(m_id.group(1)))
            out.setdefault(id_str, []).append(
                OursCandidate(id_str=id_str, path=p, run_name=run_name, epoch=epoch, psnr=psnr)
            )
    return out


def _pick_best_ours(cands: Sequence[OursCandidate]) -> OursCandidate:
    # Priority: train_7 > train_unetT_600x400 > train_6 > train_2,
    # then higher PSNR, then higher epoch.
    run_priority = {
        "train_7": 3,
        "train_unetT_600x400": 2,
        "train_6": 1,
        "train_2": 0,
    }

    def key(c: OursCandidate) -> Tuple[int, float, int]:
        run_rank = run_priority.get(c.run_name, -1)
        return (run_rank, c.psnr, c.epoch)

    return sorted(cands, key=key, reverse=True)[0]


def _choose_ids(common_ids: Sequence[str], limit: int = 8) -> List[str]:
    # Spread by value to avoid selecting clustered IDs only.
    vals = sorted(int(x) for x in common_ids)
    if len(vals) <= limit:
        return [str(v) for v in vals]

    picks: List[int] = []
    for i in range(limit):
        idx = round(i * (len(vals) - 1) / (limit - 1))
        picks.append(vals[idx])
    # Keep unique in case of round collisions.
    uniq = []
    seen = set()
    for v in picks:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    # Top-up if deduplicated.
    if len(uniq) < limit:
        for v in vals:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
            if len(uniq) == limit:
                break
    return [str(v) for v in uniq[:limit]]


def _is_triptych_panel(img: Image.Image) -> bool:
    return int(img.width) >= 700 and int(img.height) >= 250


def _triptych_tile(img: Image.Image, index: int) -> Image.Image:
    """Crop a single tile from a 3-up torchvision save_image grid.

    The saved triptych is typically [low | enhanced | gt] with padding=2.
    """

    if not _is_triptych_panel(img):
        return img

    # torchvision.utils.save_image(..., nrow=3) with default padding=2
    tile_w = int(round((img.width - 8) / 3.0))
    tile_h = int(img.height - 4)
    tile_w = max(1, tile_w)
    tile_h = max(1, tile_h)
    index = int(max(0, min(2, index)))
    x0 = 2 + index * (tile_w + 2)
    y0 = 2
    x1 = min(img.width, x0 + tile_w)
    y1 = min(img.height, y0 + tile_h)
    return img.crop((x0, y0, x1, y1))


def _load_and_resize(path: str, wh: Tuple[int, int], triptych_index: Optional[int] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if triptych_index is not None:
        img = _triptych_tile(img, triptych_index)
    return img.resize(wh, Image.Resampling.BICUBIC)


def _mean_luma(path: str, triptych_index: Optional[int] = None) -> float:
    img = Image.open(path).convert("RGB")
    if triptych_index is not None:
        img = _triptych_tile(img, triptych_index)
    stat = ImageStat.Stat(img)
    return float(sum(stat.mean) / max(1, len(stat.mean)))


def _build_panel(
    ids: Sequence[str],
    columns: Sequence[Tuple[str, Dict[str, str]]],
    ours_paths: Dict[str, str],
    out_path: str,
    cell_wh: Tuple[int, int] = (240, 180),
) -> None:
    font = ImageFont.load_default()
    cw, ch = cell_wh
    left_w = 110
    top_h = 32
    pad = 4

    n_rows = len(ids)
    n_cols = len(columns) + 1  # + Ours
    W = left_w + n_cols * (cw + pad) + pad
    H = top_h + n_rows * (ch + pad) + pad

    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    # Header
    x0 = left_w + pad
    headers = [name for name, _ in columns] + ["Ours"]
    for j, h in enumerate(headers):
        x = x0 + j * (cw + pad)
        draw.rectangle([x, pad, x + cw, top_h], fill=(235, 235, 235), outline=(180, 180, 180), width=1)
        draw.text((x + 6, pad + 9), h, fill=(20, 20, 20), font=font)

    # Rows
    for i, id_str in enumerate(ids):
        y = top_h + pad + i * (ch + pad)
        draw.rectangle([pad, y, left_w - 4, y + ch], fill=(240, 240, 240), outline=(180, 180, 180), width=1)
        draw.text((10, y + 8), f"ID {id_str}", fill=(10, 10, 10), font=font)

        # Baselines
        for j, (_name, fmap) in enumerate(columns):
            x = x0 + j * (cw + pad)
            p = fmap[id_str]
            tile = _load_and_resize(p, (cw, ch))
            canvas.paste(tile, (x, y))
            draw.rectangle([x, y, x + cw, y + ch], outline=(170, 170, 170), width=1)

        # Ours last: crop the middle tile from the triptych so it is a single enhanced image.
        x = x0 + (n_cols - 1) * (cw + pad)
        tile = _load_and_resize(ours_paths[id_str], (cw, ch), triptych_index=1)
        canvas.paste(tile, (x, y))
        draw.rectangle([x, y, x + cw, y + ch], outline=(220, 50, 50), width=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    base = "/home/zhiqinkun/LLIM"
    test_root = os.path.join(base, "test_images")
    ours_roots = [
        ("train_2", os.path.join(base, "comparision", "RL-DiffNet-main", "train_2")),
        ("train_6", os.path.join(base, "comparision", "RL-DiffNet-main", "train_6")),
        ("train_7", os.path.join(base, "comparision", "RL-DiffNet-main", "train_7")),
        ("train_unetT_600x400", os.path.join(base, "comparision", "RL-DiffNet-main", "train_unetT_600x400")),
    ]

    method_dirs = [
        "clip-lit",
        "Zero-DCE",
        "RUAS",
        "SCI-CVPR2022",
        "Retinexnet",
        "Retinexformer",
        "KinDmaster",
        "EnlightenGAN-master",
        "URetinex-nex",
    ]

    methods: List[Tuple[str, Dict[str, str]]] = []
    for m in method_dirs:
        p = os.path.join(test_root, m)
        fmap = _list_method_files(p)
        if len(fmap) == 0:
            continue
        methods.append((m, fmap))

    ours_all: Dict[str, List[OursCandidate]] = {}
    for run_name, root in ours_roots:
        part = _collect_ours(root, run_name)
        for k, v in part.items():
            ours_all.setdefault(k, []).extend(v)

    ours_best: Dict[str, OursCandidate] = {k: _pick_best_ours(v) for k, v in ours_all.items() if v}

    common = set(ours_best.keys())
    for _m, fmap in methods:
        common &= set(fmap.keys())
    if not common:
        raise RuntimeError("No common IDs between baselines and our sampled outputs")

    out_dir = os.path.join(base, "comparision", "RL-DiffNet-main", "comparison_figures")
    os.makedirs(out_dir, exist_ok=True)

    # Rank candidate IDs by low-light brightness (from the left tile of the triptych) so we can
    # generate a more diverse set of rows for paper selection.
    scored_ids: List[Tuple[float, str]] = []
    for id_str in sorted(common, key=lambda x: int(x)):
        c = ours_best[id_str]
        try:
            score = _mean_luma(c.path, triptych_index=0)
        except Exception:
            score = 0.0
        scored_ids.append((score, id_str))
    scored_ids.sort(key=lambda x: (x[0], int(x[1])))
    brightness_ranked_ids = [i for _s, i in scored_ids]

    # 8-sample set: evenly spread over brightness range, train_7 priority already baked into ours_best.
    selected_ids_8 = _choose_ids(brightness_ranked_ids, limit=8)
    ours_paths_8 = {k: ours_best[k].path for k in selected_ids_8}

    # 12-sample set: more candidate rows for visual inspection and paper selection.
    selected_ids_12 = _choose_ids(brightness_ranked_ids, limit=12)
    ours_paths_12 = {k: ours_best[k].path for k in selected_ids_12}

    # Full version (all baselines + ours), single enhanced image in the last column.
    full_out = os.path.join(out_dir, "comparison_full_single_8x10.png")
    _build_panel(
        ids=selected_ids_8,
        columns=methods,
        ours_paths=ours_paths_8,
        out_path=full_out,
        cell_wh=(200, 150),
    )

    full12_out = os.path.join(out_dir, "comparison_full_single_12x10.png")
    _build_panel(
        ids=selected_ids_12,
        columns=methods,
        ours_paths=ours_paths_12,
        out_path=full12_out,
        cell_wh=(200, 150),
    )

    # Paper-friendly compact version
    compact_names = ["Zero-DCE", "RUAS", "SCI-CVPR2022", "Retinexformer", "KinDmaster"]
    compact_cols = [(n, f) for (n, f) in methods if n in compact_names]
    compact_out = os.path.join(out_dir, "comparison_compact_single_8x6.png")
    _build_panel(
        ids=selected_ids_8,
        columns=compact_cols,
        ours_paths=ours_paths_8,
        out_path=compact_out,
        cell_wh=(240, 180),
    )

    # Ordered filename lists matching the stitching order.
    ordered_8_txt = os.path.join(out_dir, "comparison_ordered_filenames_single_8.txt")
    with open(ordered_8_txt, "w", encoding="utf-8") as f:
        f.write("# Row-major order of the 8-sample full comparison panel\n")
        f.write("# Each row: ID | baselines in panel order | Ours(single enhanced crop)\n\n")
        for id_str in selected_ids_8:
            row = [f"ID {id_str}"]
            for name, fmap in methods:
                row.append(f"{name}: {os.path.basename(fmap[id_str])}")
            row.append(f"Ours: {os.path.basename(ours_paths_8[id_str])} [crop=middle tile]")
            f.write(" | ".join(row) + "\n")

    ordered_12_txt = os.path.join(out_dir, "comparison_ordered_filenames_single_12.txt")
    with open(ordered_12_txt, "w", encoding="utf-8") as f:
        f.write("# Row-major order of the 12-sample full comparison panel\n")
        f.write("# Each row: ID | baselines in panel order | Ours(single enhanced crop)\n\n")
        for id_str in selected_ids_12:
            row = [f"ID {id_str}"]
            for name, fmap in methods:
                row.append(f"{name}: {os.path.basename(fmap[id_str])}")
            row.append(f"Ours: {os.path.basename(ours_paths_12[id_str])} [crop=middle tile]")
            f.write(" | ".join(row) + "\n")

    # Save manifest for transparency.
    manifest = os.path.join(out_dir, "comparison_manifest_single.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("Selected IDs (8-sample panel):\n")
        f.write(", ".join(selected_ids_8) + "\n\n")
        f.write("Selected IDs (12-sample panel):\n")
        f.write(", ".join(selected_ids_12) + "\n\n")
        f.write("Ours source image per ID (best available run among train_2/train_6/train_7/train_unetT_600x400):\n")
        for id_str in selected_ids_12:
            c = ours_best[id_str]
            f.write(f"ID {id_str}: {c.path} (run={c.run_name}, epoch={c.epoch}, psnr={c.psnr:.4f})\n")

    score_txt = os.path.join(out_dir, "comparison_brightness_ranking_single.txt")
    with open(score_txt, "w", encoding="utf-8") as f:
        f.write("# Lower score = darker low-light input crop (left tile of triptych)\n")
        for score, id_str in scored_ids:
            c = ours_best[id_str]
            f.write(f"{score:.4f}\tID {id_str}\t{os.path.basename(c.path)}\t{c.run_name}\tpsnr={c.psnr:.4f}\n")

    print("[OK] full8:", full_out)
    print("[OK] full12:", full12_out)
    print("[OK] compact:", compact_out)
    print("[OK] ordered8:", ordered_8_txt)
    print("[OK] ordered12:", ordered_12_txt)
    print("[OK] manifest:", manifest)
    print("[OK] ranking:", score_txt)
    print("[IDs-8]", ", ".join(selected_ids_8))
    print("[IDs-12]", ", ".join(selected_ids_12))


if __name__ == "__main__":
    main()
