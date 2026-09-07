#!/usr/bin/env python3
"""Summarize SCIST ablations and build an objective paper-visualization shortlist."""

import argparse
import csv
import json
import os
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple


EXPECTED_VARIANTS = (
    "clip_only",
    "target_state_only",
    "wo_bias_suppression",
    "wo_gsf",
    "wo_lstate",
    "full",
)
DISPLAY_NAMES = {
    "clip_only": "CLIP only",
    "target_state_only": "Target state only",
    "wo_bias_suppression": "w/o bias suppression",
    "wo_gsf": "w/o GSF",
    "wo_lstate": "w/o target-state condition",
    "full": "Full",
}
LATEX_NAMES = {
    "clip_only": "CLIP only",
    "target_state_only": "Target state only",
    "wo_bias_suppression": r"w/o bias suppression",
    "wo_gsf": "w/o GSF",
    "wo_lstate": r"w/o $\mathcal{L}_{state}$",
    "full": "Full",
}
METRICS = ("psnr", "ssim", "niqe", "musiq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="ablation_main", help="Directory containing one subdirectory per ablation.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of objective visual candidates to export.")
    parser.add_argument("--sample_ids", default="", help="Optional comma-separated image IDs overriding automatic ranking.")
    parser.add_argument("--input_root", default="", help="Low-light input directory; inferred from full/config.yaml when empty.")
    parser.add_argument("--gt_root", default="", help="Ground-truth directory; inferred from full/config.yaml when empty.")
    parser.add_argument("--no_montage", action="store_true", help="Only write metric tables and candidate rankings.")
    return parser.parse_args()


def read_summary(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_per_image(path: str) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["image"])] = {metric: float(row[metric]) for metric in METRICS}
    return rows


def discover(root: str) -> Tuple[List[str], Dict[str, Dict[str, object]], Dict[str, Dict[str, Dict[str, float]]]]:
    summaries: Dict[str, Dict[str, object]] = {}
    per_image: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant in EXPECTED_VARIANTS:
        summary_path = os.path.join(root, variant, "lolv1_test", "summary_metrics.json")
        csv_path = os.path.join(root, variant, "lolv1_test", "per_image_metrics.csv")
        if os.path.isfile(summary_path) and os.path.isfile(csv_path):
            summaries[variant] = read_summary(summary_path)
            per_image[variant] = read_per_image(csv_path)
    variants = [variant for variant in EXPECTED_VARIANTS if variant in summaries]
    return variants, summaries, per_image


def write_summary_csv(root: str, variants: Sequence[str], summaries: Dict[str, Dict[str, object]]) -> str:
    path = os.path.join(root, "ablation_summary.csv")
    full = summaries.get("full", {})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        fields = ["variant", "display_name", "num_images"] + list(METRICS) + ["delta_psnr_vs_full", "delta_ssim_vs_full", "delta_niqe_vs_full", "delta_musiq_vs_full"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in variants:
            summary = summaries[variant]
            row = {
                "variant": variant,
                "display_name": DISPLAY_NAMES[variant],
                "num_images": summary.get("num_images"),
            }
            for metric in METRICS:
                value = summary.get(metric)
                row[metric] = value
                full_value = full.get(metric)
                row["delta_{}_vs_full".format(metric)] = "" if value is None or full_value is None else float(value) - float(full_value)
            writer.writerow(row)
    return path


def best_value(variants: Sequence[str], summaries: Dict[str, Dict[str, object]], metric: str) -> Optional[float]:
    values = [float(summaries[v][metric]) for v in variants if summaries[v].get(metric) is not None]
    if not values:
        return None
    return min(values) if metric == "niqe" else max(values)


def metric_cell(value: object, best: Optional[float], digits: int) -> str:
    if value is None:
        return "NA"
    number = float(value)
    rendered = "{:.{}f}".format(number, digits)
    return "**{}**".format(rendered) if best is not None and abs(number - best) < 1e-12 else rendered


def write_summary_markdown(root: str, variants: Sequence[str], summaries: Dict[str, Dict[str, object]]) -> str:
    path = os.path.join(root, "ablation_summary.md")
    best = {metric: best_value(variants, summaries, metric) for metric in METRICS}
    missing = [DISPLAY_NAMES[v] for v in EXPECTED_VARIANTS if v not in variants]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# SCIST ablation summary\n\n")
        handle.write("PSNR, SSIM and MUSIQ are higher-is-better; NIQE is lower-is-better. Bold denotes the best available result.\n\n")
        handle.write("| Variant | Images | PSNR ↑ | SSIM ↑ | NIQE ↓ | MUSIQ ↑ |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for variant in variants:
            summary = summaries[variant]
            handle.write(
                "| {} | {} | {} | {} | {} | {} |\n".format(
                    DISPLAY_NAMES[variant],
                    summary.get("num_images", "NA"),
                    metric_cell(summary.get("psnr"), best["psnr"], 4),
                    metric_cell(summary.get("ssim"), best["ssim"], 4),
                    metric_cell(summary.get("niqe"), best["niqe"], 4),
                    metric_cell(summary.get("musiq"), best["musiq"], 4),
                )
            )
        if missing:
            handle.write("\nPending experiment(s): {}. Re-run this script after evaluation artifacts are generated.\n".format(", ".join(missing)))
        handle.write("\nThe visualization shortlist is ranked by `ΔPSNR + 20 × ΔSSIM`, comparing Full with the mean of all available ablations; ties prioritize images where Full wins both metrics against more variants.\n")
    return path


def latex_metric_cell(value: object, best: Optional[float], digits: int) -> str:
    if value is None:
        return "--"
    number = float(value)
    rendered = "{:.{}f}".format(number, digits)
    return r"\textbf{{{}}}".format(rendered) if best is not None and abs(number - best) < 1e-12 else rendered


def write_summary_latex(root: str, variants: Sequence[str], summaries: Dict[str, Dict[str, object]]) -> str:
    path = os.path.join(root, "ablation_table.tex")
    best = {metric: best_value(variants, summaries, metric) for metric in METRICS}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("% Requires \\usepackage{booktabs}\n")
        handle.write("\\begin{table}[t]\n")
        handle.write("  \\centering\n")
        handle.write("  \\caption{Ablation study on LOL-v1. Best available results are in bold.}\n")
        handle.write("  \\label{tab:scist_ablation}\n")
        handle.write("  \\begin{tabular}{lcccc}\n")
        handle.write("    \\toprule\n")
        handle.write("    Variant & PSNR $\\uparrow$ & SSIM $\\uparrow$ & NIQE $\\downarrow$ & MUSIQ $\\uparrow$ \\\\\n")
        handle.write("    \\midrule\n")
        for variant in EXPECTED_VARIANTS:
            summary = summaries.get(variant, {})
            handle.write(
                "    {} & {} & {} & {} & {} \\\\\n".format(
                    LATEX_NAMES[variant],
                    latex_metric_cell(summary.get("psnr"), best["psnr"], 4),
                    latex_metric_cell(summary.get("ssim"), best["ssim"], 4),
                    latex_metric_cell(summary.get("niqe"), best["niqe"], 4),
                    latex_metric_cell(summary.get("musiq"), best["musiq"], 4),
                )
            )
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}\n")
        handle.write("\\end{table}\n")
    return path


def rank_candidates(variants: Sequence[str], per_image: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict[str, object]]:
    ablations = [variant for variant in variants if variant != "full"]
    if "full" not in per_image or not ablations:
        return []
    common = set(per_image["full"])
    for variant in ablations:
        common.intersection_update(per_image[variant])
    ranked: List[Dict[str, object]] = []
    for image_name in sorted(common):
        full = per_image["full"][image_name]
        mean_psnr = mean(per_image[v][image_name]["psnr"] for v in ablations)
        mean_ssim = mean(per_image[v][image_name]["ssim"] for v in ablations)
        delta_psnr = full["psnr"] - mean_psnr
        delta_ssim = full["ssim"] - mean_ssim
        wins = sum(
            full["psnr"] > per_image[v][image_name]["psnr"]
            and full["ssim"] > per_image[v][image_name]["ssim"]
            for v in ablations
        )
        ranked.append(
            {
                "image": image_name,
                "wins": wins,
                "score": delta_psnr + 20.0 * delta_ssim,
                "delta_psnr": delta_psnr,
                "delta_ssim": delta_ssim,
                "full_psnr": full["psnr"],
                "full_ssim": full["ssim"],
                "mean_ablation_psnr": mean_psnr,
                "mean_ablation_ssim": mean_ssim,
            }
        )
    ranked.sort(key=lambda row: (int(row["wins"]), float(row["score"])), reverse=True)
    return ranked


def write_candidates(root: str, ranked: Sequence[Dict[str, object]]) -> str:
    path = os.path.join(root, "visual_candidates.csv")
    fields = ["rank", "image", "wins", "score", "delta_psnr", "delta_ssim", "full_psnr", "full_ssim", "mean_ablation_psnr", "mean_ablation_ssim"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(ranked, 1):
            writer.writerow(dict(row, rank=index))
    return path


def config_value(path: str, key: str) -> str:
    if not os.path.isfile(path):
        return ""
    prefix = key + ":"
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(prefix):
                raw = line.split(":", 1)[1].strip()
                try:
                    return str(json.loads(raw))
                except Exception:
                    return raw.strip("\"'")
    return ""


def locate_image(directory: str, image_name: str) -> str:
    direct = os.path.join(directory, image_name)
    if os.path.isfile(direct):
        return direct
    stem = os.path.splitext(image_name)[0]
    for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = os.path.join(directory, stem + extension)
        if os.path.isfile(candidate):
            return candidate
    return ""


def validate_output_sizes(root: str, variants: Sequence[str], input_root: str) -> str:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to validate output sizes: {}".format(exc))

    output = os.path.join(root, "image_size_validation.csv")
    rows = []
    failures = []
    input_names = sorted(name for name in os.listdir(input_root) if os.path.isfile(os.path.join(input_root, name)))
    for variant in variants:
        enhanced_root = os.path.join(root, variant, "lolv1_test", "enhanced_rgb")
        checked = matching = mismatched = missing = 0
        for image_name in input_names:
            source_path = locate_image(input_root, image_name)
            result_path = locate_image(enhanced_root, image_name)
            if not result_path:
                missing += 1
                continue
            with Image.open(source_path) as source, Image.open(result_path) as result:
                checked += 1
                if source.size == result.size:
                    matching += 1
                else:
                    mismatched += 1
                    if len(failures) < 10:
                        failures.append("{}:{} source={} result={}".format(variant, image_name, source.size, result.size))
        rows.append({"variant": variant, "checked": checked, "matching": matching, "mismatched": mismatched, "missing": missing})

    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "checked", "matching", "mismatched", "missing"])
        writer.writeheader()
        writer.writerows(rows)

    incomplete = [row for row in rows if int(row["mismatched"]) > 0 or int(row["missing"]) > 0]
    if incomplete:
        detail = "; ".join(failures) if failures else str(incomplete)
        raise RuntimeError("Output-size validation failed; refusing to build a paper montage: {}".format(detail))
    return output


def diverse_candidates(ranked: Sequence[Dict[str, object]], input_root: str, count: int, min_hamming: int = 12) -> List[str]:
    """Greedily retain high-ranked examples while rejecting near-duplicate scenes."""
    try:
        from PIL import Image, ImageOps
    except Exception:
        return [str(row["image"]) for row in ranked[:count]]

    hashes: List[int] = []
    selected: List[str] = []
    for row in ranked:
        image_name = str(row["image"])
        path = locate_image(input_root, image_name)
        if not path:
            continue
        with Image.open(path) as source:
            gray = ImageOps.exif_transpose(source).convert("L").resize((9, 8))
            pixels = list(gray.getdata())
        value = 0
        for y in range(8):
            for x in range(8):
                value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
        if any(bin(value ^ previous).count("1") < int(min_hamming) for previous in hashes):
            continue
        selected.append(image_name)
        hashes.append(value)
        if len(selected) >= int(count):
            break

    if len(selected) < int(count):
        for row in ranked:
            image_name = str(row["image"])
            if image_name not in selected:
                selected.append(image_name)
            if len(selected) >= int(count):
                break
    return selected


def make_montage(root: str, variants: Sequence[str], sample_ids: Sequence[str], input_root: str, gt_root: str) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except Exception as exc:
        raise RuntimeError("Pillow is required to create the paper montage: {}".format(exc))

    shown_variants = [v for v in variants if v != "full"] + (["full"] if "full" in variants else [])
    columns = [("Input", input_root, "input")]
    columns.extend((DISPLAY_NAMES[v], os.path.join(root, v, "lolv1_test", "enhanced_rgb"), v) for v in shown_variants)
    columns.append(("GT", gt_root, "gt"))

    tile_w, tile_h = 300, 203
    header_h, row_gap = 46, 4
    canvas_h = header_h + tile_h * len(sample_ids) + row_gap * max(0, len(sample_ids) - 1)
    canvas = Image.new("RGB", (tile_w * len(columns), canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    def title_label(label: str) -> str:
        return "w/o target-state\ncondition" if label == "w/o target-state condition" else label

    font_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ]
    font_path = next((path for path in font_candidates if os.path.isfile(path)), "")
    font = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
    for col, (label, _directory, _kind) in enumerate(columns):
        label = title_label(label)
        center_x = col * tile_w + tile_w / 2
        center_y = header_h / 2
        bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=0, align="center")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.multiline_text(
            (center_x - text_w / 2, center_y - text_h / 2),
            label,
            fill="black",
            font=font,
            spacing=0,
            align="center",
        )

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for row, image_name in enumerate(sample_ids):
        top = header_h + row * (tile_h + row_gap)
        for col, (_label, directory, _kind) in enumerate(columns):
            path = locate_image(directory, image_name)
            if not path:
                continue
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((tile_w - 8, tile_h - 8), resampling)
                x = col * tile_w + (tile_w - image.width) // 2
                y = top + (tile_h - image.height) // 2
                canvas.paste(image, (x, y))

    output = os.path.join(root, "paper_visualization.png")
    canvas.save(output)

    # Build the PDF from the original image tiles and native text instead of
    # wrapping the assembled PNG.  Text/layout therefore remain vector objects,
    # while the photographic results are embedded as individual raster images.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]
    canvas_w, canvas_h = canvas.size
    figure_w = 7.6
    fig = plt.figure(figsize=(figure_w, figure_w * canvas_h / canvas_w), facecolor="white")
    for col, (label, _directory, _kind) in enumerate(columns):
        label = title_label(label)
        fig.text(
            (col * tile_w + tile_w / 2) / canvas_w,
            1.0 - (header_h / 2) / canvas_h,
            label,
            ha="center",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            linespacing=0.9,
        )

    for row, image_name in enumerate(sample_ids):
        top = header_h + row * (tile_h + row_gap)
        for col, (_label, directory, _kind) in enumerate(columns):
            path = locate_image(directory, image_name)
            if not path:
                continue
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB").copy()
            # Match the compact PNG geometry.  interpolation='none' prevents
            # Matplotlib from baking resampling into the PDF image object.
            target_w = tile_w - 8
            target_h = target_w * image.height / image.width
            if target_h > tile_h - 8:
                target_h = tile_h - 8
                target_w = target_h * image.width / image.height
            x = col * tile_w + (tile_w - target_w) / 2
            y = top + (tile_h - target_h) / 2
            ax = fig.add_axes(
                [
                    x / canvas_w,
                    1.0 - (y + target_h) / canvas_h,
                    target_w / canvas_w,
                    target_h / canvas_h,
                ]
            )
            ax.imshow(image, interpolation="none", aspect="auto")
            ax.set_axis_off()

    pdf_output = os.path.join(root, "paper_visualization.pdf")
    vector_output = os.path.join(root, "paper_visualization_vector.pdf")
    fig.savefig(pdf_output, format="pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(vector_output, format="pdf", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.root)
    variants, summaries, per_image = discover(root)
    if not variants:
        raise SystemExit("No complete ablation evaluation artifacts found under {}".format(root))

    summary_csv = write_summary_csv(root, variants, summaries)
    summary_md = write_summary_markdown(root, variants, summaries)
    summary_tex = write_summary_latex(root, variants, summaries)
    ranked = rank_candidates(variants, per_image)
    candidates_csv = write_candidates(root, ranked)

    config_path = os.path.join(root, "full", "config.yaml")
    input_root = args.input_root or config_value(config_path, "low_root")
    gt_root = args.gt_root or config_value(config_path, "gt_root")
    requested = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    chosen = requested or diverse_candidates(ranked, input_root, max(0, int(args.top_k)))
    montage = ""
    size_validation = ""
    if input_root:
        size_validation = validate_output_sizes(root, variants, input_root)
    if not args.no_montage and chosen:
        if input_root and gt_root:
            montage = make_montage(root, variants, chosen, input_root, gt_root)
        else:
            print("[Warning] input_root/gt_root unavailable; skipped montage.")

    print("[Summary] variants={}".format(",".join(variants)))
    print("[Summary] table_csv={}".format(summary_csv))
    print("[Summary] table_md={}".format(summary_md))
    print("[Summary] table_tex={}".format(summary_tex))
    print("[Summary] candidates_csv={}".format(candidates_csv))
    if size_validation:
        print("[Summary] size_validation={}".format(size_validation))
    print("[Summary] selected={}".format(",".join(chosen)))
    if montage:
        print("[Summary] montage={}".format(montage))


if __name__ == "__main__":
    main()
