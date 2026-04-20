"""typer-based CLI: `uwf-extract run --image ... --output-dir ...`."""

from __future__ import annotations

from pathlib import Path

try:
    import typer
except ImportError:  # typer is an optional extra
    typer = None  # type: ignore[assignment]

from uwf_zonal_extraction.config import ExtractionConfig
from uwf_zonal_extraction.extractor import extract_caliber_from_image
from uwf_zonal_extraction.viz import save_extraction_summary

if typer is not None:
    app = typer.Typer(help="UWF zonal CRAE/CRVE extraction.")

    @app.command()
    def run(
        image: Path = typer.Option(..., "--image", exists=True, help="UWF image path."),
        output_dir: Path = typer.Option(..., "--output-dir", help="Where to write results."),
        config: Path | None = typer.Option(None, "--config", help="Optional YAML config."),
        laterality: str | None = typer.Option(None, "--laterality", help="'OD' or 'OS'"),
        models_dir: Path = typer.Option(Path("models"), "--models-dir"),
    ) -> None:
        cfg = ExtractionConfig.from_yaml(config) if config is not None else ExtractionConfig()
        from uwf_zonal_extraction.segmentation import SegmentationBundle

        bundle = SegmentationBundle.from_model_dir(
            models_dir,
            tile_size=cfg.segmentation.tile_size,
            stride=cfg.segmentation.stride,
            av_threshold=cfg.segmentation.av_threshold,
        )
        result = extract_caliber_from_image(image, cfg, bundle, laterality=laterality)
        save_extraction_summary(result, output_dir)
        typer.echo(f"Done. Outputs in {output_dir}")
else:  # pragma: no cover
    app = None  # type: ignore[assignment]


if __name__ == "__main__":
    if app is None:
        raise SystemExit("Install with `[cli]` extra: pip install -e '.[cli]'")
    app()
