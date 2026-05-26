# CLI Reference

The `mosaic` console command exposes benchmarking, export, and metadata checks.

## Help and Version

```bash
mosaic --help
mosaic run --help
mosaic export --help
mosaic --version
```

## Run Benchmarks

Run the configured benchmark:

```bash
mosaic run
```

Run a fast smoke check:

```bash
mosaic run --smoke
```

Use a custom TOML configuration:

```bash
mosaic run --config my_config.toml
```

Enable verbose logs:

```bash
mosaic run --verbose
```

## Export a Selected Model

Launch interactive row selection:

```bash
mosaic export
```

Export a specific row:

```bash
mosaic export --row 0
```

Use custom paths:

```bash
mosaic export --row 0 --csv output/results.csv --outdir output/
```

Use a custom configuration file:

```bash
mosaic export --config my_config.toml --row 0
```

## Smoke Guard

Smoke-mode results are marked in `results.csv` and are blocked from export by
default because they are not benchmark-quality. Override intentionally with:

```bash
mosaic export --row 0 --force
```
