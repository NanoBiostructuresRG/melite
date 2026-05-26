# CLI Reference

The `melite` console command exposes benchmarking, export, and metadata checks.

## Help and Version

```bash
melite --help
melite run --help
melite export --help
melite --version
```

## Run Benchmarks

Run the configured benchmark:

```bash
melite run
```

Run a fast smoke check:

```bash
melite run --smoke
```

Use a custom TOML configuration:

```bash
melite run --config my_config.toml
```

Enable verbose logs:

```bash
melite run --verbose
```

## Export a Selected Model

Launch interactive row selection:

```bash
melite export
```

Export a specific row:

```bash
melite export --row 0
```

Use custom paths:

```bash
melite export --row 0 --csv output/results.csv --outdir output/
```

Use a custom configuration file:

```bash
melite export --config my_config.toml --row 0
```

## Smoke Guard

Smoke-mode results are marked in `results.csv` and are blocked from export by
default because they are not benchmark-quality. Override intentionally with:

```bash
melite export --row 0 --force
```
