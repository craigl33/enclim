# enclim

Computes ensemble-median climate metrics (hot-day counts, heat-index day
counts, cumulative degree-days) from the World Bank CCKP CMIP6 daily
collection (`s3://wbg-cckp/data/cmip6-daily-x0.25/`), streaming directly
from S3.

## Project structure

```
enclim/
├── config.py        EnsembleConfig - loads & validates ensemble_config.yaml
├── manifest.py       CCKPManifest  - S3 model/run/scenario discovery + cache
├── metrics.py        pure functions: heat index, day-count/CDD grids, naming
├── io.py             CCKPDailyLoader - streams one variable/dataset/year from S3
├── processor.py      ModelYearProcessor - load once, compute all metrics
├── ensemble.py        EnsembleBuilder - per-metric ensemble median
├── writer.py          NetCDFWriter - zlib-compressed NetCDF output
└── run_ensemble.py    CLI entrypoint
config/
└── ensemble_config.yaml
```

Everything that controls a run - source location, which variables to
manifest, model selection, scenarios/years, metrics, output paths, log
level - is in `config/ensemble_config.yaml`. Nothing is hardcoded in the
package.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m enclim.run_ensemble --config config/ensemble_config.yaml
```

This processes every `(scenario, year)` pair defined in `scenarios:`. To run
a single combination (useful for testing, and the same flags used per AWS
Batch array task):

```bash
python -m enclim.run_ensemble --config config/ensemble_config.yaml \
    --scenario historical --year 2014
```

First run builds and caches the S3 manifest to
`manifest.cache_path` (one `ListObjectsV2` per variable in
`manifest.variables`); subsequent runs reuse the cache until
`manifest.refresh: true`.

## Small-scale test (5 models, 1 year)

The shipped config is already scoped to `historical` / 2014. To restrict to
~5 models instead of the full auto-discovered ensemble, set `models` to an
explicit list of `{model}-{run}-{scenario}` dataset strings, e.g.:

```yaml
models:
  - access-cm2-r1i1p1f1-historical
  - canesm5-r1i1p1f1-historical
  - cnrm-cm6-1-r1i1p1f2-historical
  - ec-earth3-r1i1p1f1-historical
  - mpi-esm1-2-lr-r1i1p1f1-historical
```

`models: auto` (the default) uses every dataset the manifest finds, per
metric (see "Per-metric ensemble membership" below).

## Per-metric ensemble membership

- `hd` (hot days) needs `tasmax` only.
- `hi` (heat-index days) needs `tasmax` + `hurs`.
- `cdd` needs `tas`; with `humidity: true`, also `hurs`.

For each metric, `CCKPManifest.models_for_metric()` resolves the set of
datasets that have *all* the variables that metric needs, for the requested
scenario. A model missing `hurs` still contributes to `hd`, but not to `hi`.

## Docker

```bash
docker build -t cckp-ensemble .

docker run --rm -v "$(pwd)/output:/app/output" cckp-ensemble \
    --config config/ensemble_config.yaml --scenario historical --year 2014
```

The image bundles the package and the default config; mount a different
config with `-v ./my_config.yaml:/app/config/ensemble_config.yaml`.

## AWS deployment (Batch array jobs)

The recommended deployment is **AWS Batch with array jobs**, not Lambda:
each `(scenario, year)` is independent, output files are small, but reading
~25 models x 2-3 variables of ~900MB streamed NetCDF per scenario/year can
exceed Lambda's 15-minute / 10GB limits.

- Push the image built above to ECR.
- Set the Batch job's **array size** to `len(EnsembleConfig(...).scenario_years())`.
- Each array task automatically picks its `(scenario, year)` via the
  `AWS_BATCH_JOB_ARRAY_INDEX` env var that Batch sets - no extra wiring
  needed (see `run_ensemble.py:_resolve_targets`).
- `output.dir` is a local path inside the container. To land results in S3,
  either mount an S3-backed volume there, or add a small post-step (e.g. an
  `aws s3 cp` in the entrypoint, or a thin subclass of `NetCDFWriter`) -
  this doesn't require changing `EnsembleBuilder`/`NetCDFWriter`.

Local Docker runs and Batch runs use the exact same image and entrypoint;
only the config and environment differ.
