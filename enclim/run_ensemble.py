"""CLI entrypoint.

Usage:
    python -m enclim.run_ensemble --config config/ensemble_config.yaml
    python -m enclim.run_ensemble --config config/ensemble_config.yaml --scenario historical --year 2014

With no --scenario/--year, processes every (scenario, year) defined in
config.scenarios, in order.

If the AWS_BATCH_JOB_ARRAY_INDEX environment variable is set (AWS Batch sets
this automatically for array jobs), only that single index from the full
scenario/year sweep is processed. This is the hook for a
one-container-per-scenario-year deployment on AWS Batch: set the array job's
size to len(config.scenario_years()) and each task picks its own target.
"""

import argparse
import logging
import os
import sys

from enclim.config import EnsembleConfig
from enclim.ensemble import EnsembleBuilder
from enclim.exceptions import ConfigError, EnsembleError
from enclim.writer import NetCDFWriter

logger = logging.getLogger(__name__)


def _resolve_targets(config: EnsembleConfig, scenario: str, year: int) -> list:
    if scenario and year:
        return [(scenario, year)]
    if scenario or year:
        raise ConfigError("--scenario and --year must be given together")

    targets = config.scenario_years()

    array_index = os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX')
    if array_index is not None:
        try:
            idx = int(array_index)
        except ValueError:
            raise ConfigError(f"AWS_BATCH_JOB_ARRAY_INDEX={array_index!r} is not an integer")
        if not (0 <= idx < len(targets)):
            raise ConfigError(
                f"AWS_BATCH_JOB_ARRAY_INDEX={idx} out of range for "
                f"{len(targets)} scenario/year combination(s) in config.scenarios"
            )
        targets = [targets[idx]]

    return targets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compute CCKP CMIP6 ensemble-median climate metrics")
    parser.add_argument('--config', required=True, help='Path to ensemble_config.yaml')
    parser.add_argument('--scenario', help='Process a single scenario (must be paired with --year)')
    parser.add_argument('--year', type=int, help='Process a single year (must be paired with --scenario)')
    args = parser.parse_args(argv)

    config = EnsembleConfig(args.config)
    logging.basicConfig(level=config.log_level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    targets = _resolve_targets(config, args.scenario, args.year)
    logger.info("Processing %d scenario/year combination(s): %s", len(targets), targets)

    builder = EnsembleBuilder(config)
    writer = NetCDFWriter(config)

    for scenario, year in targets:
        logger.info("=== %s / %s ===", scenario, year)
        ensemble = builder.build(scenario, year)
        writer.write(ensemble, scenario, year)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ConfigError, EnsembleError) as exc:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error(str(exc))
        sys.exit(1)
