# CLI configuration

Each command-line entry point reads defaults from the matching JSON file under
this directory. For example, `Solver/train_rl.py` reads
`config/Solver/train_rl.json`. Values in the JSON object's `defaults` mapping
use argparse destination names such as `batch_size`.

Command-line options override configured defaults. Use `--config PATH` to load
a different JSON file. A required option can be supplied by the config; leave
it required on the command line when its configured value is `null`.

Training scripts save the resolved arguments to `startup_config.json` in the
checkpoint directory whenever they write a checkpoint. The snapshot records
the selected config file and original command line as well as the effective
argument values.

Run the interactive panel ordering benchmark with
`python Solver/human_benchmark.py --dataset-dir <ordering-dataset>`. It uses
the shared panel-ordering dataset reader, opens each randomized panel sheet,
and accepts a sequence such as `3 1 2 4 5 6` in reading order. Set
`--output-dir` to save the shown sheets for later review.
