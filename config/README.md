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
the shared panel-ordering dataset reader and shows the correct first panel as
`Panel 1 (given start)` on its own row. Only the remaining panels are shuffled.
Enter their IDs in reading order, such as `3 2 4 6 5` for a six-panel puzzle;
do not include Panel 1. Set `--output-dir` to save the shown sheets for later
review. Optional `--results-path` JSONL records identify the task as
`order_remaining_given_first`, with `given_display_order: [1]`; both `answer`
and `reference_display_order` contain only the remaining panel IDs.
