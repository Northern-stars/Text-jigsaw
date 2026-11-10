import argparse
import json
import tempfile
import unittest
from pathlib import Path

from config.cli import parse_args_with_config, save_startup_config


class ConfigCliTests(unittest.TestCase):
    def test_config_defaults_satisfy_required_arguments_and_cli_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_file = tmp_path / "cli.json"
            config_file.write_text(
                json.dumps({"defaults": {"input": "data/source", "amount": 4}}),
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            parser.add_argument("--input", type=Path, required=True)
            parser.add_argument("--amount", type=int, default=1)

            args = parse_args_with_config(
                parser,
                ["--config", str(config_file), "--amount", "9"],
                script_file="Solver/example.py",
            )

            self.assertEqual(args.input, Path("data/source"))
            self.assertEqual(args.amount, 9)
            self.assertEqual(args.config_file, str(config_file))

    def test_unknown_config_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "cli.json"
            config_file.write_text(json.dumps({"missing": 1}), encoding="utf-8")
            parser = argparse.ArgumentParser()
            parser.add_argument("--known", type=int, default=0)

            with self.assertRaisesRegex(ValueError, "Unknown config keys"):
                parse_args_with_config(parser, ["--config", str(config_file)])

    def test_startup_config_is_written_into_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            checkpoint_dir = tmp_path / "checkpoint.with.dots"
            args = argparse.Namespace(config_file="config/train.json", epochs=3, output=tmp_path / "out")

            output_path = save_startup_config(args, checkpoint_dir)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(output_path, checkpoint_dir / "startup_config.json")
            self.assertEqual(payload["arguments"]["epochs"], 3)
            self.assertEqual(payload["arguments"]["output"], str(tmp_path / "out"))
            self.assertEqual(payload["config_file"], "config/train.json")


if __name__ == "__main__":
    unittest.main()
