from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m bts_nvs",
        description=(
            "BTS NVS pipeline. Use a submodule: audit, ingest, holdout, prepare, prepare_dataset, "
            "train, render, nearest_view, evaluate, score_submission, submit, feedback, "
            "validate_submission, or ledger."
        ),
    )
    parser.print_help()


if __name__ == "__main__":
    main()
