from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m bts_nvs",
        description=(
            "BTS NVS baseline. Use a submodule: prepare, prepare_dataset, train, "
            "render, nearest_view, evaluate, or package."
        ),
    )
    parser.print_help()


if __name__ == "__main__":
    main()
