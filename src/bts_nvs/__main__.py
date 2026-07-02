from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m bts_nvs",
        description="BTS NVS baseline. Use a submodule: prepare, train, render, or evaluate.",
    )
    parser.print_help()


if __name__ == "__main__":
    main()
