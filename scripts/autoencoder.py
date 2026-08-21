"""Document the autoencoder notebook as a script entry point."""

from pathlib import Path


def main() -> None:
    """Point users to the source notebook and its required data input."""
    print(f"Run the autoencoder experiment from {Path('autoencoder.ipynb')}")


if __name__ == "__main__":
    main()
