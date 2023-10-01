"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Clehrity."""


if __name__ == "__main__":
    main(prog_name="clehrity")  # pragma: no cover
