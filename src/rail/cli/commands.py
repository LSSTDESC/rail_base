import click

from rail.core import __version__
from rail.cli import options, scripts


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """RAIL utility scripts"""


@cli.command()
@options.outdir(default="docs")
@options.clear_output()
@options.dry_run()
@options.inputs()
@options.skip()
def render_nb(outdir, clear_output, dry_run, inputs, skip, **_kwargs):
    """Render jupyter notebooks"""
    return scripts.render_nb(outdir, clear_output, dry_run, inputs, skip)


@cli.command()
@options.outdir(default="..")
@options.git_mode()
@options.dry_run()
@options.package_file()
def clone_source(outdir, git_mode, dry_run, package_file, **_kwargs):
    """Install packages from source"""
    scripts.clone_source(outdir, git_mode, dry_run, package_file)
    return 0


@cli.command()
@options.outdir(default="..")
@options.dry_run()
@options.package_file()
def update_source(outdir, dry_run, package_file, **_kwargs):
    """Update packages from source"""
    scripts.update_source(outdir, dry_run, package_file)
    return 0


@cli.command()
@options.outdir(default="..")
@options.dry_run()
@options.from_source()
@options.package_file()
def install(outdir, dry_run, from_source, package_file, **_kwargs):
    """Install rail packages one by one, to be fault tolerant"""
    scripts.install(outdir, from_source, dry_run, package_file)
    return 0


@cli.command()
@options.outdir(default="..")
@options.print_all()
@options.print_packages()
@options.print_namespaces()
@options.print_modules()
@options.print_tree()
@options.print_stages()
def info(**kwargs):
    """Print information about the rail ecosystem"""
    scripts.info(**kwargs)
    return 0


@cli.command()
@options.bpz_demo_data()
@options.verbose_download()
def get_data(verbose, **kwargs):
    """Downloads data from NERSC (if not already found)"""
    scripts.get_data(verbose, **kwargs)
    return 0
