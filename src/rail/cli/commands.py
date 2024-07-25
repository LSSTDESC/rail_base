import os
import click

from rail.core import __version__
from rail.cli import options, scripts
from rail.interfaces.pz_factory import PZFactory
from rail.interfaces.tool_factory import ToolFactory
import ceci


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


@cli.command()
@options.stage_name()
@options.stage_class()
@options.stage_module()
@options.model_file()
@options.dry_run()
@options.input_file()
def estimate(stage_name, stage_class, stage_module, model_file, dry_run, input_file):
    """Run a pz estimation stage"""
    stage = PZFactory.build_cat_estimator_stage(
        stage_name=stage_name,
        class_name=stage_class,
        module_name=stage_module,
        model_path=model_file,
        data_path='dummy.in',
    )

    output = PZFactory.run_cat_estimator_stage(
        stage,
        data_path=input_file,
    )


@options.pipeline_class()
@options.output_yaml()
@options.catalog_tag()
@options.stages_config()
@options.outdir()
@options.inputs()
def build_pipe(pipeline_class, output_yaml, catalog_tag, stages_config, outdir, inputs):
    """Build a pipeline yaml file"""
    input_dict = {}
    for input_ in inputs:
        tokens = input_.split('=')
        assert len(tokens) == 2
        input_dict[tokens[0]] = tokens[1]    
    scripts.build_pipeline(pipeline_class,  output_yaml, catalog_tag, input_dict, stages_config, outdir)
    return 0


@cli.command()
@options.pipeline_yaml()
@options.stage_name()
@options.dry_run()
@options.inputs()
def run_stage(pipeline_yaml, stage_name, dry_run, inputs):
    """Run a pipeline stage"""
    pipe = ceci.Pipeline.read(pipeline_yaml)
    input_dict = {}
    for input_ in inputs:
        tokens = input_.split('=')
        assert len(tokens) == 2
        input_dict[tokens[0]] = tokens[1]
    com = pipe.generate_stage_command(stage_name, **input_dict)
    if dry_run:
        print(com)
    else:
        os.system(com)
    return 0


@cli.command()
@options.stage_name()
@options.stage_class()
@options.stage_module()
@options.dry_run()
@options.input_file()
def run_tool(stage_name, stage_class, stage_module, dry_run, input_file):
    """Run a pz estimation stage"""
    stage = ToolFactory.build_tool_stage(
        stage_name=stage_name,
        class_name=stage_class,
        module_name=stage_module,
        data_path='dummy.in',
    )

    output = ToolFactory.run_tool_stage(
        stage,
        data_path=input_file,
    )

