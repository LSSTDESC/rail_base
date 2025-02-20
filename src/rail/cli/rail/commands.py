import os
import pprint
from typing import Any

import ceci
import click
import yaml

from rail.cli.rail import options, scripts
from rail.core import __version__
from rail.interfaces.pz_factory import PZFactory
from rail.interfaces.tool_factory import ToolFactory
from rail.utils import catalog_utils


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
def render_nb(
    outdir: str,
    clear_output: bool,
    dry_run: bool,
    inputs: list[str],
    skip: list[str],
    **_kwargs: Any,
) -> int:
    """Render jupyter notebooks"""
    return scripts.render_nb(outdir, clear_output, dry_run, inputs, skip)


@cli.command()
@options.outdir(default="..")
@options.git_mode()
@options.dry_run()
@options.package_file()
def clone_source(
    outdir: str,
    git_mode: options.GitMode,
    dry_run: bool,
    package_file: str,
    **_kwargs: Any,
) -> int:
    """Install packages from source"""
    scripts.clone_source(outdir, git_mode, dry_run, package_file)
    return 0


@cli.command()
@options.outdir(default="..")
@options.dry_run()
@options.package_file()
def update_source(outdir: str, dry_run: bool, package_file: str, **_kwargs: Any) -> int:
    """Update packages from source"""
    scripts.update_source(outdir, dry_run, package_file)
    return 0


@cli.command()
@options.outdir(default="..")
@options.dry_run()
@options.from_source()
@options.package_file()
def install(
    outdir: str, dry_run: bool, from_source: bool, package_file: str, **_kwargs: Any
) -> int:
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
def info(**kwargs: Any) -> int:
    """Print information about the rail ecosystem"""
    scripts.info(**kwargs)
    return 0


@cli.command()
@options.bpz_demo_data()
@options.verbose_download()
def get_data(verbose: bool, **kwargs: Any) -> int:  # pragma: no cover
    """Downloads data from NERSC (if not already found)"""
    scripts.get_data(verbose, **kwargs)
    return 0


@cli.command()
@options.stage_name()
@options.stage_class()
@options.stage_module()
@options.stages_config()
@options.model_file()
@options.catalog_tag()
@options.dry_run()
@options.input_file()
@options.params()
def estimate(
    stage_name: str,
    stage_class: str,
    stage_module: str,
    stages_config: str | None,
    model_file: str,
    catalog_tag: str,
    dry_run: bool,
    input_file: str,
    params: dict,
) -> int:  # pragma: no cover
    """Run a pz estimation stage"""
    if catalog_tag:
        catalog_utils.apply_defaults(catalog_tag)

    if stages_config not in [None, "none", "None"]:
        assert stages_config is not None
        with open(stages_config, "r", encoding="utf-8") as fin:
            config_data = yaml.safe_load(fin)
            if stage_name in config_data:
                kwargs = config_data[stage_name]
            elif isinstance(config_data, dict):
                kwargs = config_data
            else:
                raise ValueError(
                    f"Config file {stages_config} is not properly constructed"
                )
    else:
        kwargs = {}

    kwargs.update(**options.args_to_dict(params))

    stage = PZFactory.build_cat_estimator_stage(
        stage_name=stage_name,
        class_name=stage_class,
        module_name=stage_module,
        model_path=model_file,
        data_path="dummy.in",
        **kwargs,
    )

    if dry_run:
        print(f"Running stage {stage.name} of type {type(stage)} on {input_file}")
        print("Stage config:   ")
        print_dict = stage.config.to_dict().copy()
        pprint.pprint(print_dict)
        return 0

    _output = PZFactory.run_cat_estimator_stage(
        stage,
        data_path=input_file,
    )
    return 0


@cli.command()
@options.pipeline_class()
@options.output_yaml()
@options.catalog_tag()
@options.stages_config()
@options.outdir()
@options.inputs()
def build_pipe(
    pipeline_class: str,
    output_yaml: str,
    catalog_tag: str,
    stages_config: dict,
    outdir: str,
    inputs: dict[str, str],
) -> int:  # pragma: no cover
    """Build a pipeline yaml file"""
    input_dict = {}
    for input_ in inputs:
        tokens = input_.split("=")
        assert len(tokens) == 2
        input_dict[tokens[0]] = tokens[1]
    scripts.build_pipeline(
        pipeline_class, output_yaml, catalog_tag, input_dict, stages_config, outdir
    )
    return 0


@cli.command()
@options.pipeline_yaml()
@options.stage_name()
@options.dry_run()
@options.inputs()
def run_stage(
    pipeline_yaml: str, stage_name: str, dry_run: bool, inputs: dict[str, str]
) -> int:  # pragma: no cover
    """Run a pipeline stage"""
    pipe = ceci.Pipeline.read(pipeline_yaml)
    input_dict = {}
    for input_ in inputs:
        tokens = input_.split("=")
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
def run_tool(
    stage_name: str, stage_class: str, stage_module: str, dry_run: bool, input_file: str
) -> int:  # pragma: no cover
    """Run a pz estimation stage"""
    stage = ToolFactory.build_tool_stage(
        stage_name=stage_name,
        class_name=stage_class,
        module_name=stage_module,
        data_path="dummy.in",
    )

    if dry_run:
        print(f"Running stage {stage.name} of type {type(stage)} on {input_file}")
        print("Stage config:   ")
        print_dict = stage.config.to_dict().copy()
        pprint.pprint(print_dict)
        return 0

    _output = ToolFactory.run_tool_stage(
        stage,
        data_path=input_file,
    )

    return 0
