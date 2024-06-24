import enum
from functools import partial
from typing import Any, Type, TypeVar

import click

__all__ = [
    "clear_output",
    "bpz_demo_data",
    "catalog_tag",
    "dry_run",
    "outdir",
    "output_yaml",
    "from_source",
    "model_file",
    "git_mode",
    "pipeline_class",
    "pipeline_yaml",
    "print_all",
    "print_packages",
    "print_namespaces",
    "print_modules",
    "print_tree",
    "print_stages",
    "package_file",
    "skip",
    "stage_class",
    "stage_module",
    "stage_name",
    "stages_config",
    "inputs",
    "verbose_download",
]


class GitMode(enum.Enum):
    """Choose git clone mode"""

    ssh = 0
    https = 1
    cli = 2


EnumType_co = TypeVar("EnumType_co", bound=Type[enum.Enum], covariant=True)


class EnumChoice(click.Choice):
    """A version of click.Choice specialized for enum types"""

    def __init__(self, the_enum: EnumType_co, case_sensitive: bool = True) -> None:
        self._enum = the_enum
        super().__init__(
            list(the_enum.__members__.keys()), case_sensitive=case_sensitive
        )

    def convert(self, value: Any, param, ctx) -> EnumType_co:  # pragma: no cover
        converted_str = super().convert(value, param, ctx)
        return self._enum.__members__[converted_str]


class PartialOption:
    """Wraps click.option with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(
            click.option, *param_decls, cls=partial(click.Option), **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return self._partial(*args, **kwargs)


class PartialArgument:
    """Wraps click.argument with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(
            click.argument, *param_decls, cls=partial(click.Argument), **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return self._partial(*args, **kwargs)


clear_output = PartialOption(
    "--clear-output",
    help="Clear Notebook output",
    is_flag=True,
)

dry_run = PartialOption(
    "--dry-run",
    help="Dry run only",
    is_flag=True,
)


catalog_tag = PartialOption(
    "--catalog_tag",
    default=None,
    help="Type of input catalog, used to determine column names",
)


from_source = PartialOption(
    "--from-source",
    help="Install from source",
    is_flag=True,
)

outdir = PartialOption(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory.",
)

output_yaml = PartialOption(
    "--output_yaml",
    type=click.Path(),
    default=None,
    help="Path for output yaml file",
)

pipeline_class =  PartialOption(
    "--pipeline_class",
    type=str,
    help="Full class name for pipeline, e.g., rail.pipelines.estimation.train_z.TrainZPipeline",
)

model_file =  PartialOption(
    "--model_file",
    type=str,
    help="Model for pz estimation",
)

input_file =  PartialOption(
    "--input_file",
    type=str,
    help="Input data file for pz estimation",
)

pipeline_yaml =  PartialOption(
    "--pipeline_yaml",
    type=click.Path(),
    help="Yaml for that defines pipeline",
)

git_mode = PartialOption(
    "--git-mode",
    type=EnumChoice(GitMode),
    default="ssh",
    help="Git Clone mode",
)

print_all = PartialOption(
    "--print-all",
    help="Print all RAIL information",
    is_flag=True,
)

print_packages = PartialOption(
    "--print-packages",
    help="Print RAIL packages",
    is_flag=True,
)

print_namespaces = PartialOption(
    "--print-namespaces",
    help="Print RAIL namespaces",
    is_flag=True,
)

print_modules = PartialOption(
    "--print-modules",
    help="Print RAIL modules",
    is_flag=True,
)

print_tree = PartialOption(
    "--print-tree",
    help="Print RAIL namespace tree",
    is_flag=True,
)

print_stages = PartialOption(
    "--print-stages",
    help="Print RAIL stages",
    is_flag=True,
)

project_yaml = PartialOption(
    "--project_yaml",
    type=click.Path(),
    default=None,
    help="File with project description",
)

package_file = PartialOption(
    "--package-file",
    type=click.Path(),
    default=None,
    help="File with package",
)

skip = PartialOption(
    "--skip",
    type=click.Path(),
    multiple=True,
    help="Skip files",
)

stage_class = PartialOption(
    "--stage_class",
    type=str,
    help="Name of a pipeline stage python class",
)

stage_module = PartialOption(
    "--stage_module",
    type=str,
    help="Import path for a python module",
)

stage_name = PartialOption(
    "--stage_name",
    type=str,
    help="Name of a pipeline stage",
)

stages_config = PartialOption(
    "--stages_config",
    type=str,
    help="Stage config file",
    default=None,
)

inputs = PartialArgument("inputs", nargs=-1)

verbose_download = PartialOption(
    "-v", "--verbose", help="Verbose output when downloading", is_flag=True
)

bpz_demo_data = PartialOption(
    "--bpz-demo-data",
    help="Download data that is explicitly only for use in the bpz demo and nowhere else"
    "(it is dummy data that will not make sense)",
    is_flag=True,
)
