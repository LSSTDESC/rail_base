"""
When running this you may also want to run:

python wumpydoc.py
pytest tests/interactive
[this file]

python examples/interactive/interactive.py
"""

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import black
import isort
import pytest

from rail.core.introspection import RailEnv
from rail.estimation.algos.cc_yaw import *  # workaround for yaw not otherwise showing up
from rail.utils.interactive.initialize_utils import _initialize_interactive_module

interactive_modules = [
    "calib",
    "creation.degraders",
    "creation.engines",
    "estimation.algos",
    "evaluation",
    "evaluation.metrics",
    "tools",
]
interactive_path = Path("src/rail/interactive")


@dataclass
class InteractiveModule:
    subfolder: str = ""
    docstring: str = ""
    absolute_imports: list[str] = field(default_factory=list)
    relative_imports: list[str] = field(default_factory=list)
    code: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return interactive_path / self.subfolder / "__init__.py"

    def __str__(self) -> str:
        docstring = ""
        if len(self.docstring) > 0:
            docstring += f'"""{self.docstring}"""\n'
        imports = "\n".join(
            self.absolute_imports
            + [f"from . import {module}" for module in self.relative_imports]
        )

        return f"{docstring}{imports}\n\n" + "\n".join(self.code)


def check_rail_packages() -> list[str]:
    module_to_package = {
        "rail": "pz-rail",
        "rail.astro_tools": "pz-rail-astro-tools",
        "rail.bpz": "pz-rail-bpz",
        "rail.calib": "pz-rail-calib",
        "rail.cmnn": "pz-rail-cmnn",
        "rail.delight": "pz-rail-delight",
        "rail.dnf": "pz-rail-dnf",
        "rail.dsps": "pz-rail-dsps",
        "rail.flexzboost": "pz-rail-flexzboost",
        "rail.fsps": "pz-rail-fsps",
        "rail.gpz": "pz-rail-gpz-v1",
        # "rail.inception": "pz-rail-inception", # not ready for general use
        "rail.lephare": "pz-rail-lephare",
        "rail.lib_gp_comp": "pz-rail-lib_gp_comp",
        "rail.pzflow": "pz-rail-pzflow",
        # "rail.shire": "pz-rail-shire", # not ready for general use
        "rail.sklearn": "pz-rail-sklearn",
        "rail.som": "pz-rail-som",
        "rail.sompz": "pz-rail-sompz",
        "rail.rail_tpz": "pz-rail-tpz",
        "rail.yaw_rail": "pz-rail-yaw",
    }
    package_info = RailEnv.list_rail_packages()

    # remove the rail.X submodules from the list of packages, that don't represent their
    # own independent PyPI packages
    # if, in the future, the `module_to_package[name]` line in this function's return
    # statement throws a KeyError one of two things needs to happen:
    # 1. this is a new pz-rail- package that should be added to the above
    #    `module_to_package` dict
    # 2. this is a new unaccounted for module, that should be added to the below list of
    #    items that get deleted
    for non_rail_package in ["rail.hub", "rail._pipelines"]:
        if non_rail_package in package_info:
            del package_info[non_rail_package]

    rail_base_path = package_info["rail.core"][0].path
    return ["pz-rail"] + [
        module_to_package[name]
        for name, info in package_info.items()
        if info[0].path != rail_base_path
    ]


def write_modules() -> None:
    all_modules: dict[str, InteractiveModule] = {}

    all_modules["."] = InteractiveModule(
        docstring="Needed to run `import rail.interactive`",
    )

    # sort to make sure we do parents first
    for module_name in sorted(interactive_modules):
        portions = module_name.split(".")

        # add import statement to rail.interactive
        if portions[0] not in all_modules["."].relative_imports:
            all_modules["."].relative_imports.append(portions[0])

        # if this isn't a top level (i.e., there's one nesting, like
        # creation.degraders), import degraders from creation
        if len(portions) == 2:
            if portions[0] not in all_modules:
                all_modules[portions[0]] = InteractiveModule(subfolder=portions[0])
            if portions[1] not in all_modules[portions[0]].relative_imports:
                all_modules[portions[0]].relative_imports.append(portions[1])

        # create the lowest level initialization
        all_modules[module_name] = InteractiveModule(
            subfolder=module_name.replace(".", "/"),
            docstring=f"Module docstring for interactive {portions[-1]}",
            absolute_imports=[
                "from rail.utils.interactive.initialize_utils import _initialize_interactive_module"
            ],
            code=["_initialize_interactive_module(__name__)"],
        )

    for module in all_modules.values():
        module.path.parent.mkdir(parents=True, exist_ok=True)
        module.path.write_text(str(module))

        # INTERACTIVE-DO: Make this a utility since it's used here and in stubs
        black.format_file_in_place(
            module.path,
            fast=False,
            mode=black.Mode(is_pyi=True),
            write_back=black.WriteBack.YES,
        )
        isort.api.sort_file(module.path, quiet=True, profile="black")

        print(f"Created {str(module.path)}")


def write_stubs() -> None:
    for module in interactive_modules:
        module = "rail.interactive." + module
        importlib.import_module(module)
        _initialize_interactive_module(module, write_stubs=True)


if __name__ == "__main__":
    print("Running for rail packages:\n\t" + "\n\t".join(check_rail_packages()))

    if pytest.main(["tests/interactive"]) != 0:
        sys.exit()
    # add function to delete everything in interactive?
    write_modules()
    write_stubs()
