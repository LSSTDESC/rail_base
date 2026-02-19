"""
When running this you may also want to run python wumpydoc.py, which runs a subset of
the numpydoc tests against the docstrings of RAILStages.

The script deletes all the stub files in the interactive folder, then re-creates them.
It returns an exit code based on the comparison of the re-generated and original stub
files.
0 indicates no change
1 is an error somewhere in the script
2 indicates files have changed

Note that this script does update the files, so if you run it once (resulting in exit
code 2), running it again immediately will show an exit code of 0.
Consider using `git diff` or `git diff --name-status` if you need to see a summary of
changes to stub files since the last commit.

Before it deletes the current, or creates the new stub files, it does run the
interactive tests, which run against the .py (NOT .pyi) files. This is to ensure the .py
files are well-formatted such that the .pyi generation will succeed.
"""

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from rail.core.introspection import RailEnv
from rail.utils.interactive.base_utils import _write_formatted_file
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
rail_base = Path(__file__).parent
interactive_path = rail_base / "src/rail/interactive"


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
        _write_formatted_file(module.path, str(module))


def write_stubs() -> None:
    for module in interactive_modules:
        stub_directory = InteractiveModule(
            subfolder=module.replace(".", "/")
        ).path.parent
        full_module = "rail.interactive." + module
        importlib.import_module(full_module)
        _initialize_interactive_module(
            full_module, write_stubs=True, stub_directory=stub_directory
        )


def store_pyi(delete: bool) -> dict[Path, str | None]:
    stubs: dict[Path, str | None] = {}
    for path in interactive_path.glob("**/*.pyi"):
        stubs[path] = path.read_text()
        if delete:
            path.unlink()
    return stubs


def compare_pyi(old_stubs: dict[Path, str | None]) -> int:
    new_stubs = store_pyi(delete=False)
    stub_paths = set([*old_stubs, *new_stubs])

    changes = []
    for path in stub_paths:
        # if not path.exists(): # equivalent to being a key in old, but not in new
        if (path not in new_stubs) and (path in old_stubs):
            changes.append((path, "deleted"))
            continue
        if (path in new_stubs) and (not path in old_stubs):
            changes.append((path, "added"))
            continue
        if old_stubs[path] != new_stubs[path]:
            changes.append((path, "modified"))

    if len(changes) > 0:
        print()
        for path, change in changes:
            print(f"{str(path)} was {change}")
        return 2
    return 0


if __name__ == "__main__":
    print("\nRunning for rail packages:\n\t" + "\n\t".join(check_rail_packages()))

    pytest_exit = pytest.main(["tests/interactive"])
    if pytest_exit != 0:
        sys.exit(pytest_exit)

    old_stubs = store_pyi(delete=True)

    write_modules()
    write_stubs()

    sys.exit(compare_pyi(old_stubs))
