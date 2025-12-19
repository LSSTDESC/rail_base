"""
Functions to create rst stubs of api documentation for the rail.interactive module.
"""

from pathlib import Path

import rail
import rail.interactive
from rail.utils.interactive.base_utils import STAGE_NAMES, _get_virtual_submodule_names

INTERACTIVE_RST = """
.. _page-interactive-api:

***********
Interactive
***********

{extra_content}

.. toctree::
   :maxdepth: 4

   {children}
"""
NAMESPACE_RST = """
{name} namespace
{underline}==========

{extra_content}

.. py:module:: {name}

.. toctree::
   :maxdepth: 4
   :caption: {name}

   {children}
"""
MODULE_RST = """
{name} module
{underline}-------

{extra_content}

.. automodule:: {name}
   :members:
   :undoc-members:
   :imported-members:
   :show-inheritance:
"""

recursive = {}


def add_to_recursive(name: str) -> None:
    """Function used to populate a recursive dict of rail.interactive submodules"""

    parent = recursive
    for p in name.split(".")[2:]:
        if p in parent:
            parent = parent[p]
        else:
            parent[p] = {}


def get_children(parts: list[str]) -> list[str]:
    """Get a list of submodules of some specific rail.interactive module"""

    children = recursive
    for p in parts:
        children = children[p]
    return list(children.keys())


def get_extra_content(name: str, docs_path: Path) -> str:
    """Fetch any descriptive rst content written for this module"""
    extra_content_dir = docs_path / "source/interactive_api_content"
    content_path = extra_content_dir / (name + ".rst")
    if content_path.exists():
        return content_path.read_text()

    return ""


def make_rst(name: str, children: list[str], docs_path: Path) -> None:
    """Write an rst file for a rail.interactive module"""

    path = docs_path / f"api/{name}.rst"

    if len(children) > 0:

        rst = NAMESPACE_RST.format(
            name=name,
            underline="=" * len(name),
            children="\n   ".join(children),
            extra_content=get_extra_content(name, docs_path),
        )
    else:
        rst = MODULE_RST.format(
            name=name,
            underline="-" * len(name),
            extra_content=get_extra_content(name, docs_path),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {name}.rst")
    path.write_text(rst.strip())


def write_interactive_api_rst(docs_path: str) -> None:
    """Write rst api files for the rail.interactive module and associated submodules

    The default RailEnv api writer doesn't understand the dynamically created submodules
    for interactive.
    Additionally, we need the :imported-members: rst directive for autodoc in order to
    actually render the interactive functions.

    Parameters
    ----------
    docs_path : str
        docs directory where Sphinx is run from
    """
    docs_path = Path(docs_path)
    module_names = _get_virtual_submodule_names(rail.interactive, STAGE_NAMES)
    for name in module_names:
        add_to_recursive(name)

    for name in module_names:
        parts = name.split(".")[2:]
        children = [".".join([name, child]) for child in get_children(parts)]
        make_rst(name, children, docs_path)

    # Make a special, separate page for the top-level rail.interactive module
    top_level_path = docs_path / "api_interactive.rst"
    print(f"Writing {top_level_path}")
    top_level_path.write_text(
        INTERACTIVE_RST.format(
            extra_content=get_extra_content("rail.interactive", docs_path),
            children="\n   ".join(
                [f"api/rail.interactive.{namespace}" for namespace in recursive]
            ),
        ).strip()
    )
