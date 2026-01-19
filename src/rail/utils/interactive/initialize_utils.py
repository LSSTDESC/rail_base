"""
Utility functions for the rail.interactive module.
"""

import collections
import functools
import inspect
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rail.core.data import DataHandle
from rail.core.stage import RailStage
from rail.utils.interactive.base_utils import (
    GLOBAL_INTERACTIVE_PARAMETERS,
    STAGE_NAMES,
    _get_stage_definition,
    _get_stage_module,
    _get_virtual_submodule_names,
)
from rail.utils.interactive.docstring_utils import (
    DOCSTRING_INDENTATION,
    create_interactive_docstring,
)


@dataclass
class VirtualModule:
    """A wrapper class to hold a code-created module/namespace, as well as it's
    relationships to other modules"""

    module: types.ModuleType
    children: dict[str, str]
    parent: str


def _interactive_factory(
    rail_stage: type[RailStage],
    function_input_is_wrapped: bool,
    function_requires_input: bool,
    **kwargs,
) -> Any:
    """Create the actual interactive function for a RAIL stage

    Parameters
    ----------
    rail_stage : type[RailStage]
        The stage being operated on
    function_input_is_wrapped : bool
        Whether or not the interactive version of the entrypoint function has wrapped up
        multiple positional parameters into a dictionary, that needs to be unwrapped
        when passing it onwards
    function_requires_input : bool
        Whether or not the interactive version of the function needs to have an `input` kwarg

    Returns
    -------
    Any
        This function returns the result of calling the stage's entrypoint function
        (after calling make_stage)
    """
    entrypoint_function_name = rail_stage.entrypoint_function

    # extract the input kwarg, and turn it into the appropriate DataHandle
    stage_has_input = "input" in kwargs
    if stage_has_input:
        entrypoint_inputs = kwargs.pop("input")
    elif function_requires_input:
        raise ValueError(f"{entrypoint_function_name} requires an `input` kwarg")

    for key, value in GLOBAL_INTERACTIVE_PARAMETERS.items():
        if key in kwargs:
            raise ValueError(f"In rail.interactive, {key} is set to {value}")

    instance = rail_stage.make_stage(**kwargs, **GLOBAL_INTERACTIVE_PARAMETERS)
    entrypoint_function: Callable = getattr(instance, entrypoint_function_name)

    if function_input_is_wrapped:
        if (
            not stage_has_input
        ):  # INTERACTIVE-DO: probably redundant with function_requires_input check above
            raise ValueError(f"{entrypoint_function_name} requires an `input` kwarg")
        if not isinstance(
            entrypoint_inputs, dict  # pylint: disable=possibly-used-before-assignment
        ):
            raise ValueError(
                f"input parameter for {entrypoint_function_name} must be a dictionary"
            )
        output = entrypoint_function(**entrypoint_inputs, **kwargs)
    elif stage_has_input:
        output = entrypoint_function(entrypoint_inputs, **kwargs)
    else:
        output = entrypoint_function(**kwargs)

    # convert output FROM a DataHandle into pure data
    output_info = rail_stage.outputs  # list of (tag, class)

    # one ceci output
    if len(output_info) == 1:
        tag, class_ = output_info[0]
        return {tag: _unpack_output_handle(class_, output)}

    # multiple ceci outputs, but only one actual output
    if isinstance(output, DataHandle):
        return {"output": _unpack_output_handle(DataHandle, output)}

    # multiple ceci outputs
    return {
        tag: _unpack_output_handle(class_, output[tag])
        for tag, class_ in rail_stage.outputs
    }


def _unpack_output_handle(
    output_class: type[DataHandle], output_content: DataHandle
) -> Any:
    """Unpack the DataHandle returned by a RailStage's entrypoint function to access the
    actual data behind it. Functionized for potential expansion if ever a stage returns
    something other than a DataHandle, or a new type of Handle that needs to be
    specially unpacked.

    Parameters
    ----------
    output_class : type[DataHandle]
        The type of handle being returned, as specified in the ceci `outputs` property
        of a RailStage class
    output_content : DataHandle
        Whatever part of the RailStage outputs is matched to this `output_class` (see
        usage in `_interactive_factory`)

    Returns
    -------
    Any
        The unpacked data

    Raises
    ------
    ValueError
        Raised if the `output_class` is not explicitly handled
    """
    if isinstance(output_content, DataHandle):
        return output_content.data

    raise ValueError(f"Class {output_class} not implemented")


def _create_virtual_submodules(
    module: types.ModuleType, stage_names: list[str]
) -> dict[str, VirtualModule]:
    """Create all the relevant child namespaces of `module` in which the interactive
    functions will live.

    Parameters
    ----------
    module : types.ModuleType
        The parent module
    stage_names : list[str]
        RAIL stages which will have interactive functions

    Returns
    -------
    dict[str, VirtualModule]
        All of the created namespaces
    """
    recursive_module_names = _get_virtual_submodule_names(module, stage_names)

    # populate a dictionary of virtual modules, keeping track of relationships
    virtual_modules = {}
    for virtual_module_name in recursive_module_names:
        virtual_modules[virtual_module_name] = VirtualModule(
            module=types.ModuleType(virtual_module_name),
            children=[],
            parent=module.__name__,
        )

        # assign parent/child relationships if necessary
        if len(virtual_module_name.split(".")) != len(module.__name__.split(".")) + 1:
            parent = virtual_modules[
                virtual_module_name[: virtual_module_name.rindex(".")]
            ]
            virtual_modules[virtual_module_name].parent = parent.module.__name__
            parent.children.append(virtual_modules[virtual_module_name].module.__name__)

    # attatch virtual modules to their correct parents
    for name, virtual_module in virtual_modules.items():
        if virtual_module.parent == module.__name__:
            setattr(module, name.split(".")[-1], virtual_module.module)
        else:
            parent = virtual_modules[virtual_module.parent]
            setattr(parent.module, name.split(".")[-1], virtual_module.module)

    return virtual_modules


def _attatch_interactive_function(
    stage_module_dict: dict[str, VirtualModule], stage_name: str
) -> None:
    """Create a wrapper function for a RAIL stage, and assign it to the appropriate namespace.

    Parameters
    ----------
    stage_module_dict : dict[str, VirtualModule]
        The set of namespaces that may be relevant
    stage_name : str
        The name of the RAIL stage class
    """
    stage_definition = _get_stage_definition(stage_name)
    function_name = stage_definition.interactive_function
    virtual_module_name = _get_stage_module(stage_name, interactive=True)
    virtual_module = stage_module_dict[virtual_module_name]

    # create the function
    docstring, function_input_is_wrapped, function_requires_input = (
        create_interactive_docstring(stage_name)
    )
    created_function: Callable = functools.partial(
        _interactive_factory,
        stage_definition,
        function_input_is_wrapped,
        function_requires_input,
    )
    created_function.__doc__ = docstring
    created_function.__module__ = virtual_module.module

    setattr(virtual_module.module, function_name, created_function)


def _get_stub_path(
    stub_directory: Path,
    virtual_module_name: str | None = None,
) -> Path:
    """Get the final path to where a stub file should live

    Parameters
    ----------
    stub_directory : Path
        Parent directory for stub files
    virtual_module_name : str | None, optional
        The virtual module that's being written about if applicable, by default None

    Returns
    -------
    Path
        Path to the stub file
    """
    if virtual_module_name is not None:
        final_segment = virtual_module_name.split(".")[-1]
        filename = f"{final_segment}.pyi"
        return stub_directory / filename

    return stub_directory / "__init__.pyi"


def _write_stubs(
    module: types.ModuleType,
    stage_names: list[str],
    virtual_modules: dict[str, VirtualModule],
) -> None:
    """Write stub (type hint) files for interactive functions.

    Multiple runs of this function should result in identical stub files, and will not
    trigger a modification as read by Git.

    Parameters
    ----------
    module : types.ModuleType
        The RAIL module that the interactive functions are children of
    stage_names : list[str]
        Names of RAIL stages that have interactive functions
    virtual_modules : dict[str, VirtualModule]
        The code-created namespaces that the interactive functions live in
    """
    # formatters used for the stub files, imported here since these are only required
    # for developers
    # pylint: disable=import-outside-toplevel
    import black
    import isort

    stub_files = collections.defaultdict(list)
    stub_directory = Path(module.__path__[0])

    # import top level virtual modules
    stub = stub_files[_get_stub_path(stub_directory)]
    for vm_name, vm_VM in virtual_modules.items():
        if vm_VM.parent == module.__name__:
            stub.append(f"from . import {vm_name.split('.')[-1]}")

    # import child virtual modules from their parents
    for vm_name, vm_VM in virtual_modules.items():
        stub = stub_files[_get_stub_path(stub_directory, virtual_module_name=vm_name)]
        for child_module_name in vm_VM.children:
            stub.append(f"from . import {child_module_name.split('.')[-1]}")

    # add function stubs
    for stage_name in stage_names:
        virtual_module_name = _get_stage_module(stage_name, interactive=True)
        stage_definition = _get_stage_definition(stage_name)

        function_name = stage_definition.interactive_function
        virtual_module = virtual_modules[virtual_module_name].module
        created_function = getattr(virtual_module, function_name)

        signature = inspect.signature(created_function)
        buffer = " " * DOCSTRING_INDENTATION + '"""'
        docstring = f"{buffer}\n{created_function.__doc__}\n{buffer}"

        stub_path = _get_stub_path(
            stub_directory, virtual_module_name=virtual_module_name
        )
        stub_files[stub_path].append(
            f"def {function_name}{str(signature)}:\n{docstring}"
        )

    # merge lists to strings
    stub_files_strings: dict[Path, str] = {}
    for path, content in stub_files.items():
        joined_content = "\n".join(content)
        if ") -> Any:" in joined_content:
            joined_content = f"from typing import Any\n\n{joined_content}"
        stub_files_strings[path] = joined_content

    # write and format the file
    for path, content in stub_files_strings.items():
        path.write_text(content)
        black.format_file_in_place(
            path,
            fast=False,
            mode=black.Mode(is_pyi=True),
            write_back=black.WriteBack.YES,
        )
        isort.api.sort_file(path, quiet=True, profile="black")

        print(f"Created {str(path)}")


def _initialize_interactive_module(
    calling_module_name: str, write_stubs: bool = False
) -> None:
    """Create wrappers for RAIL stages as single-call functions for interactive use.

    Optionally create stub files to improve static analysis of these dynamically created functions.

    Parameters
    ----------
    calling_module_name : str
        The __name__ attribute of the module calling this function.
    write_stubs : bool, optional
        Whether to write stub files, by default False
    """

    calling_module = sys.modules[calling_module_name]

    # filter down all the RAIL stages to only ones which are children of this module
    relevant_stages = [
        stage
        for stage in STAGE_NAMES
        if _get_stage_module(stage, interactive=True).startswith(calling_module_name)
    ]

    virtual_module_dict = _create_virtual_submodules(calling_module, relevant_stages)

    for stage_name in relevant_stages:
        _attatch_interactive_function(virtual_module_dict, stage_name)

    if write_stubs:
        _write_stubs(calling_module, relevant_stages, virtual_module_dict)
