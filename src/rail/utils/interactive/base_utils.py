"""Low-level utility functions used by the rest of the interactive utils"""

from types import ModuleType

import rail.stages
from rail.core import RailEnv
from rail.core.stage import RailStage

rail.stages.import_and_attach_all(silent=True)

STAGE_NAMES = list(RailStage.pipeline_stages.keys())
STAGE_NAMES = sorted(
    [
        i
        for i in STAGE_NAMES
        if i not in RailEnv._base_stages_names  # pylint: disable=protected-access
    ]
)

# parameters that are passed to make_stage for all stages
GLOBAL_INTERACTIVE_PARAMETERS = {"output_mode": "return", "force_exact": True}


def _get_stage_definition(stage_name: str) -> type[RailStage]:
    """Fetch the original class definition for a RAIL stage from its name

    Parameters
    ----------
    stage_name : str
        Name of the RAIL stage class

    Returns
    -------
    type[RailStage]
        The class definition
    """
    return RailStage.pipeline_stages[stage_name][0]


def _get_stage_module(stage_name: str, interactive: bool = False) -> str:
    """Get the name of the module from which you would import the given RAIL stage

    Parameters
    ----------
    stage_name : str
        Name of the RAIL stage class
    interactive : bool, optional
        Whether this would be imported from rail.interactive instead of rail, by default False

    Returns
    -------
    str
        Absolute import path
    """
    module = _get_stage_definition(stage_name).__module__
    if interactive:
        return module.replace("rail.", "rail.interactive.", 1)
    return module


def _get_virtual_submodule_names(
    module: ModuleType, stage_names: list[str]
) -> list[str]:
    """Get a list of all of the submodules of `module` that interactive will have

    Parameters
    ----------
    module : ModuleType
        The module to get children of
    stage_names : list[str]
        The stages that we need locations for

    Returns
    -------
    list[str]
        A sorted, unique list of module names, including intermediaries between `module`
        and the definition of each stage
    """
    stage_module_names = [
        _get_stage_module(stage, interactive=True) for stage in stage_names
    ]
    stage_module_names = sorted(list(set(stage_module_names)))

    # get a recursive version of the names
    recursive_module_names = []
    for module_name in stage_module_names:
        relative_name = module_name.replace(module.__name__, "")[1:]
        parts = relative_name.split(".")

        for depth in range(len(parts)):
            relative_name = ".".join(parts[: depth + 1])
            if relative_name not in recursive_module_names:
                recursive_module_names.append(module.__name__ + "." + relative_name)

    return sorted(list(set(recursive_module_names)))
