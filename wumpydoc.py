"""
Use numpydoc (https://numpydoc.readthedocs.io) to validate the user-facing docstrings of
RAIL stages.

Currently this operates on
- whatever passed in, OR
- all entrypoint functions
It could be worth doing, in addition to the EPFs, the RailStages themselves

NumpyDoc error codes are pulled from
https://github.com/numpy/numpydoc/blob/98d88350c7580a8c875eae7968bc500f3efefdd5/numpydoc/validate.py#L46
Unused error codes are commented out, but kept for clarity.

if __name__ == "__main__" section is to be replaced with pyproject.toml script
[project.scripts]
name = "folder.filename:main"
Exit codes etc., will still work
"""

import argparse
import sys

import numpydoc.validate

import rail.stages
from rail.core.introspection import RailEnv
from rail.core.stage import RailStage
from rail.utils.interactive.initialize_utils import (
    STAGE_NAMES,
    _get_stage_definition,
)

RELEVANT_ERRORCODES = [
    # "GL01", # Which line summary starts on
    # "GL02", # Which line closing quotes are on
    "GL03",  # Double line break found
    "GL05",  # Tabs vs spaces
    "GL06",  # Unknown section - we can't match on sections we don't understand
    "GL07",  # Sections in wrong order
    "GL08",  # Missing docstring
    # "GL09", # Deprecation warning placement
    # "GL10", # reST directives formatting
    "SS01",  # No summary found
    # "SS02", # Summary does not start with a capital letter
    # "SS03", # Summary does not end with a period
    # "SS04", # Summary contains heading whitespaces
    # "SS05", # Summary must start with infinitive verb, not third person
    # "SS06", # Summary should fit in a single line
    # "ES01", # No extended summary found
    "PR01",  # Parameters {missing_params} not documented - we special case kwargs
    "PR02",  # Unknown parameters
    "PR03",  # Wrong parameters order
    "PR04",  # Parameter "{param_name}" has no type
    # "PR05", # Parameter "{param_name}" type should not finish with "."
    "PR06",  # Misspelled parameter types (integer instead of int)
    "PR07",  # Parameter "{param_name}" has no description
    # "PR08", # Parameter "{param_name}" description should start with a capital letter
    # "PR09", # Parameter "{param_name}" description should finish with "."
    "PR10",  # Parameter "{param_name}" requires a space before the colon
    "RT01",  # No Returns section found
    "RT02",  # The first line of the Returns section should contain only the type
    "RT03",  # Return value has no description
    # "RT04", # Return value description should start with a capital letter
    # "RT05", # Return value description should finish with "."
    # "YD01", # No Yields section found
    # "SA01", # See Also section not found
    # "SA02", # Missing period at end of description for See Also "{reference_name}" reference
    # "SA03", # Description should be capitalized for See Also "{reference_name}" reference
    # "SA04", # Missing description for See Also "{reference_name}" reference
    # "EX01", # No examples section found
]


def make_import_path(stage_definition: type[RailStage], stage_name: str) -> str:
    return ".".join(
        [stage_definition.__module__, stage_name, stage_definition.entrypoint_function]
    )


def get_entrypoint_function_paths() -> list[str]:
    """Get the import paths for all RailStage entrypoint functions

    Returns
    -------
    list[str]
        List of import paths (e.g.,
        "rail.creation.degraders.addRandom.AddColumnOfRandom.__call__")
    """

    paths = [
        make_import_path(getattr(rail.stages, base_stage), base_stage)
        for base_stage in RailEnv._base_stages_names  # pylint: disable=protected-access
    ]
    for stage_name in STAGE_NAMES:
        stage_definition = _get_stage_definition(stage_name)
        paths.append(make_import_path(stage_definition, stage_name))
    return paths


def validate_function(import_path: str) -> int:
    """Run the numpydoc validator for a single function. Only reports the errors that we
    care about.

    Parameters
    ----------
    import_path : str
        The Python-import style path to the function or other object

    Returns
    -------
    int
        0 if no errors, 1 otherwise
    """

    return_code = 0
    results = numpydoc.validate.validate(import_path)
    errors = [
        (code, description)
        for (code, description) in results["errors"]
        if code in RELEVANT_ERRORCODES
    ]

    for code, description in errors:
        if (code == "PR01") and ("{'**kwargs'}" in description):
            continue
        print(":".join([import_path, code, description]))
        return_code = 1

    return return_code


def main() -> int:
    """Main script, reads from CLI arguments to determine which docstrings to validate.

    Returns
    -------
    int
        0 if no errors found, 1 otherwise
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("entrypoint_functions", nargs="*")
    entrypoint_functions = parser.parse_args().entrypoint_functions

    if len(entrypoint_functions) == 0:
        # validate all stages
        entrypoint_functions = get_entrypoint_function_paths()

    return_code = 0
    for epf in entrypoint_functions:
        validation_result = validate_function(epf)
        if validation_result != 0:
            return_code = validation_result

    return return_code


if __name__ == "__main__":
    sys.exit(main())
