"""
Not sure about the placement of this file
"""

import collections
import inspect
from warnings import warn

import numpydoc.validate

import rail.core.data
from rail.core.introspection import RailEnv
from rail.core.stage import RailStage
from rail.utils.interactive.docstring_utils import (
    InteractiveParameter,
    _parse_annotation_string,
    _split_docstring,
    _stringify_type_annotation,
)
from rail.utils.interactive.initialize_utils import STAGE_NAMES, _get_stage_definition


def test_interactive_function_names() -> None:
    """Check for unset names and name collisions in the `interactive_function` attribute
    of RailStages."""
    if_name_attribute = "interactive_function"
    function_name_uses = collections.defaultdict(
        list  # {"name of interactive fn":[stages using that name, ...]}
    )
    exists_stage_without_attribute = False
    exists_name_with_multiple_uses = False

    # initial loop through stages
    for stage_name in STAGE_NAMES:
        # get and store the name
        interactive_function_name = getattr(
            _get_stage_definition(stage_name), if_name_attribute
        )

        # run the is None check for this stage
        if interactive_function_name is None:
            print(f"{if_name_attribute} is not set for RailStage {stage_name}")
            exists_stage_without_attribute = True

        else:
            function_name_uses[interactive_function_name].append(stage_name)
    assert not exists_stage_without_attribute

    # secondary loop to check for duplicates
    for interactive_function_name, stages in function_name_uses.items():
        if len(stages) > 1:
            msg = "{if_name_attribute} = {function} is repeated in mutiple stages: {stages}"
            print(
                msg.format(
                    if_name_attribute=if_name_attribute,
                    function=interactive_function_name,
                    stages=", ".join(stages),
                )
            )
            exists_name_with_multiple_uses = True
    assert not exists_name_with_multiple_uses


###################################


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


def test_railstage_docstrings() -> None:
    entrypoint_functions = get_entrypoint_function_paths()
    for epf in entrypoint_functions:
        validation_result = validate_function(epf)
        assert validation_result == 0


###################################


def test_entrypoint_kwargs() -> None:
    for stage_name in STAGE_NAMES:
        stage_definition = _get_stage_definition(stage_name)
        epf_inspected_parameters = inspect.signature(
            getattr(stage_definition, stage_definition.entrypoint_function)
        ).parameters.values()

        if "kwargs" not in [i.name for i in epf_inspected_parameters]:
            raise ValueError(f"Missing kwargs in interactive function of {stage_name}")


###################################


def check_duplicate_parameters(stage_name) -> None:
    stage_definition = _get_stage_definition(stage_name)

    epf_inspected_parameters = inspect.signature(
        getattr(stage_definition, stage_definition.entrypoint_function)
    ).parameters.values()
    epf_docstring = getattr(
        stage_definition, stage_definition.entrypoint_function
    ).__doc__
    epf_sections = _split_docstring(epf_docstring)
    epf_parameters = _parse_annotation_string(
        epf_sections["Parameters"], epf_inspected_parameters
    )

    class_parameters = [
        InteractiveParameter.from_ceci(name, ceci_param)
        for name, ceci_param in stage_definition.config_options.items()
    ]

    existing_names = [p.name for p in epf_parameters]
    for parameter in class_parameters:
        if parameter.name in existing_names:
            warn(
                f"WARNING - parameter '{parameter.name}' is duplicated in config_options and EPF of {stage_name}"  # pylint: disable=line-too-long
            )


def test_duplicate_parameters() -> None:
    for stage_name in STAGE_NAMES:
        check_duplicate_parameters(stage_name)


###################################


def validate_return_annotation(
    stage_name: str,
    stage_definition: type[RailStage],
    return_elements: list[InteractiveParameter],
) -> None:
    """Helper function for test_return_annotations"""

    # grab the different return types
    from_docstring = [r.annotation for r in return_elements]
    from_inspect = _stringify_type_annotation(
        inspect.signature(
            getattr(stage_definition, stage_definition.entrypoint_function)
        ).return_annotation
    )
    from_ceci = [i[1] for i in stage_definition.outputs]

    warning_start = "Return type annotation for entrypoint function"

    if from_inspect == inspect.Parameter.empty:
        raise ValueError(f"{warning_start} is missing in {stage_name}")

    if isinstance(
        from_inspect, type
    ):  # a class, rather than, say, dict[str, str] which is a GenericAlias

        # if we have a single ceci output, make sure it matches the annotation
        if (len(from_ceci) == 1) and (from_inspect != from_ceci[0]):
            warn(
                f"WARNING: {warning_start} doesn't match the output property of {stage_name}"
            )
        if len(from_ceci) > 1:
            warn(
                f"WARNING: {warning_start} cannot be checked against multiple ceci outputs defined in {stage_name}"  # pylint: disable=line-too-long
            )

        from_inspect = str(from_inspect).split("'")[1]

        # if we have a single docstring item, make sure it matches the annotation
        if len(from_docstring) == 1:
            from_docstring = from_docstring[0]
            if from_inspect.endswith(from_docstring):  # Ensemble & Ensemble
                return
            if "." in from_docstring:  # Ensemble and qp.Ensemble
                docstring_final = from_docstring[from_docstring.rindex(".") + 1 :]
                if from_inspect.endswith(docstring_final):
                    return
            raise ValueError(
                f"{warning_start} doesn't match the docstring in {stage_name} (docstring={repr(from_docstring)}, annotation={from_inspect})"  # pylint: disable=line-too-long
            )
        if len(from_docstring) > 1:
            warn(
                f"WARNING: {warning_start} cannot be checked against multiple docstring entries in {stage_name}"  # pylint: disable=line-too-long
            )
    elif from_inspect is None:
        return
    else:
        warn(f"WARNING: {warning_start} is too complex to be checked in {stage_name}")


def check_returned_datahandle(
    stage_name: str, return_elements: list[InteractiveParameter]
) -> None:
    """Helper function for test_return_annotations"""

    for item in return_elements:
        # Handle return annotations that are DataHandles
        if hasattr(rail.core.data, item.annotation):
            return_type = getattr(rail.core.data, item.annotation)
            if return_type.interactive_type is None:
                raise ValueError(
                    f"{return_type} used in {stage_name} is missing interactive details"
                )


def test_return_annotations() -> None:
    for stage_name in STAGE_NAMES:
        stage_definition = _get_stage_definition(stage_name)
        epf_docstring = getattr(
            stage_definition, stage_definition.entrypoint_function
        ).__doc__
        epf_sections = _split_docstring(epf_docstring)
        return_elements = _parse_annotation_string(
            epf_sections["Returns"], return_annotations=True
        )

        validate_return_annotation(stage_name, stage_definition, return_elements)
        check_returned_datahandle(stage_name, return_elements)
