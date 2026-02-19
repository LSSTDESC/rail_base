"""Utility functions to generate the docstrings attatched to the interactive versions of
RailStages"""

import inspect
import textwrap
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

from ceci.config import StageConfig, StageParameter

import rail.core.data
from rail.core.stage import RailStage
from rail.utils.interactive.base_utils import (
    GLOBAL_INTERACTIVE_PARAMETERS,
    _get_stage_definition,
)
from rail.utils.path_utils import RAILDIR, unfind_rail_file

# INTERACTIVE_DO: is there any case where an interactive function might not return? no, right?
DOCSTRING_FORMAT = """
{class_summary}

---

{function_summary}

---

This function was generated from the function {source_file}

Parameters
----------
{parameters}

Returns
-------
{returns}

{extra_documentation}
"""
DOCSTRING_INDENTATION = 4
DOCSTRING_LINE_LENGTH = 88


# very small subset of numpydoc.validate.ALLOWED_SECTIONS
SECTION_HEADERS = [
    "Summary",
    "Parameters",
    "Returns",
    "Notes",
]


@dataclass
class InteractiveParameter:
    """Class to hold a small amount of information about a parameter to be passed to an
    interactive RailStage function

    We don't just re-use ceci's StageParameter as the information structure doesn't
    match what we want for docstrings
    """

    name: str | None
    annotation: str | type
    description: str
    parameter_usage: Literal["required"] | Literal["optional"] | Literal["return"]

    def __post_init__(self) -> None:
        self.annotation = _stringify_type_annotation(self.annotation)

        # append optional/required to the annotation
        if (self.parameter_usage != "return") and (
            not self.annotation.endswith(self.parameter_usage)
        ):
            self.annotation = f"{self.annotation}, {self.parameter_usage}"

    def __str__(self) -> str:
        description = textwrap.indent(self.description, " " * DOCSTRING_INDENTATION)
        if len(description) > 0:
            description = f"\n{description}"
        if self.name is None:
            return f"{self.annotation}{description}"
        return f"{self.name} : {self.annotation}{description}"

    @classmethod
    def from_ceci(cls, name: str, ceci_param: Any) -> "InteractiveParameter":
        """Create an InteractiveParameter object from the ceci config_options items,
        branchs for cases where the item is a single item, config set, or not actually
        set as a StageParameter"""
        if isinstance(ceci_param, StageParameter):
            return cls.from_ceci_parameter(name, ceci_param)
        if isinstance(ceci_param, StageConfig):
            return cls.from_ceci_parameter(name, dict.__getitem__(ceci_param, name))

        return cls(
            name=name,
            annotation="unknown type, optional",
            description=f"Default: {ceci_param}",
            parameter_usage="optional",
        )

    @classmethod
    def from_ceci_parameter(
        cls, name: str, ceci_param: StageParameter
    ) -> "InteractiveParameter":
        """Parse a ceci StageParameter to reformat the information as desired by
        InteractiveParameter"""
        dtype_name = "unknown type"
        if ceci_param.dtype is not None:
            dtype_name = ceci_param.dtype.__name__

        description = " ".join(
            textwrap.wrap(
                ceci_param.msg, width=DOCSTRING_LINE_LENGTH - DOCSTRING_INDENTATION * 2
            )
        )
        annotation = dtype_name

        if not ceci_param.required:
            default_value = ceci_param.default
            max_default_length = (
                DOCSTRING_LINE_LENGTH - DOCSTRING_INDENTATION * 2 - len("Default: ")
            )

            # handle default values that might be paths
            if isinstance(ceci_param.default, (str, Path)):
                default_value = _handle_default_path(ceci_param.default)
            if isinstance(ceci_param.default, dict):
                default_value = _handle_dictionary_paths(ceci_param.default)

            # truncate long default values
            if (len(str(default_value)) > max_default_length) and (
                isinstance(default_value, (dict, list))
            ):
                end = str(default_value)[-1]
                shortened_string = textwrap.shorten(
                    str(default_value), width=max_default_length - 2, placeholder="..."
                )
                default_value = f"{shortened_string}{end}"

            description += f"\nDefault: {default_value}"

        return InteractiveParameter(
            name=name,
            annotation=annotation,
            description=description.strip(),  # don't start with newline if ceci msg=""
            parameter_usage="required" if ceci_param.required else "optional",
            # is_required=ceci_param.required,
        )

    def merge(self, other: "InteractiveParameter") -> "InteractiveParameter":
        """Combine two InteractiveParameters to create a new one.

        Parameters
        ----------
        other : InteractiveParameter
            The object to merge in

        Returns
        -------
        InteractiveParameter
            A merger ofthe two objects
        """

        # build the new description
        if self.description == other.description:
            new_description = self.description
        else:
            new_description = f"{self.description}\n{other.description}".strip()

        # build the new annotation, this depends on the format of the current annotations
        annotations = [self.annotation, other.annotation]
        n_str_annotations = sum(isinstance(t, str) for t in annotations)
        if n_str_annotations == 1:
            # one string annotation and one class annotation, take the string
            new_annotation = (
                [t for t in annotations if isinstance(t, str)][0]
                .strip()
                .removesuffix(", optional")
            )
        else:
            if n_str_annotations == 0:
                # two classes, convert to strings
                annotations = [str(t) for t in annotations]

            # simplify annotations, take the first viable one
            annotations = [
                t.strip().removesuffix(", optional").replace("unknown type", "")
                for t in annotations
            ]
            annotations = [t for t in annotations if len(t) > 0]

            new_annotation = annotations[0]

        # build the new parameter usage
        if any(u == "return" for u in [self.parameter_usage, other.parameter_usage]):
            raise ValueError("Can't combine return parameters")
        parameter_usage = (
            "required"
            if any(
                u == "required" for u in [self.parameter_usage, other.parameter_usage]
            )
            else "optional"
        )

        return InteractiveParameter(
            name=self.name,
            annotation=new_annotation,
            description=new_description,
            parameter_usage=parameter_usage,
        )


################################
# Minor utility functions
################################


def _stringify_type_annotation(annotation: str | type) -> str:
    """Try to convert a type annotation into a an interactive-friendly form.

    Parameters
    ----------
    annotation : str | type
        The original version

    Returns
    -------
    str
        A friendlier version of the type annotation
    """
    # change classes to strings
    if isinstance(annotation, type):
        annotation = annotation.__name__

    # handle annotations that are DataHandles
    if isinstance(annotation, str) and hasattr(rail.core.data, annotation):
        return_type = getattr(rail.core.data, annotation)
        if isinstance(return_type, type) and issubclass(
            return_type, rail.core.data.DataHandle
        ):
            annotation = return_type.interactive_type

    return annotation


def _handle_default_path(path: str | Path) -> str:
    """Replace absolute paths with relative ones for RailStage config items that have
    default values that are paths. Items that don't appear to be paths are passed
    through unchanged.

    Parameters
    ----------
    path : str | Path
        The default value of a config item

    Returns
    -------
    str
        A version of the input relative to the install locations of rail packages
    """
    if isinstance(path, Path):
        path = str(path)

    # if path.startswith(RAILDIR):
    #     return path.replace(RAILDIR, "rail.utils.path_utils.RAILDIR")
    if path.startswith("/"):
        return unfind_rail_file(path)

    return path


def _handle_dictionary_paths(dictionary: dict[str, Any]) -> dict[str, Any]:
    """Check through RailStage config items that are dictionaries, and call
    _handle_default_path for any items that are paths

    Parameters
    ----------
    dictionary : dict[str, Any]
        The default value of a config item

    Returns
    -------
    dict[str, Any]
        The same default, but with any paths changed to be relative
    """
    for key, value in dictionary.items():
        if isinstance(value, (str, Path)):
            dictionary[key] = _handle_default_path(value)
        if isinstance(value, dict):
            dictionary[key] = _handle_dictionary_paths(value)
    return dictionary


def _is_section_header(line_no: int, docstring_lines: list[str]) -> bool:
    """Check whether a given line is the start (text line) of a header.

    Headers are checked against a list of "splitting headers", and a line containing
    exactly this text, followed be a line of hyphens of the same length indicates a
    header

    This roughly follows the numpydoc function NumpyDocString._is_at_section()

    Parameters
    ----------
    line_no : int
        The line number to check
    docstring_lines : list[str]
        All lines in the docstring

    Returns
    -------
    bool
        Whether this line number starts a header
    """

    current_line = docstring_lines[line_no].strip()
    if current_line in SECTION_HEADERS:
        next_line = docstring_lines[line_no + 1].strip()
        return next_line == ("-" * len(current_line))
    return False


def _map_ceci_to_param(input_name: str, parameter_names: list[str]) -> str:
    """Check a ceci input tag against a set of parameter names for the entrypoint
    function, and determine if the tag matches up to any of them, and if so, which.

    The idea here is that a stage may have inputs `data` and `model`, and expect that
    `model` is passed to `make_stage`, and `data` is passed to the entrypoints function.
    This has the side effect that while `data` appears in the entrypoint function that
    we analyze, `model` does not.
    Since `model` is still a required element to run the stage, we need to make sure it
    is added to the docstring of the interactive version. However, we don't want to
    duplicate `data` in that docstring.
    Finally, we have the added complication that often tag names don't line up with the
    parameter names used for the same piece of information. So the stage may have the
    ceci input `data`, which corresponds to the parameter `input_data`.

    Parameters
    ----------
    input_name : str
        The tag name for the ceci input to a RailStage
    parameter_names : list[str]
        A list of names to check against, that would disqualify the ceci tag from being
        added to the docstring as an independent item

    Returns
    -------
    str
        The name of the parameter corresponding to this tag, or the original tag name if
        none were found
    """
    if input_name in parameter_names:
        return input_name

    if f"{input_name}_data" in parameter_names:
        # `input` and `input_data`, sometimes `truth` and `truth_data`
        return f"{input_name}_data"

    if input_name == "input":
        for name in ["data", "training_data", "catalog", "sample"]:
            # `input` often corresponds to one of these
            if name in parameter_names:
                return name

    if input_name.endswith("_input") and (
        input_name.replace("_input", "_data") in parameter_names
    ):
        # `spec_input` and `spec_data`
        return input_name.replace("_input", "_data")

    return input_name


################################
# Major utility functions
################################


def _split_docstring(docstring: str) -> defaultdict[str, str]:
    """Split the docstring into sections based on specific numpy-style headers.

    Does some whitespace formatting on the returned sections

    Parameters
    ----------
    docstring : str
        The raw docstring (.__doc__, inspect.getdoc, pydoc.doc)

    Returns
    -------
    defaultdict[str, str]
        Dictionary of {header: content}
    """

    result = defaultdict(list)
    docstring_lines = docstring.splitlines()

    # add lines to `result` one at a time, tracking which section is being used
    line_no = 0
    current_section = 0
    while line_no < len(docstring_lines) - 1:
        if _is_section_header(line_no, docstring_lines):
            current_section = SECTION_HEADERS.index(docstring_lines[line_no].strip())
            line_no += 2
        else:
            result[SECTION_HEADERS[current_section]].append(docstring_lines[line_no])
            line_no += 1
    result[SECTION_HEADERS[current_section]].append(
        docstring_lines[line_no]
    )  # add the final line

    # merge list items together
    joined_result = defaultdict(str)
    for title, lines in result.items():
        # add an indent for the first line of the summary, in case it starts on the same
        # line as the quotations
        if title == "Summary" and lines[0] == lines[0].strip():
            lines[0] = " " * 4 + lines[0]

        # unify and simplify whitespace
        joined_result[title] = textwrap.dedent(
            "\n".join(lines).replace("\n\n\n", "\n\n")
        ).strip()
    return joined_result


def _parse_annotation_string(
    text: str,
    inspected_parameters: list[inspect.Parameter] | None = None,
    return_annotations: bool = False,
) -> list[InteractiveParameter]:
    """Parse through an numpy-style Parameters section, and convert the information into
    InteractiveParameters. Also used for Returns.

    Parameters
    ----------
    text : str
        The numpy-style string
    inspected_parameters : list[inspect.Parameter] | None, optional
        Parameters found by using the inspect module, if any, by default None
    return_annotations : bool, optional
        Whether return the InterativeParameters with `parameter_usage="return"`

    Returns
    -------
    list[InteractiveParameter]
        The information contained in the docstring, but reformatted as
        InteractiveParameter objects.
    """

    lines = text.replace("\n\n", "\n").splitlines()
    annotation_linenos = []
    for i, line in enumerate(lines):
        if len(line.lstrip()) == len(line):
            annotation_linenos.append(i)

    parameters = []
    for i, lineno in enumerate(annotation_linenos):
        # get the item type and name (if supplied)
        if " : " in lines[lineno]:
            param_name, param_type = lines[lineno].split(" : ")

            if param_name.startswith("*"):
                continue
        else:
            param_name = None
            param_type = lines[lineno]

        # get the description
        if i < len(annotation_linenos) - 1:  # this is not the last annotation
            description_end = annotation_linenos[i + 1]
            description_lines = lines[lineno + 1 : description_end]
        else:
            description_lines = lines[lineno + 1 :]

        if return_annotations:
            parameter_usage = "return"
        else:
            # check if there is a default
            parameter_usage = "optional"
            if inspected_parameters is not None:
                inspect_parameter = [
                    p for p in inspected_parameters if p.name == param_name
                ][0]
                if inspect_parameter.default == inspect.Parameter.empty:
                    parameter_usage = "required"

        parameters.append(
            InteractiveParameter(
                name=param_name,
                annotation=param_type,
                description="\n".join([j.strip() for j in description_lines]),
                parameter_usage=parameter_usage,
            )
        )

    return parameters


def _create_parameters_section(
    stage_definition: type[RailStage], epf_parameter_string: str
) -> tuple[str, list[str]]:
    """Create the parameters section of the docstring for the interactive section.
    Abstracted into a dedicated function because of the volume of parsing required in
    managing the class and entrypoint function parameters, along with the ceci inputs

    The interactive function will take only kwargs, some of which are required.
    Required parameters come from:
    - positional parameters declared in the entrypoint function
    - required stage config options
    - ceci inputs that aren't already noted elsewhere (e.g., the `model` input for most
      estimators)

    Parameters
    ----------
    stage_definition : type[RailStage]
        Class definition for the stage
    epf_parameter_string : str
        Portion of the entrypoint function's docstring pertaining to parameters

    Returns
    -------
    str
        A string to use in the docstring of the interactive function
    list[str]
        List of kwargs that are required in the interactive function
    """

    # collect parameters from the entrypoint function, stage config, and ceci inputs
    class_parameters = [
        InteractiveParameter.from_ceci(name, ceci_param)
        for name, ceci_param in stage_definition.config_options.items()
    ]
    epf_inspected_parameters = inspect.signature(
        getattr(stage_definition, stage_definition.entrypoint_function)
    ).parameters.values()
    epf_parameters = _parse_annotation_string(
        epf_parameter_string, epf_inspected_parameters
    )
    ceci_inputs = getattr(stage_definition, "inputs", [])

    # separate parameters into required and optional
    epf_req, epf_opt = (
        [p for p in epf_parameters if p.parameter_usage == "required"],
        [p for p in epf_parameters if p.parameter_usage == "optional"],
    )
    class_req, class_opt = (
        [p for p in class_parameters if p.parameter_usage == "required"],
        [p for p in class_parameters if p.parameter_usage == "optional"],
    )
    ceci_req = [
        InteractiveParameter(
            # rename the ceci inputs to match the names of epf parameters, if matching
            # entries appear to exist
            # if we have the ceci input "spec_input", and the epf parameter "spec_data",
            # this section will create an InteractiveParameter with name "spec_data"
            name=_map_ceci_to_param(i[0], [p.name for p in epf_parameters]),
            annotation=i[1],
            description="",
            parameter_usage="required",
        )
        for i in ceci_inputs
    ]

    # create and populate the parameter lists for the interactive function

    # this section builds the list of required parameters, ensuring no duplicates
    ri_required = [*epf_req]
    for p in class_req:
        # store the required config_options, unless they've already been duplicated in
        # the positional epf parameters (doesn't seem to actually happen in any rail
        # stages)
        match = [(i, g) for i, g in enumerate(ri_required) if p.name == g.name]
        if len(match) == 0:
            ri_required.append(p)
        else:
            ri_required[match[0][0]] = p.merge(match[0][1])
    for p in ceci_req:
        # store the ceci inputs, unless they've already been duplicated in the
        # positional epf parameters
        if p.name not in [g.name for g in ri_required]:
            ri_required.append(p)

    # this section builds the list of optional parameters, and has some additional
    # complexity as it needs to be unique within itself, and also not duplicate any
    # items that already made it into the list of required parameters
    ri_optional: list[InteractiveParameter] = []
    for p in epf_opt:
        # store the epf kwargs, as optional, unless they show up in the required
        # parameters likely from being ceci inputs or config_options (e.g. flow_creator
        # n_samples)
        # for yaw stages this makes more things required than is actually necessary
        match = [(i, g) for i, g in enumerate(ri_required) if p.name == g.name]
        if len(match) == 0:
            ri_optional.append(p)
        else:
            ri_required[match[0][0]] = p.merge(match[0][1])
    for p in class_opt:
        req_match = [(i, g) for i, g in enumerate(ri_required) if p.name == g.name]
        opt_match = [(i, g) for i, g in enumerate(ri_optional) if p.name == g.name]
        if len(req_match) > 0:
            # a required ceci input or epf positional is duplicated as an optional
            # config opt (doesn't seem to actually happen in any rail stages)
            ri_required[req_match[0][0]] = p.merge(req_match[0][1])
        elif len(opt_match) > 0:
            # an optional config_opt is a duplicate of a epf kwarg - this happens for
            # `seed` in lots of places (e.g., add_column_of_random)
            ri_optional[opt_match[0][0]] = p.merge(opt_match[0][1])
        else:
            ri_optional.append(p)

    ri_optional = [
        p for p in ri_optional if p.name not in GLOBAL_INTERACTIVE_PARAMETERS
    ]
    docstring = "\n".join([str(i) for i in [*ri_required, *ri_optional]])
    return (docstring, [p.name for p in ri_required])


def _wrap_docstring(
    text: str,
    max_line_length: int,
    line_filter: Callable[[int, list[str]], bool] | None = None,
) -> str:
    """Wrap a docstring (or portion thereof) to a given length

    Parameters
    ----------
    text : str
        The text to wrap
    max_line_length : int
        The width to wrap at
    line_filter : Callable[[int, list[str]], bool] | None, optional
        A filter function to check if a line should skip wrapping, by default None.
        The function takes the line number and the full text (for multi-line analysis)
        and returns True if this line should skip wrapping, and False if this line
        should be wrapped. This function returning True (skip) will override the line
        length check, so is a way to force long lines to not wrap

    Returns
    -------
    str
        The wrapped text
    """
    wrapped_lines = []

    text = text.replace("\t", " " * 4)  # probably not necessary
    lines = text.splitlines()

    for i, line in enumerate(lines):
        # exit early if the line is short, or the filter says so
        line_is_short = len(line) <= max_line_length
        line_skips = line_filter(i, lines) if line_filter is not None else False
        if line_is_short or line_skips:
            wrapped_lines.append(line)
            continue

        unindented = line.lstrip()
        indent_size = len(line) - len(unindented)

        # wrap the text, keeping the indent width in mind
        wrapped_line = "\n".join(
            textwrap.wrap(unindented, width=max_line_length - indent_size)
        )

        # re-indent to the original depth, and save the line
        wrapped_lines.append(textwrap.indent(wrapped_line, " " * indent_size))

    return "\n".join(wrapped_lines)


def _param_annotion_wrap_filter(
    parameters_section_header: int,
    blank_lines: list[int],
    lineno: int,
    docstring_lines: list[str],
) -> bool:
    """Filter out annotation (not description) lines in the Parameters section of a
    docstring from being wrapped.

    Needs to be applied to a docstring BEFORE any indentation, as this uses the fact
    that annotations are un-indented.
    This also isn't applied to the Returns section (or any others that might have
    annotation-style items that shouldn't be wrapped; because Parameters is the only
    section we guarantee the existence of in an isolate-able fashion.

    Parameters
    ----------
    parameters_section_header : int
        The line number where the Parameters header is
    blank_lines : list[int]
        Empty lines in the docstring
    lineno : int
        The line number of the docstring to check
    docstring_lines : list[str]
        The entire docstring, split into lines

    Returns
    -------
    bool
        Whether to skip line wrapping because this is an annotation (True) or not (False)
    """

    # lines up to and including the parameter header are not parameter annotations
    if lineno <= parameters_section_header + 1:
        return False

    # check if we've left the parameters section
    if max(blank_lines) > parameters_section_header:
        # there exist blank lines after the param header, these should denote a new
        # section (though that new section might not have a header, which is why we're
        # checking with newlines)
        parameters_section_end = [
            i for i in blank_lines if i > parameters_section_header
        ][-1]
        if parameters_section_end < lineno:
            return False  # passed the end of the param section

    # we are inside the parameters section
    line = docstring_lines[lineno]
    unindented = line.lstrip()

    # if true this is a parameter annotation (not the description of it), skip wrapping
    return len(line) == len(unindented)


################################
# Primary function
################################


def create_interactive_docstring(stage_name: str) -> str:
    """Merge the relevant information from the class and entrypoint function of a RAIL
    stage to create a docstring for the interactive function

    Parameters
    ----------
    stage_name : str
        Name of the RAIL stage

    Returns
    -------
    str
        The final docstring for the interactive function
    """
    stage_definition = _get_stage_definition(stage_name)

    # get the raw docstrings
    class_docstring = stage_definition.__doc__
    epf_docstring = getattr(
        stage_definition, stage_definition.entrypoint_function
    ).__doc__
    epf_docstring = textwrap.dedent(
        "    " + epf_docstring
    )  # need to handle the first line lacking indent

    # do some pre-processing
    class_sections = _split_docstring(class_docstring)
    epf_sections = _split_docstring(epf_docstring)
    source_file = ".".join(
        [stage_definition.__module__, stage_name, stage_definition.entrypoint_function]
    )

    # handle the parameters
    parameters_content, required_parameters = _create_parameters_section(
        stage_definition, epf_sections["Parameters"]
    )

    # handle the return elements
    return_elements = _parse_annotation_string(
        epf_sections["Returns"], return_annotations=True
    )
    if len(return_elements) == 0:
        return_elements = ["None"]
    returns_content = "\n".join([str(i) for i in return_elements])

    # handle any other content
    extra_documentation = ""
    for section_name, section_content in class_sections.items():
        if section_name not in ["Summary", "Parameters", "Returns"]:
            header = f"\n{section_name}\n{'-'*len(section_name)}"
            extra_documentation += f"{header}\n{section_content}"
    for section_name, section_content in epf_sections.items():
        if section_name not in ["Summary", "Parameters", "Returns"]:
            header = f"\n{section_name}\n{'-'*len(section_name)}"
            extra_documentation += f"{header}\n{section_content}"
    if stage_definition.extra_interactive_documentation is not None:
        extra_class_docs = stage_definition.extra_interactive_documentation
        extra_class_docs = textwrap.dedent(extra_class_docs)
        extra_documentation += f"\n{extra_class_docs}"

    # assemble the docstring
    docstring = DOCSTRING_FORMAT.format(
        class_summary=class_sections["Summary"],
        function_summary=epf_sections["Summary"],
        source_file=source_file,
        parameters=parameters_content,
        returns=returns_content,
        extra_documentation=extra_documentation,
    )

    # prepare to wrap the docstring
    docstring_lines = docstring.splitlines()
    section_headers = [
        i for i in range(len(docstring_lines)) if _is_section_header(i, docstring_lines)
    ]
    parameters_section_header = [
        i for i in section_headers if docstring_lines[i] == "Parameters"
    ][0]
    blank_lines = [
        i for i in range(len(docstring_lines)) if len(docstring_lines[i]) == 0
    ]
    param_annotation_filter = partial(
        _param_annotion_wrap_filter, parameters_section_header, blank_lines
    )

    # wrap the docstring
    docstring = _wrap_docstring(
        docstring.strip(),
        max_line_length=DOCSTRING_LINE_LENGTH - DOCSTRING_INDENTATION,
        line_filter=param_annotation_filter,
    )

    return docstring, required_parameters
