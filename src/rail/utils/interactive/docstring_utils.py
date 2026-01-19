"""Utility functions to generate the docstrings attatched to the interactive versions of
RailStages"""

import inspect
import textwrap
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

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
    annotation: str
    description: str
    is_required: bool = True

    def __str__(self) -> str:
        description = textwrap.indent(self.description, " " * DOCSTRING_INDENTATION)
        if self.name is None:
            return f"{self.annotation}\n{description}"
        return f"{self.name} : {self.annotation}\n{description}"

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
            is_required=False,
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
            annotation += ", optional"
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
            description=description.strip(),  # don't start with newline if cece msg=""
            is_required=ceci_param.required,
        )


################################
# Minor utility functions
################################


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

    if path.startswith(RAILDIR):
        return path.replace(RAILDIR, "rail.utils.path_utils.RAILDIR")
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


def _sort_parameters(
    parameters: list[InteractiveParameter],
) -> list[InteractiveParameter]:
    """Sort the parameter list of a docstring to put required parameters first

    Parameters
    ----------
    parameters : list[InteractiveParameter]
        The unsorted list of parameters

    Returns
    -------
    list[InteractiveParameter]
        Parameters, beginning with "input", then any required parameters, then optional
        ones.
    """
    input_parameter = None
    if parameters[0].name == "input":
        input_parameter = parameters.pop(0)

    remaining = sorted(parameters, key=lambda p: not p.is_required)

    if input_parameter is not None:
        remaining.insert(0, input_parameter)

    return remaining


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
    text: str, inspected_parameters: list[inspect.Parameter] | None = None
) -> list[InteractiveParameter]:
    """Parse through an numpy-style Parameters section, and convert the information into
    InteractiveParameters. Also used for Returns.

    Parameters
    ----------
    text : str
        The numpy-style string
    inspected_parameters : list[inspect.Parameter] | None, optional
        Parameters found by using the inspect module, if any, by default None

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

        # check if there is a default
        is_required = False
        if inspected_parameters is not None:
            inspect_parameter = [
                p for p in inspected_parameters if p.name == param_name
            ][0]
            is_required = inspect_parameter.default == inspect.Parameter.empty

        parameters.append(
            InteractiveParameter(
                name=param_name,
                annotation=param_type,
                description="\n".join([j.strip() for j in description_lines]),
                is_required=is_required,
            )
        )

    return parameters


def _create_parameters_section(
    stage_definition: type[RailStage], epf_parameter_string: str
) -> str:
    """Create the parameters section of the docstring for the interactive section.
    Abstracted into a dedicated function because of the volume of parsing required in
    managing both the class and entrypoint function parameters

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
    """

    # Read in parameters to both the class and EPF
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

    # Handle positional parameters to EPF
    input_is_wrapped = (
        False  # flag for the interactive function, indicating whether "input" is a dict
    )
    requires_input = True  # flag for the interactive function, indicating whether "input" should exist
    input_parameter_names = [
        i.name
        for i in epf_inspected_parameters
        if i.name not in ["self", "kwargs"] and i.default == inspect.Parameter.empty
    ]
    input_parameters_indices = [
        i
        for i, param in enumerate(epf_parameters)
        if param.name in input_parameter_names
    ]

    if len(input_parameters_indices) == 1:
        input_parameter = epf_parameters.pop(input_parameters_indices[0])
        input_parameter.name = "input"
        epf_parameters.insert(0, input_parameter)
    elif len(input_parameters_indices) > 1:
        input_is_wrapped = True
        annotation_entries = []
        description_entries = ["Dictionary of input data with the following keys:"]
        for i, index in enumerate(input_parameters_indices):
            param = epf_parameters.pop(index - i)  # adjust index as we shorten the list
            annotation_entries.append(f'"{param.name}": {param.annotation}')
            description_entries.append(
                f"{param.name}: {param.annotation} - {param.description.replace('\n',' ')}"
            )
        annotation = f"dict[{', '.join(annotation_entries)}]"
        input_parameter = InteractiveParameter(
            name="input",
            annotation=annotation,
            description="\n".join(description_entries),
        )
        epf_parameters.insert(0, input_parameter)
    else:
        requires_input = False

    # Class parameters
    for parameter in class_parameters:
        matching_parameter_indices = [
            i for (i, p) in enumerate(epf_parameters) if p.name == parameter.name
        ]
        if len(matching_parameter_indices) > 0:
            # don't add it to the list, since it'll be a duplicate, but check if this is
            # a make_stage required, entrypoint_function optional parameter
            i = matching_parameter_indices[0]
            if (not epf_parameters[i].is_required) and (parameter.is_required):
                epf_parameters[i].is_required = True
                epf_parameters[i].annotation = epf_parameters[
                    i
                ].annotation.removesuffix(", optional")
        else:
            epf_parameters.append(parameter)

    # remove the parameters that we force the values of
    epf_parameters = [
        p for p in epf_parameters if p.name not in GLOBAL_INTERACTIVE_PARAMETERS
    ]

    return (
        "\n".join([str(i) for i in _sort_parameters(epf_parameters)]),
        input_is_wrapped,
        requires_input,
    )


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
    parameters_content, input_is_wrapped, requires_input = _create_parameters_section(
        stage_definition, epf_sections["Parameters"]
    )

    # handle the return elements
    return_elements = _parse_annotation_string(epf_sections["Returns"])
    for item in return_elements:
        # Handle return annotations that are DataHandles
        if hasattr(rail.core.data, item.annotation):
            return_type = getattr(rail.core.data, item.annotation)
            item.annotation = return_type.interactive_type
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

    return docstring, input_is_wrapped, requires_input
