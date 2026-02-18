"""Utility functions"""

import os

import rail
import rail.core

RAILDIR = os.path.abspath(os.path.join(os.path.dirname(rail.core.__file__), "..", ".."))


def find_rail_file(relpath: str) -> str:
    """Find a file somewhere in rail by searching the namespace path

    This lets us avoid issues that the paths can be different depending
    on if we have installed things from source or not
    """
    for path_ in rail.__path__:
        fullpath = os.path.abspath(os.path.join(path_, relpath))
        if os.path.exists(fullpath):
            return fullpath
    raise ValueError(f"Could not find file {relpath} in {rail.__path__}")


def unfind_rail_file(abspath: str) -> str:
    """Go backwards from an absolute path to a rail file to the value that was passed to
    `find_rail_file`"""
    # paths created with find_rail_file
    for path_ in rail.__path__:
        if abspath.startswith(path_):
            rail_package_directory = os.path.abspath(
                os.path.join(path_, "..", "..", "..")
            )
            return os.path.normpath(abspath.removeprefix(rail_package_directory + "/"))

    # cache locations, which would be in user directory:

    user_home_directory = os.path.expanduser("~")

    if abspath.startswith(user_home_directory):
        return abspath.replace(os.path.expanduser("~"), "~")

    # Return error if not caught

    raise ValueError(f"Could not find the originating source of {abspath}")
