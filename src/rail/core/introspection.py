import importlib
import os
import pkgutil
from types import ModuleType

import setuptools

import rail


class RailEnv:

    _packages: dict[str, pkgutil.ModuleInfo] = {}
    _namespace_path_dict: dict[str, list[str]] = {}
    _namespace_module_dict: dict[str, list[pkgutil.ModuleInfo]] = {}
    _module_dict: dict[str, list[str]] = {}
    _module_path_dict: dict[str, str] = {}
    _tree: dict[str, list] = {}
    _stage_dict: dict[str, list[str]] = {}
    _base_stages: list[type] = []

    @classmethod
    def list_rail_packages(cls) -> dict[str, pkgutil.ModuleInfo]:
        """List all the packages that are available in the RAIL ecosystem

        Returns
        -------
        dict[str,str]:
            Dict mapping the package names to the path to the package

        """
        cls._packages = {
            pkg.name: pkg
            for pkg in pkgutil.iter_modules(rail.__path__, rail.__name__ + ".")
        }
        return cls._packages

    @classmethod
    def print_rail_packages(cls) -> None:
        """Print all the packages that are available in the RAIL ecosystem"""
        if not cls._packages:  # pragma: no cover
            cls.list_rail_packages()
        for pkg_name, pkg in cls._packages.items():
            assert isinstance(pkg[0], importlib.machinery.FileFinder)
            path = pkg[0].path
            print(f"{pkg_name} @ {path}")

    @classmethod
    def list_rail_namespaces(cls) -> dict[str, list[str]]:
        """List all the namespaces within rail

        Returns
        -------
        dict[str, list[str]]:
            Dict mapping the namespaces to the paths contributing to
            each namespace
        """
        cls._namespace_path_dict.clear()

        for path_ in rail.__path__:
            namespaces = setuptools.find_namespace_packages(path_)
            for namespace_ in namespaces:
                # exclude stuff that starts with 'example'
                if namespace_.find("example") == 0:
                    continue
                if namespace_ in cls._namespace_path_dict:  # pragma: no cover
                    cls._namespace_path_dict[namespace_].append(path_)
                else:
                    cls._namespace_path_dict[namespace_] = [path_]

        return cls._namespace_path_dict

    @classmethod
    def print_rail_namespaces(cls) -> None:
        """Print all the namespaces that are available in the RAIL ecosystem"""
        if not cls._namespace_path_dict:
            cls.list_rail_namespaces()
        for key, val in cls._namespace_path_dict.items():
            print(f"Namespace {key}")
            for vv in val:
                print(f"     {vv}")

    @classmethod
    def list_rail_modules(cls) -> dict[str, str]:
        """List all modules within rail

        Returns
        -------
        dict[str, str]
            Dict mapping module names to their import paths
        """
        cls._module_dict.clear()
        cls._module_path_dict.clear()
        cls._namespace_module_dict.clear()
        if not cls._namespace_path_dict:  # pragma: no cover
            cls.list_rail_namespaces()
        for key, val in cls._namespace_path_dict.items():
            cls._namespace_module_dict[key] = []
            for vv in val:
                fullpath = os.path.join(vv, key.replace(".", "/"))
                modules = list(
                    pkgutil.iter_modules([fullpath], rail.__name__ + "." + key + ".")
                )
                for module_ in modules:
                    if module_.name in cls._module_dict:  # pragma: no cover
                        cls._module_dict[module_.name].append(key)
                    else:
                        cls._module_dict[module_.name] = [key]
                    cls._namespace_module_dict[key].append(module_)
                    assert isinstance(module_[0], importlib.machinery.FileFinder)
                    cls._module_path_dict[module_.name] = module_[0].path

        return cls._module_path_dict

    @classmethod
    def print_rail_modules(cls) -> None:
        """Print all the moduels that are available in the RAIL ecosystem"""
        if not cls._module_dict:
            cls.list_rail_modules()

        for key, val in cls._module_dict.items():
            print(f"Module {key}")
            for vv in val:
                print(f"     {vv}")

        for key, val2 in cls._namespace_module_dict.items():
            print(f"Namespace {key}")
            for vv2 in val2:
                print(f"     {vv2}")

    @classmethod
    def build_rail_namespace_tree(cls) -> dict[str, list[dict]]:
        """Build a tree of the namespaces and packages in rail

        Returns
        -------
        dict[str, list[dict]]:
           Tree of the namespaces and packages in rail
        """
        cls._tree.clear()
        if not cls._namespace_module_dict:  # pragma: no cover
            cls.list_rail_modules()

        if not cls._packages:  # pragma: no cover
            cls.list_rail_packages()

        level_dict: dict[int, list[str]] = {}
        for key in cls._namespace_module_dict:
            count = key.count(".")
            if count in level_dict:
                level_dict[count].append(key)
            else:
                level_dict[count] = [key]

        depth = max(level_dict.keys())
        for current_depth in range(depth + 1):
            for key in level_dict[current_depth]:
                _nsname = f"rail.{key}"
                if current_depth == 0:
                    _nsname = f"rail.{key}"
                    cls._tree[key] = cls._namespace_module_dict[key]
                else:
                    parent_key = ".".join(key.split(".")[0:current_depth])
                    if parent_key in cls._tree:
                        cls._tree[parent_key].append(
                            {key: cls._namespace_module_dict[key]}
                        )

        return cls._tree

    @classmethod
    def pretty_print_tree(cls, the_dict: dict | None = None, indent: str = "") -> None:
        """Utility function to help print the namespace tree

        This can be called recurisvely to walk the tree structure, which has nested dicts

        Parameters
        ----------
        the_dict:
            Current dictionary to print, if None it will print cls._tree

        indent:
            Indentation string prepended to each line
        """
        if the_dict is None:  # pragma: no cover
            the_dict = cls._tree
        for key, val in the_dict.items():
            nsname = f"rail.{key}"
            if nsname in cls._packages:
                pkg_type = "Package"
            else:
                pkg_type = "Namespace"

            print(f"{indent}{pkg_type} {nsname}")
            for vv in val:
                if isinstance(vv, dict):
                    cls.pretty_print_tree(vv, indent=indent + "    ")
                else:
                    print(f"    {indent}{vv.name}")

    @classmethod
    def print_rail_namespace_tree(cls) -> None:
        """Print the namespace tree in a nice way"""
        if not cls._tree:
            cls.build_rail_namespace_tree()
        cls.pretty_print_tree(cls._tree)

    @classmethod
    def do_pkg_api_rst(cls, basedir: str, key: str, val: list) -> None:
        """Build the api rst file for a rail package

        Parameters
        ----------
        basedir:
            Directory to write file to

        key:
            Name of the rail package

        val:
            Namespace tree for the package
        """

        api_pkg_toc = f"rail.{key} package\n"
        api_pkg_toc += "=" * len(api_pkg_toc)

        api_pkg_toc += f"""
.. automodule:: rail.{key}
    :members:
    :undoc-members:
    :show-inheritance:

Submodules
----------

.. toctree::
    :maxdepth: 4

"""

        for vv in val:
            if isinstance(vv, dict):  # pragma: no cover
                for _k3, v3 in vv.items():
                    for v4 in v3:
                        api_pkg_toc += f"    {v4.name}.rst\n"
            else:
                api_pkg_toc += f"    {vv.name}.rst\n"

        with open(
            os.path.join(basedir, "api", f"rail.{key}.rst"), "w", encoding="utf-8"
        ) as apitocfile:
            apitocfile.write(api_pkg_toc)

    @classmethod
    def do_namespace_api_rst(cls, basedir: str, key: str, val: list) -> None:
        """Build the api rst file for a rail namespace

        Parameters
        ----------
        basedir:
            Directory to write file to

        key:
            Name of the rail namespace

        val:
            Namespace tree for the namespace
        """

        api_pkg_toc = f"{key} namespace\n"
        api_pkg_toc += "=" * len(api_pkg_toc)

        api_pkg_toc += """

.. py:module:: rail.{key}

Subpackages
-----------

.. toctree::
    :maxdepth: 4

{sub_packages}

Submodules
----------

.. toctree::
    :maxdepth: 4

{sub_modules}
"""

        sub_packages = ""
        sub_modules = ""
        for vv in val:
            if isinstance(vv, dict):
                for k3, v3 in vv.items():
                    cls.do_namespace_api_rst(basedir, k3, v3)
                    sub_packages += f"    rail.{k3}\n"
            else:
                sub_modules += f"    {vv.name}\n"
        api_pkg_toc = api_pkg_toc.format(
            key=key, sub_packages=sub_packages, sub_modules=sub_modules
        )

        with open(
            os.path.join(basedir, "api", f"rail.{key}.rst"), "w", encoding="utf-8"
        ) as apitocfile:
            apitocfile.write(api_pkg_toc)

    @classmethod
    def do_api_rst(cls, basedir: str = ".") -> None:
        """Build the top-level API documentation

        Parameters
        ----------
        basedir:
            Directory to write file to
        """

        if not cls._tree:  # pragma: no cover
            cls.build_rail_namespace_tree()

        apitoc = """API Documentation
=================

Information on specific functions, classes, and methods.

Base Packages
-------------

.. toctree::
    :maxdepth: 4

{base_packages}


Namespaces
----------

.. toctree::
    :maxdepth: 4

{namespaces}


Algorithm Packages
------------------

.. toctree::
    :maxdepth: 4

{algorithm_packages}

"""
        try:
            os.makedirs(basedir)
        except Exception:
            pass

        try:
            os.makedirs(os.path.join(basedir, "api"))
        except Exception:  # pragma: no cover
            pass

        base_packages = ""
        namespaces = ""
        algorithm_packages = ""

        for key, val in cls._tree.items():
            nsname = f"rail.{key}"
            nsfile = os.path.join("api", f"{nsname}.rst")

            if nsname in cls._packages:
                # Skip rail_projects
                if nsname in ["rail.projects", "rail.plotting"]:
                    continue
                cls.do_pkg_api_rst(basedir, key, val)
                if nsname in ["rail.core", "rail.interfaces", "rail.stages"]:
                    base_packages += f"    {nsfile}\n"
                else:
                    algorithm_packages += f"    {nsfile}\n"

            else:
                cls.do_namespace_api_rst(basedir, key, val)
                namespaces += f"    {nsfile}\n"

        apitoc = apitoc.format(
            base_packages=base_packages,
            namespaces=namespaces,
            algorithm_packages=algorithm_packages,
        )
        with open(
            os.path.join(basedir, "api.rst"), "w", encoding="utf-8"
        ) as apitocfile:
            apitocfile.write(apitoc)

    @classmethod
    def import_all_packages(cls) -> None:
        """Import all the packages that are available in the RAIL ecosystem"""
        pkgs = cls.list_rail_packages()
        for pkg in pkgs:
            try:
                _imported_module = importlib.import_module(pkg)
                print(f"Imported {pkg}")
            except Exception as msg:
                print(f"Failed to import {pkg} because: {str(msg)}")

    @classmethod
    def attach_stages(cls, to_module: ModuleType) -> None:
        """Attach all the available stages to this module

        Parameters
        ----------
        to_module:
            python module we are attaching stages to

        Notes
        -----
        This allow you to do 'from rail.stages import *'
        """
        from rail.core.stage import \
            RailStage  # pylint: disable=import-outside-toplevel

        cls._stage_dict.clear()
        cls._stage_dict["none"] = []
        cls._base_stages.clear()

        n_base_classes = 0
        n_stages = 0

        for stage_name, stage_info in RailStage.incomplete_pipeline_stages.items():
            if stage_info[0] in [RailStage]:
                continue
            cls._base_stages.append(stage_info[0])
            cls._stage_dict[stage_info[0].__name__] = []
            n_base_classes += 1

        for stage_name, stage_info in RailStage.pipeline_stages.items():
            setattr(to_module, stage_name, stage_info[0])
            n_stages += 1

        for stage_name, stage_info in RailStage.pipeline_stages.items():
            baseclass = "none"
            for possible_base in cls._base_stages:
                if issubclass(stage_info[0], possible_base):
                    baseclass = possible_base.__name__
                    break
            cls._stage_dict[baseclass].append(stage_name)

        print(
            f"Attached {n_base_classes} base classes and {n_stages} fully formed stages to rail.stages"
        )

    @classmethod
    def print_rail_stage_dict(cls) -> None:
        """Print an dict of all the RailSages organized by their base class"""
        for key, val in cls._stage_dict.items():
            print(f"BaseClass {key}")
            for vv in val:
                print(f"  {vv}")
