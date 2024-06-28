import os
import yaml

import rail.stages
from rail.core import RailEnv
from rail.core.stage import RailPipeline
from rail.cli.options import GitMode
from rail.utils.path_utils import RAILDIR
from rail.utils import catalog_utils


def render_nb(outdir, clear_output, dry_run, inputs, skip, **_kwargs):
    command = "jupyter nbconvert"
    options = "--to html"

    status = {}

    for nb_file in inputs:
        if nb_file in skip:
            continue
        subdir = os.path.dirname(nb_file).split("/")[-1]
        basename = os.path.splitext(os.path.basename(nb_file))[0]
        outfile = os.path.join("..", "..", outdir, f"{subdir}/{basename}.html")
        relpath = os.path.join(outdir, f"{subdir}")

        try:
            print(relpath)
            os.makedirs(relpath)
        except FileExistsError:
            pass

        if clear_output:
            comline = f"{command} --clear-output {nb_file}"
        else:
            comline = f"{command} {options} --output {outfile} --execute {nb_file}"

        if dry_run:
            render = 0
            print(comline)
        else:
            render = os.system(comline)
        status[nb_file] = render

    failed_notebooks = []
    for key, val in status.items():
        print(f"{key} {val}")
        if val != 0:  # pragma: no cover
            failed_notebooks.append(key)

    if failed_notebooks:  # pragma: no cover
        raise ValueError(f"The following notebooks failed {str(failed_notebooks)}")


def clone_source(outdir, git_mode, dry_run, package_file):  # pragma: no cover
    with open(package_file, encoding="utf-8") as pfile:
        package_dict = yaml.safe_load(pfile)

    for key, _val in package_dict.items():
        if os.path.exists(f"{outdir}/{key}"):
            print(f"Skipping existing {outdir}/{key}")
            continue

        if git_mode == GitMode.ssh:
            com_line = f"git clone https://github.com/LSSTDESC/{key}.git {outdir}/{key}"
        elif git_mode == GitMode.https:
            com_line = f"git clone git@github.com:LSSTDESC/{key}.git {outdir}/{key}"
        elif git_mode == GitMode.cli:
            com_line = f"gh repo clone LSSTDESC/{key} {outdir}/{key}"

        if dry_run:
            print(com_line)
        else:
            os.system(com_line)


def update_source(outdir, dry_run, package_file):
    with open(package_file, encoding="utf-8") as pfile:
        package_dict = yaml.safe_load(pfile)

    currentpath = os.path.abspath(".")
    for key, _val in package_dict.items():
        abspath = os.path.abspath(f"{outdir}/{key}")

        if os.path.exists(f"{outdir}/{key}") is not True:  # pragma: no cover
            print(f"Package {outdir}/{key} does not exist!")
            continue

        com_line = f"cd {abspath} && git pull && cd {currentpath}"

        if dry_run:
            print(com_line)
        else:  # pragma: no cover
            os.system(com_line)


def install(outdir, from_source, dry_run, package_file):
    with open(package_file, encoding="utf-8") as pfile:
        package_dict = yaml.safe_load(pfile)

    for key, val in package_dict.items():
        if not from_source:
            com_line = f"pip install {val}"
        else:
            if not os.path.exists(f"{outdir}/{key}"):  # pragma: no cover
                print(f"Skipping missing {outdir}/{key}")
                continue
            com_line = f"pip install -e {outdir}/{key}"

        if dry_run:
            print(com_line)
        else:  # pragma: no cover
            os.system(com_line)


def info(**kwargs):
    rail.stages.import_and_attach_all()

    print_all = kwargs.get("print_all", False)
    if kwargs.get("print_packages") or print_all:
        print("======= Printing RAIL packages ==============")
        RailEnv.print_rail_packages()
        print("\n\n")
    if kwargs.get("print_namespaces") or print_all:
        print("======= Printing RAIL namespaces ==============")
        RailEnv.print_rail_namespaces()
        print("\n\n")
    if kwargs.get("print_modules") or print_all:
        print("======= Printing RAIL modules ==============")
        RailEnv.print_rail_modules()
        print("\n\n")
    if kwargs.get("print_tree") or print_all:
        print("======= Printing RAIL source tree ==============")
        RailEnv.print_rail_namespace_tree()
        print("\n\n")
    if kwargs.get("print_stages") or print_all:
        print("======= Printing RAIL stages ==============")
        RailEnv.print_rail_stage_dict()
        print("\n\n")


def get_data(verbose, **kwargs):  # pragma: no cover
    standard_data_files = [
        {
            "local_path": "rail/examples_data/goldenspike_data/data/base_catalog.pq",
            "remote_path": "https://portal.nersc.gov/cfs/lsst/PZ/base_catalog.pq",
        }
    ]
    bpz_data_files = [
        {
            "local_path": "rail/examples_data/estimation_data/data/nonphysical_dc2_templates.tar",
            "remote_path": "https://portal.nersc.gov/cfs/lsst/PZ/nonphysical_dc2_templates.tar",
        },
        {
            "local_path": "rail/examples_data/estimation_data/data/test_dc2_training_9816_broadtypes.hdf5",
            "remote_path": "https://portal.nersc.gov/cfs/lsst/PZ/test_dc2_training_9816_broadtypes.hdf5",
        },
        {
            "local_path": "rail/examples_data/estimation_data/data/test_dc2_train_customtemp_broadttypes.hdf5",  # pylint: disable=line-too-long
            "remote_path": "https://portal.nersc.gov/cfs/lsst/PZ/test_dc2_train_customtemp_broadttypes.hdf5",
        },
    ]

    data_files = standard_data_files
    if kwargs.get("bpz_demo_data"):
        # The bpz demo data is quarantined into its own flag, as it contains some
        # non-physical features that would add systematics if run on any real data.
        # This data should NOT be used for any science with real data!
        data_files = bpz_data_files
        print("Downloading BPZ demo data...")
        print(
            "(Note: you can run get-data without the bpz-demo-data flag to download standard data.)"
        )

    for data_file in data_files:
        local_abs_path = os.path.join(RAILDIR, data_file["local_path"])
        if verbose:
            print(
                f"Check file exists: {local_abs_path} ({os.path.exists(local_abs_path)})"
            )
        if not os.path.exists(local_abs_path):
            os.system(
                f'curl -o {local_abs_path} {data_file["remote_path"]} --create-dirs'
            )


def build_pipeline(
    pipeline_class,
    output_yaml,
    catalog_tag=None,
    input_dict=None,
    stages_config=None,
    output_dir='.',
    log_dir=None,
    **kwargs
):
    tokens = pipeline_class.split('.')
    module = '.'.join(tokens[:-1])
    class_name = tokens[-1]

    if catalog_tag:
        catalog_utils.apply_defaults(catalog_tag)

    if log_dir is None:
        log_dir = os.path.join(output_dir, 'logs', class_name)

    __import__(module)
    RailPipeline.build_and_write(class_name, output_yaml, input_dict, stages_config, output_dir, log_dir, **kwargs)
