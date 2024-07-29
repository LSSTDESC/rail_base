import glob
import os
from rail.cli import scripts
from rail.cli.options import GitMode


def test_render_nb():
    nb_dir = "./tests/cli/"
    nb_files = glob.glob(os.path.join(nb_dir, "*.ipynb"))
    scripts.render_nb("docs", False, True, nb_files, skip=[])
    scripts.render_nb(
        "docs", True, True, nb_files, skip=["./tests/cli/single_number.ipynb"]
    )
    scripts.render_nb(
        "docs", True, False, nb_files, skip=["./tests/cli/single_number.ipynb"]
    )


def test_clone_source():
    scripts.clone_source("..", GitMode.ssh, True, "rail_packages.yml")
    scripts.clone_source("..", GitMode.https, True, "rail_packages.yml")
    scripts.clone_source("..", GitMode.cli, True, "rail_packages.yml")


def test_update_source():
    scripts.update_source("..", True, "rail_packages.yml")


def test_install():
    scripts.install("..", False, True, "rail_packages.yml")
    scripts.install("..", True, True, "rail_packages.yml")


def test_info():
    scripts.info(print_all=True)


def test_build_pipeline():
    scripts.build_pipeline(
        'rail.pipelines.estimation.train_z_pipeline.TrainZPipeline',
        'trainz_pipe.yaml',
        input_dict = dict(input='dummy.ibn'),
    )
