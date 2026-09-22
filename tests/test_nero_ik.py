from pathlib import Path

import numpy as np

from robot_control.nero_ik import NeroIK, URDF_PATH


def test_nero_urdf_is_project_asset():
    expected = Path(__file__).resolve().parents[1] / "assets" / "nero" / "nero_description.urdf"
    assert URDF_PATH.resolve() == expected.resolve()
    assert expected.is_file()


def test_nero_fk_identity_shape_and_limits():
    solver = NeroIK()
    assert solver.forward(np.zeros(7)).shape == (4, 4)
    assert solver.limits.shape == (7, 2)

