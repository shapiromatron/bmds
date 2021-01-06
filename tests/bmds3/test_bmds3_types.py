from bmds.bmds3.types.common import residual_of_interest


def test_residual_of_interest():
    # failure case
    assert residual_of_interest(bmd=-9999, doses=[1, 2, 3], residuals=[0.1, 0.2, 0.3]) == -9999

    # above value
    assert residual_of_interest(bmd=2.1, doses=[1, 2, 3], residuals=[0.1, 0.2, 0.3]) == 0.2

    # below value
    assert residual_of_interest(bmd=1.9, doses=[1, 2, 3], residuals=[0.1, 0.2, 0.3]) == 0.2

    # exactly between value; chooses smaller value
    assert residual_of_interest(bmd=1, doses=[0, 2, 4], residuals=[0.1, 0.2, 0.3]) == 0.1
