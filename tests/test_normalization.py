import pytest


@pytest.mark.parametrize("norm, expected_result", [
    ('diet', [0.375, 0.5, 0.5, 0.625]),
    ('log', [0.47712125471966244, 0.3010299956639812, 0.3010299956639812, 0.6989700043360189]),
    ('donor_control', [3.0, 2.0, 0.6666666666666666, 1.0]),
    ('predator_control', [1.0, 0.4, 0.4, 1.6666666666666667]),
    ('mixed_control', [3.0, 0.8, 0.26666666666666666, 1.6666666666666667]),
    ('tst', [0.25, 0.16666666666666666, 0.16666666666666666, 0.4166666666666667])

])
def test_get_normalized_flows(food_web, norm, expected_result):
    weights = [data['weight'] for _, _, data in food_web.get_flows(normalization=norm)]
    assert weights == expected_result
