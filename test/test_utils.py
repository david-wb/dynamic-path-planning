import src.utils as utils


def test_circle_touches_rect():
    rect = (-10, 10, 20, 20)

    assert utils.circle_touches_rect(0, 0, 1, *rect)
    assert not utils.circle_touches_rect(13, 13, 1, *rect)
