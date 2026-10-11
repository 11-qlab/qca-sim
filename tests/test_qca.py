from qca import majority, QCACell


def test_majority():
    assert majority(1, 1, 0) == 1
    assert majority(1, 0, 0) == 0


def test_cell_switch():
    c = QCACell(-1.0)
    c.switch(1.0)
    assert c.polarization == 1.0
