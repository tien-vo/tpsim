from ..mpi_helpers import chunk
import numpy as np


def test_even_chunks():
    """When data is chunked evenly"""
    data_size = 8
    world_size = 4
    # Calculated
    counts, disps = chunk(data_size, world_size)
    # Expected
    counts_exp = np.array([2, 2, 2, 2])
    disps_exp = np.array([0, 2, 4, 6])
    assert (counts == counts_exp).all()
    assert (disps == disps_exp).all()


def test_uneven_chunks():
    """When data is chunked unevenly"""
    data_size = 9
    world_size = 4
    # Calculated
    counts, disps = chunk(data_size, world_size)
    # Expected
    counts_exp = np.array([3, 2, 2, 2])
    disps_exp = np.array([0, 3, 5, 7])
    assert (counts == counts_exp).all()
    assert (disps == disps_exp).all()


def test_too_large_chunks():
    """When data size can fit into a few processes"""
    # Expected
    counts_exp = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]
    )
    disps_exp = np.array(
        [
            [0, 1, 1, 1],
            [0, 1, 2, 2],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
    )
    for (i, data_size) in enumerate([1, 2, 3, 4]):
        world_size = 4
        # Calculated
        counts, disps = chunk(data_size, world_size)
        assert (counts == counts_exp[i]).all()
        assert (disps == disps_exp[i]).all()
