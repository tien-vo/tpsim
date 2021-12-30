from ..chaos import gram_schmidt
import numpy as np


def test_orthogonality_and_normalization():

    for _ in range(10):
        M = np.random.randn(3, 3)
        W, V = gram_schmidt(M)
        # Test normalization, V.V^T should be the identity map
        assert np.isclose(np.matmul(V, V.T), np.identity(3)).all()

        # Test orthogonality, dot products of different column vectors should
        # be zero
        for j in range(W.shape[1]):
            Wj = W[:, j]
            for k in range(j + 1, W.shape[1]):
                Wk = W[:, k]
                assert np.isclose(np.dot(Wj, Wk), 0)


def test_with_worked_out_examples():

    #################
    # Sample matrices
    #################
    M = np.array(
        [
            # First matrix
            [
                [2, 0, 0],
                [1, 1, 0],
                [0, 0, 1],
            ],
            # Second matrix
            [
                [2, 1, 2],
                [1, 1, 0],
                [0, 1, 1],
            ],
        ]
    )
    ####################
    # Worked out results
    ####################
    W_test = np.array(
        [
            [
                [2, -2 / 5, 0],
                [1, 4 / 5, 0],
                [0, 0, 1],
            ],
            # Second matrix
            [
                [2, -1 / 5, 1 / 2],
                [1, 2 / 5, -1],
                [0, 1, 1 / 2],
            ],
        ]
    )
    V_test = np.array(
        [
            [
                [2 / np.sqrt(5), -1 / np.sqrt(5), 0],
                [1 / np.sqrt(5), 2 / np.sqrt(5), 0],
                [0, 0, 1],
            ],
            # Second matrix
            [
                [2 / np.sqrt(5), -1 / np.sqrt(30), 1 / np.sqrt(6)],
                [1 / np.sqrt(5), 2 / np.sqrt(30), -np.sqrt(6) / 3],
                [0, 5 / np.sqrt(30), 1 / np.sqrt(6)],
            ],
        ]
    )

    # Test non-vectorized code
    for i in range(M.shape[0]):
        W, V = gram_schmidt(M[i])
        assert np.isclose(W, W_test[i], rtol=1e-10).all()
        assert np.isclose(V, V_test[i], rtol=1e-10).all()

    # Test vectorized code
    W, V = gram_schmidt(M)
    assert np.isclose(W, W_test, rtol=1e-10).all()
    assert np.isclose(V, V_test, rtol=1e-10).all()
