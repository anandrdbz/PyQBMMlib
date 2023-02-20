import sys

sys.path.append("../src/")
sys.path.append("../utils/")
from qbmm_manager import *
from stats_util import *
import numpy.polynomial.hermite as hermite_poly
import pytest
import copy
import matplotlib.pyplot as plt


def test_project():
    tol = 1.0e-10
    success = True

    # 1D
    config = {}
    config["qbmm"] = {}
    config["qbmm"]["governing_dynamics"] = "4*x - 2*x**2"
    config["qbmm"]["num_internal_coords"] = 1
    config["qbmm"]["num_quadrature_nodes"] = 3
    config["qbmm"]["method"] = "qmom"
    qbmm_mgr = qbmm_manager(config)

    mu = 0.0
    sig = 1.1
    init_moments = raw_gaussian_moments_univar(qbmm_mgr.num_moments, mu, sig)
    abscissas, weights = qbmm_mgr.moment_invert(init_moments)
    projected_moments = qbmm_mgr.projection(weights, abscissas, qbmm_mgr.indices)
    err_1D = np.linalg.norm(projected_moments - init_moments)

    # 2D
    """
    config = {}
    config["qbmm"] = {}
    config["qbmm"]["governing_dynamics"] = " - 1.5*xdot**2/x - 4*0.01*xdot/x - 2*0.1/x - 80 + 100 / x**5.2  "
    config["qbmm"]["num_internal_coords"] = 2
    config["qbmm"]["num_quadrature_nodes"] = 4
    config["qbmm"]["method"] = "chyqmom"
    qbmm_mgr = qbmm_manager(config)


    mu = [1.0, 0.0]
    sig = [0.1, 0.01]
    init_moments = gaussian_moments_bivar(
        qbmm_mgr.indices, mu[0], mu[1], sig[0], sig[1]
    )
    abscissas, weights = qbmm_mgr.moment_invert(init_moments, qbmm_mgr.indices)
    projected_moments = qbmm_mgr.projection(weights, abscissas, qbmm_mgr.indices)

    err_2D = np.linalg.norm(projected_moments - init_moments)
    moments = init_moments.copy()
    rhs = 0*moments
    dt = 1e-6

    tfinal = 13719

    L = np.array([])


    for i in range(tfinal):
        print(moments)
        print(i)
        L = np.append(L, moments[1])

        init_moments = moments.copy()
        qbmm_mgr.compute_rhs(moments, rhs)
        projected_moments = moments.copy()
        err_2D = np.linalg.norm(projected_moments - init_moments)
        

        moments = moments + dt*rhs

    """
    config = {}
    config["qbmm"] = {}
    config["qbmm"]["governing_dynamics"] = " - 1.5*xdot**2/x - 4*0.01*xdot/x**2 - 2*0.1/x**2 - 80/x + 100 / x**5.2  "
    config["qbmm"]["num_internal_coords"] = 2
    config["qbmm"]["num_quadrature_nodes"] = 6
    config["qbmm"]["method"] = "cqmom_avg"
    qbmm_mgr = qbmm_manager(config)
    tfinal = 50000


    mu = [1.0, 0.0]
    sig = [0.1, 0.1]
    init_moments = gaussian_moments_bivar(
        qbmm_mgr.indices, mu[0], mu[1], sig[0], sig[1]
    )
    abscissas, weights = qbmm_mgr.moment_invert(init_moments, qbmm_mgr.indices)
    projected_moments = qbmm_mgr.projection(weights, abscissas, qbmm_mgr.indices)

    err_2D = np.linalg.norm(projected_moments - init_moments)
    state = init_moments.copy()
    rhs = 0*init_moments
    time_step = 1e-8

    L2 = np.array([])
    stage_state = np.zeros((3, np.size(init_moments)))
    stage_k = 0 * stage_state


    error_tol = 1e-9


    for i in range(tfinal):
        #print(moments)
        #print(i)
        print(i)
        

        stage_state[0] = state.copy()
        qbmm_mgr.compute_rhs(stage_state[0], stage_k[0])
        stage_state[1] = stage_state[0] + time_step * stage_k[0]

        # Stage 2: { y_2, k_2 } = f( t_n, y_1 + dt * k_1 )
        qbmm_mgr.compute_rhs(stage_state[1], stage_k[1])
        test_state = 0.5 * (
            stage_state[0]
            + (stage_state[1] + time_step * stage_k[1])
        )

        # Stage 3: { y_3, k_3 } = f( t_n + 0.5 * dt, ... )
        stage_state[2] = 0.75 * stage_state[0] + 0.25 * (
            stage_state[1] + time_step * stage_k[1]
        )
        qbmm_mgr.compute_rhs(stage_state[2], stage_k[2])

        # Updates
        state = (
            stage_state[0]
            + 2.0 * (stage_state[2] + time_step * stage_k[2])
        ) / 3.0

        ts_error = np.linalg.norm(state - test_state) / np.linalg.norm(
            state
        )

        error_fraction = np.sqrt(0.5 * error_tol / ts_error)
        time_step_factor = min(max(error_fraction, 0.3), 2.0)
        new_time_step = time_step_factor * time_step
        new_time_step = min(
            max(0.9 * new_time_step, 1e-8), 1e5
        )
        time_step = new_time_step
        print(time_step)

        L2 = np.append(L2, state[1])

        print(state)


    plt.plot(np.arange(tfinal), L2, label = "CHYQMOM")
    #plt.plot(np.arange(tfinal), L2, label = "CQMOM")
    plt.xlabel("tstep")
    plt.ylabel("R")
    plt.legend()
    plt.show()

    
    

    

    assert err_1D < tol
    assert err_2D < tol




def test_wheeler():
    """
    This function tests QBMM Wheeler inversion by comparing
    against numpy's Gauss-Hermite for given mu and sigma
    """
    num_nodes = 4
    config = {}
    config["qbmm"] = {}
    config["qbmm"]["governing_dynamics"] = "4*x - 2*x**2"
    config["qbmm"]["num_internal_coords"] = 1
    config["qbmm"]["num_quadrature_nodes"] = num_nodes
    config["qbmm"]["method"] = "qmom"
    ###
    ### QBMM
    qbmm_mgr = qbmm_manager(config)

    ###
    ### Tests

    # Anticipate success
    tol = 1.0e-10  # Round-off error in moments computation
    success = True

    # Test 1
    mu = 5.0
    sigma = 1.0
    ###
    ### Reference solution
    sqrt_pi = np.sqrt(np.pi)
    sqrt_two = np.sqrt(2.0)
    h_abs, h_wts = hermite_poly.hermgauss(num_nodes)
    g_abs = sqrt_two * sigma * h_abs + mu
    g_wts = h_wts / sqrt_pi

    ###
    ### QBMM
    moments = raw_gaussian_moments_univar(qbmm_mgr.num_moments, mu, sigma)
    my_abs, my_wts = qbmm_mgr.moment_invert(moments)

    ###
    ### Errors & Report
    diff_abs = my_abs - g_abs
    diff_wts = my_wts - g_wts

    error_abs = np.linalg.norm(my_abs - g_abs)
    error_wts = np.linalg.norm(my_wts - g_wts)

    assert error_abs < tol
    assert error_wts < tol


if __name__ == "__main__":

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main

        main([__file__])
