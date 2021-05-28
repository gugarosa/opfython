import numpy as np

from opfython.math import distance


def test_additive_symmetric_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.additive_symmetric_distance(x, y)

    assert dist == 0.4813445378151263


def test_average_euclidean_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.average_euclidean_distance(x, y)

    assert dist == 0.2236067977499792


def test_bhattacharyya_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.bhattacharyya_distance(x, y)

    assert dist == -2.3499617065547542


def test_bray_curtis_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.bray_curtis_distance(x, y)

    assert dist == 0.03809523809523813


def test_canberra_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.canberra_distance(x, y)

    assert dist == 0.33983837574300413


def test_chebyshev_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.chebyshev_distance(x, y)

    assert dist == 0.3000000000000007


def test_chi_squared_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.chi_squared_distance(x, y)

    assert dist == 0.029526480999131792


def test_chord_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.chord_distance(x, y)

    assert dist == 0.05642296490643823


def test_clark_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.clark_distance(x, y)

    assert dist == 0.22448075858553082


def test_cosine_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.cosine_distance(x, y)

    assert dist == 0.00159177548441658


def test_dice_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.dice_distance(x, y)

    assert dist == 0.0023820867079562547


def test_divergence_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.divergence_distance(x, y)

    assert dist == 0.10078322195027073


def test_euclidean_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.euclidean_distance(x, y)

    assert dist == 0.4472135954999584


def test_gaussian_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.gaussian_distance(x, y)

    assert dist == 0.6394073191618967


def test_gower_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.gower_distance(x, y)

    assert dist == 0.2000000000000002


def test_hamming_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.hamming_distance(x, y)

    assert dist == 4


def test_hassanat_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.hassanat_distance(x, y)

    assert dist == 0.2571314102564104


def test_hellinger_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.hellinger_distance(x, y)

    assert dist == 0.2435717239311041


def test_jaccard_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.jaccard_distance(x, y)

    assert dist == 0.004752851711026626


def test_jeffreys_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.jeffreys_distance(x, y)

    assert np.round(dist, 5) == 0.11884


def test_jensen_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.jensen_distance(x, y)

    assert np.round(dist, 5) == 0.00740


def test_jensen_shannon_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.jensen_shannon_distance(x, y)

    assert np.round(dist, 5) == 0.01481


def test_k_divergence_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.k_divergence_distance(x, y)

    assert dist == -0.18527516196697194


def test_kulczynski_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.kulczynski_distance(x, y)

    assert dist == 0.07920792079207929


def test_kullback_leibler_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.kullback_leibler_distance(x, y)

    assert np.round(dist, 5) == -0.34023


def test_log_euclidean_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.log_euclidean_distance(x, y)

    assert dist == 36964.004940249884


def test_log_squared_euclidean_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.log_squared_euclidean_distance(x, y)

    assert dist == 18232.155679395495


def test_lorentzian_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.lorentzian_distance(x, y)

    assert dist == 0.7153488885436325


def test_manhattan_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.manhattan_distance(x, y)

    assert dist == 0.8000000000000008


def test_matusita_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.matusita_distance(x, y)

    assert dist == 0.17223121769698138


def test_max_symmetric_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.max_symmetric_distance(x, y)

    assert dist == 0.12254901960784322


def test_mean_censored_euclidean_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.mean_censored_euclidean_distance(x, y)

    assert dist == 0.2236067977499792


def test_min_symmetric_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.min_symmetric_distance(x, y)

    assert dist == 0.11812324929971998


def test_neyman_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.neyman_distance(x, y)

    assert dist == 0.11812324929971998


def test_non_intersection_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.non_intersection_distance(x, y)

    assert dist == 0.4000000000000004


def test_pearson_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.pearson_distance(x, y)

    assert dist == 0.12254901960784322


def test_sangvi_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.sangvi_distance(x, y)

    assert dist == 0.11810592399652717


def test_soergel_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.soergel_distance(x, y)

    assert dist == 0.07339449541284411


def test_squared_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.squared_distance(x, y)

    assert dist == 0.059052961998263584


def test_squared_chord_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.squared_chord_distance(x, y)

    assert dist == 0.029663592349384996


def test_squared_euclidean_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.squared_euclidean_distance(x, y)

    assert dist == 0.20000000000000046


def test_statistic_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.statistic_distance(x, y)

    assert dist == 0.08914713150337261


def test_topsoe_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.topsoe_distance(x, y)

    assert np.round(dist, 5) == 0.02962


def test_vicis_symmetric1_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.vicis_symmetric1_distance(x, y)

    assert dist == 0.3002436268625096


def test_vicis_symmetric2_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.vicis_symmetric2_distance(x, y)

    assert dist == 0.134873949579832


def test_vicis_symmetric3_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.vicis_symmetric3_distance(x, y)

    assert dist == 0.10579831932773118


def test_vicis_wave_hedges_distance():
    x = np.asarray([5.1, 3.5, 1.4, 0.3])
    y = np.asarray([5.4, 3.4, 1.7, 0.2])

    dist = distance.vicis_wave_hedges_distance(x, y)

    assert dist == 0.8025210084033614
