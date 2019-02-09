"""
Hermite Genz-Keister 24 rule.
"""
import numpy


def quad_genz_keister_24 ( order ):
    """
    Hermite Genz-Keister 24 rule.

    Args:
        order (int):
            The quadrature order. Must be in the interval (0, 8).

    Returns:
        (:py:data:typing.Tuple[numpy.ndarray, numpy.ndarray]):
            Abscissas and weights

    Examples:
        >>> abscissas, weights = quad_genz_keister_24(1)
        >>> print(numpy.around(abscissas, 4))
        [-1.7321  0.      1.7321]
        >>> print(numpy.around(weights, 4))
        [0.1667 0.6667 0.1667]
    """
    order = sorted(GENZ_KEISTER_24.keys())[order]

    abscissas, weights = GENZ_KEISTER_24[order]
    abscissas = numpy.array(abscissas)
    weights = numpy.array(weights)

    weights /= numpy.sum(weights)
    abscissas *= numpy.sqrt(2)

    return abscissas, weights


GENZ_KEISTER_24 = {
    1 : ((
        0.0000000000000000,
    ), (
        1.7724538509055159E+00,
    )),
    3 : ((
        -1.2247448713915889,
        0.0000000000000000,
        1.2247448713915889,
    ), (
        2.9540897515091930E-01,
        1.1816359006036772E+00,
        2.9540897515091930E-01,
    )),
    9 : ((
        -2.9592107790638380,
        -2.0232301911005157,
        -1.2247448713915889,
        -0.52403354748695763,
        0.0000000000000000,
        0.52403354748695763,
        1.2247448713915889,
        2.0232301911005157,
        2.9592107790638380,
    ), (
        1.6708826306882348E-04,
        1.4173117873979098E-02,
        1.6811892894767771E-01,
        4.7869428549114124E-01,
        4.5014700975378197E-01,
        4.7869428549114124E-01,
        1.6811892894767771E-01,
        1.4173117873979098E-02,
        1.6708826306882348E-04,
    )),
    19 : ((
        -4.4995993983103881,
        -3.6677742159463378,
        -2.9592107790638380,
        -2.2665132620567876,
        -2.0232301911005157,
        -1.8357079751751868,
        -1.2247448713915889,
        -0.87004089535290285,
        -0.52403354748695763,
        0.0000000000000000,
        0.52403354748695763,
        0.87004089535290285,
        1.2247448713915889,
        1.8357079751751868,
        2.0232301911005157,
        2.2665132620567876,
        2.9592107790638380,
        3.6677742159463378,
        4.4995993983103881,
    ), (
        1.5295717705322357E-09,
        1.0802767206624762E-06,
        1.0656589772852267E-04,
        5.1133174390883855E-03,
        -1.1232438489069229E-02,
        3.2055243099445879E-02,
        1.1360729895748269E-01,
        1.0838861955003017E-01,
        3.6924643368920851E-01,
        5.3788160700510168E-01,
        3.6924643368920851E-01,
        1.0838861955003017E-01,
        1.1360729895748269E-01,
        3.2055243099445879E-02,
        -1.1232438489069229E-02,
        5.1133174390883855E-03,
        1.0656589772852267E-04,
        1.0802767206624762E-06,
        1.5295717705322357E-09,
    )),
    43 : ((
        -10.167574994881873,
        -7.231746029072501,
        -6.535398426382995,
        -5.954781975039809,
        -5.434053000365068,
        -4.952329763008589,
        -4.4995993983103881,
        -4.071335874253583,
        -3.6677742159463378,
        -3.295265921534226,
        -2.9592107790638380,
        -2.633356763661946,
        -2.2665132620567876,
        -2.089340389294661,
        -2.0232301911005157,
        -1.8357079751751868,
        -1.583643465293944,
        -1.2247448713915889,
        -0.87004089535290285,
        -0.52403354748695763,
        -0.196029453662011,
        0.0000000000000000,
        0.196029453662011,
        0.52403354748695763,
        0.87004089535290285,
        1.2247448713915889,
        1.583643465293944,
        1.8357079751751868,
        2.0232301911005157,
        2.089340389294661,
        2.2665132620567876,
        2.633356763661946,
        2.9592107790638380,
        3.295265921534226,
        3.6677742159463378,
        4.071335874253583,
        4.4995993983103881,
        4.952329763008589,
        5.434053000365068,
        5.954781975039809,
        6.535398426382995,
        7.231746029072501,
        10.167574994881873,
    ), (
        0.546191947478318097E-37,
        0.87544909871323873E-23,
        0.992619971560149097E-19,
        0.122619614947864357E-15,
        0.421921851448196032E-13,
        0.586915885251734856E-11,
        0.400030575425776948E-09,
        0.148653643571796457E-07,
        0.316018363221289247E-06,
        0.383880761947398577E-05,
        0.286802318064777813E-04,
        0.184789465688357423E-03,
        0.150909333211638847E-02,
        - 0.38799558623877157E-02,
        0.67354758901013295E-02,
        0.139966252291568061E-02,
        0.163616873493832402E-01,
        0.450612329041864976E-01,
        0.928711584442575456E-01,
        0.145863292632147353E+00,
        0.164880913687436689E+00,
        0.579595986101181095E-01,
        0.164880913687436689E+00,
        0.145863292632147353E+00,
        0.928711584442575456E-01,
        0.450612329041864976E-01,
        0.163616873493832402E-01,
        0.139966252291568061E-02,
        0.67354758901013295E-02,
        - 0.38799558623877157E-02,
        0.150909333211638847E-02,
        0.184789465688357423E-03,
        0.286802318064777813E-04,
        0.383880761947398577E-05,
        0.316018363221289247E-06,
        0.148653643571796457E-07,
        0.400030575425776948E-09,
        0.586915885251734856E-11,
        0.421921851448196032E-13,
        0.122619614947864357E-15,
        0.992619971560149097E-19,
        0.87544909871323873E-23,
        0.546191947478318097E-37,
    ))
}