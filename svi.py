# Port (attempt) from R code to compute SVI

import numpy as np
from math import pi
import scipy.optimize as sop


def raw_svi(par, k):
    """
    Returns total variance for a given set of parameters from RAW SVI
    parametrization at given moneyness points.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Total variance
    """
    w = par[0] + par[1] * (par[2] * (k - par[3]) + (
                (k - par[3]) ** 2 + par[4] ** 2) ** 0.5)
    return w


def diff_svi(par, k):
    """
    First derivative of RAW SVI with respect to moneyness.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: First derivative evaluated at k points
    """
    a, b, rho, m, sigma = par
    return b*(rho+(k-m)/(np.sqrt((k-m)**2+sigma**2)))


def diff2_svi(par, k):
    """
    Second derivative of RAW SVI with respect to moneyness.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Second derivative evaluated at k points
    """
    a, b, rho, m, sigma = par
    disc = (k-m)**2 + sigma**2
    return (b*sigma**2)/((disc)**(3/2))


def gfun(par, k):
    """
    Computes the g(k) function. Auxiliary to retrieve implied density and
    essential to test for butterfly arbitrage.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Function g(k) evaluated at k points
    """
    w = raw_svi(par, k)
    w1 = diff_svi(par, k)
    w2 = diff2_svi(par, k)

    g = (1-0.5*(k*w1/w))**2 - (0.25*w1**2)*(w**-1+0.25) + 0.5*w2
    return g


def d1(par, k):
    """
    Auxiliary function to compute d1 from BSM model.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Values of d1 evaluated at k points
    """
    v = np.sqrt(raw_svi(par, k))
    return -k/v + 0.5*v


def d2(par, k):
    """
    Auxiliary function to compute d2 from BSM model.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Values of d2 evaluated at k points
    """
    v = np.sqrt(raw_svi(par, k))
    return -k/v - 0.5*v

def density(par, k):
    """
    Probability density implied by an SVI.
    @param par: Set of raw parameters, (a, b, rho, m, sigma)
    @type k: PdSeries
    @param k: Moneyness points to evaluate
    @return: Implied risk neutral probability density from an SVI
    """
    g = gfun(par, k)
    w = raw_svi(par, k)
    dtwo = d2(par, k)

    dens = (g / np.sqrt(2 * pi * w)) * np.exp(-0.5 * dtwo**2)
    return dens


def rmse(w, k, param):
    """
    Returns root mean square error of a RAW SVI parametrization.
    @type w: PdSeries
    @param w: Market total variance
    @param k: Moneyness
    @param param: List of parameters (a, b, rho, m, sigma)
    @return: A float number representing the RMSE
    """
    return np.mean(np.sqrt((raw_svi(param, k)-w)**2))


def wrmse(w, k, param, weights):
    """
    Weighted RMSE
    :param w: Market total variance
    :param k: Moneyness
    :param param: List of parameters (a, b, rho, m, sigma)
    :param weights: Weights. Do not need to sum to 1
    :return: A float number representing the weighted RMSE
    """
    sum_w = np.sum(weights)
    return (1/sum_w) * np.mean(weights*(raw_svi(param, k)-w)**2)



# a 参数 ((0., max(w))):
# 含义: a 参数通常代表了一个平移参数，它定义了 SVI 曲线的最小值，也可以被视为平坦的波动率部分。
# 取值范围: 由于 a 是一个方差参数，且波动率不能为负，所以下界是 0 (min(w))。上界被设为市场总方差 w 的最大值 max(w)，这样设置是为了保证拟合的 SVI 曲线不会超出合理的市场波动范围。

# b 参数 ((0., 1.)):
# 含义: b 参数是与斜率有关的参数，它影响曲线的陡峭度。
# 取值范围: 斜率参数 b 通常被限定在 [0, 1] 之间。0 代表没有斜率，即平坦的曲线，而 1 是一个经验上合理的上限，用于确保模型不会变得过于陡峭而失去稳定性。

# rho 参数 ((-1., 1.)):
# 含义: rho 是相关系数，控制 SVI 曲线的对称性或不对称性。
# 取值范围: rho 代表了曲线的偏斜方向。rho = -1 表示最大负偏斜，而 rho = 1 表示最大正偏斜。限定 rho 在 [-1, 1] 范围内确保了合理的偏斜，而不会产生不切实际的模型行为。

# m 参数 ((2*min(k), 2*max(k))):
# 含义: m 是与曲线中心相关的参数，代表行权价的中心。
# 取值范围: m 的范围设为 2*min(k) 到 2*max(k)，以确保曲线的中心在实际数据范围内。这种设置允许 m 覆盖行权价范围的两倍，以保证拟合的灵活性。

# sigma 参数 ((0., 1.)):
# 含义: sigma 是与波动率扩展相关的参数，影响曲线的宽度。
# 取值范围: sigma 是与波动率相关的参数，通常取值范围为 [0, 1]，这使得波动率处于一个合理的范围内，既不会过窄，也不会过宽。


def svi_fit_direct(k, w, weights, method, ngrid):
    """
    Direct procedure to calibrate a RAW SVI
    :param k: Moneyness
    :param w: Market total variance
    :param weights: Weights. Do not need to sum to 1
    :param method: Method of optimization. Currently implemented: "brute", "DE"
    :param ngrid: Number of grid points for each parameter in brute force
    algorithm
    :return: SVI parameters (a, b, rho, m, sigma)
    """
    # Bounds on parameters
    bounds = [(min(w), max(w)), # a
               (0., 1.), # b       
               (-1., 1.), # rho
               (2*min(k), 2*max(k)), # m
               (0., 1.)] # sigma
    
    # Tuples for brute force
    bfranges = tuple(bounds)

    # Objective function
    def obj_fun_wrmse(par):
        return wrmse(w, k, par, weights)
    
    def obj_fun_rmse(par):
        return rmse(w, k, par)

    # Chooses algo and run
    if method == "brute":
        # Brute force algo. fmin will take place to refine solution
        p0 = sop.brute(obj_fun_wrmse, bfranges, Ns=ngrid, full_output=True)
        return p0
    elif method == "DE":
        # Differential Evolution algo.
        p0 = sop.differential_evolution(obj_fun_wrmse, bounds)
        return p0
    else:
        print("Unknown method passed to svi_fit_direct.")
        raise