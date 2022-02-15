import numpy as np
from matplotlib import pyplot as plt, gridspec
import seaborn as sns
import warnings

import scipy.stats as stats


def plot_distribution(values: np.ndarray, perc=None, perc_pos=0, th=None, **kwargs):
    """
    Plots histogram, density and values sorted. If percentile parameter is set,
    it is also plotted the position from which the values account for that
    percentage of the total sum.

    Parameters:
        - values (np.array): list of values (1D).
        - perc (float): The percentage of the total sum of the values, or the
            position in the CDF from which to consider the values to extract.
        - th (float): The value in the distribution used as lower limit to
            compute the percentage of samples above it.
        - verbose(bool): Verbosity

    Return:
        - (threshold, position) (float, int): the value from which the cum
            sum accounts for the 'percentile' percentage of the total sum, and
            the position in the sorted list of values where that threshold is
            located.
    """
    compact = kwargs.get("compact", False)
    fsize = (11, 2) if compact else (8, 6)

    fig = plt.figure(tight_layout=True, figsize=fsize)
    gs = gridspec.GridSpec(1, 4) if compact else gridspec.GridSpec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(values, edgecolor="white", alpha=0.5, bins=25)
    ax1.set_title(f"Histogram (threshold={th:.2g})", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.kdeplot(values, ax=ax2, bw_adjust=0.5)
    ax2.set_title(f"Density (threshold={th:.2g})", fontsize=9)
    ax2.set(ylabel=None)

    if th > 0.0:
        ax1.axvline(th, linewidth=0.5, c="red", linestyle="dashed")
        ax2.axvline(th, linewidth=0.5, c="red", linestyle="dashed")

    ax3 = fig.add_subplot(gs[0, 2]) if compact else fig.add_subplot(gs[1, 0])
    x, y = np.arange(len(values)), np.sort(values)
    ax3.plot(x, y)
    if perc is not None:
        if perc < 1.0:
            ax3.set_title(f"{perc * 100:.0f}% of rev.cum.sum (>{th:.2f})", fontsize=9)
        else:
            cdf = (y[int(perc_pos) :].sum() / y.sum()) * 100.0
            ax3.set_title(
                f"Pos.{int(perc_pos)} (th. > {th:.2g}) = {cdf:.0f}%", fontsize=9
            )
        ax3.axvline(perc_pos, linewidth=0.5, c="red", linestyle="dashed")
        ax3.fill_between(x, min(y), y, where=x >= perc_pos, alpha=0.2)
    else:
        ax3.set_title("Ordered values", fontsize=9)

    ax4 = fig.add_subplot(gs[0, 3]) if compact else fig.add_subplot(gs[1, 1])
    xe = np.sort(values)
    ye = np.arange(1, len(xe) + 1) / float(len(xe))
    ax4.plot(xe, ye)
    if perc is not None:
        cdf = ye[np.max(np.where(xe < th))] * 100.0
        if perc < 1.0:
            ax4.set_title(
                f"rev.ECDF {perc * 100:.0f}% (th.> {th:.2g}) ~ {cdf:.0f}%",
                fontsize=9,
            )
        else:
            ax4.set_title(
                f"Pos.{int(perc_pos)} of rev.ECDF (th.>{th:.2g}) ~ {cdf:.0f}%",
                fontsize=9,
            )
        ax4.fill_between(xe, min(ye), ye, where=xe >= th, alpha=0.2)
        ax4.axvline(th, linewidth=0.5, c="red", linestyle="dashed")
    else:
        ax4.set_title("ECDF", fontsize=9)

    fig.align_labels()
    plt.tight_layout()
    plt.show()


def get_threshold(values, percentile=0.8, **kwargs):
    """
    Computes the value from which either: the accumulated sum of values represent
    the percentage passed as argument (<1), or the number of values in the lower range
    equals the value passed (>1). The value is computed sorting the values in
    descending order, so the this metric determines what are the most important values.

    Parameters:
        - values (np.array): List of values (1D) to analyze
        - percentile (float): The percentage of the total sum of the values.

    Returns:
        float with either the threshold in the values that account for the percentile
            passed, or the percentage of distribution above the threshold passed.

    Examples:
        >>>> a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>>> get_threshold(a, verbose=True)
        Values shape: 10
        Sum values: 55.00
        Sum@Percentile(0.80): 44.00
        Position @ percentile 0.80 in cum.sum: 6
        Threshold (under perc.0.80): 7.00
    """
    verbose = kwargs.get("verbose", False)
    sum_values = np.sum(values)
    cumsum = np.cumsum(sorted(values, reverse=True))
    if verbose:
        print("Computing threshold")
        print(f"Values shape: {values.shape[0]}")
        print(f"Sum values: {sum_values:.2f}")
        if percentile < 1.0:
            print(f"Sum@Percentile({percentile:.2f}): {sum_values * percentile:.2f}")
    # Substract because cumsum is reversed
    if percentile < 1.0:
        perc_sum = sum_values * percentile
        pos_q = values.shape[0] - len(np.where(cumsum < perc_sum)[0])
    else:
        pos_q = float(percentile)
    if pos_q == values.shape[0]:
        pos_q -= 1
    if verbose:
        if percentile < 1.0:
            print(f"Position @ percentile {percentile:.2f} in cum.sum: {pos_q}")
        else:
            print(f"Position in values: {int(pos_q)}")
    threshold = sorted(values)[int(pos_q)]
    if verbose:
        print(f"Threshold @ p. {percentile:.2f}): {threshold:.2f}")
    return threshold, pos_q


def get_percentile(values, threshold, **kwargs):
    """
    Computes the percentage of distribution that represents the values above a given
    threshold.

    Parameters:
        - values (np.array): List of values (1D) to analyze
        - threshold (float): The lower value to be considered in the list of values.

    Returns:
        float with either the threshold in the values that account for the percentile
            passed, or the percentage of distribution above the threshold passed.

    Examples:
        >>>> a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>>> get_percentile(a, 5, verbose=True)
    """
    sum_values = np.sum(values)
    verbose = kwargs.get("verbose", False)
    ord_values = np.array(sorted(values, reverse=True))
    new_values = ord_values[ord_values >= threshold]
    num_new_values = len(new_values)
    sum_new_values = np.sum(new_values)
    perc = sum_new_values / sum_values
    pos = values.shape[0] - num_new_values
    if verbose:
        print("Computing percentile")
        print(f"Sum values: {sum_values:.2f}")
        print(f"Len values above {threshold}: {num_new_values}")
        print(f"Position @ threshold: {pos}")
        print(f"Sum upper values: {sum_new_values:.2f}")
        print(f"Values above threshold: {perc:.2f}")

    return perc, pos


def get_boundaries(values, percentile=None, threshold=None, **kwargs):
    """
    Search for the threshold for a given percentile, the percentile for a given
    threshold, and plots the results if the corresponding flag is set to True.


    Args:
        values (np.array): the 1D values
        percentile (float or int): represents the percentage of the value, from
            right of the distribution to consider. If an integer, represents the
            position in the descending list of values to be used as lower boundary.
        threshold (float): A lower limit for the values in the distribution.
        **kwargs: 'plot', 'verbose'.

    Returns:
        The threshold, the percentage of distribution and the position in the
            descending list of values where those cutoffs have been found.
    """
    if percentile is not None:
        th, pos = get_threshold(values, percentile=percentile, **kwargs)
        perc = percentile
    elif threshold is not None:
        perc, pos = get_percentile(values, threshold=threshold, **kwargs)
        th = threshold
    else:
        perc = None
        th = np.min(values)
        pos = 0
    if kwargs.get("plot", True):
        plot_distribution(values, perc, pos, th, **kwargs)

    return th, perc, pos


def analyze_distribution(values, percentile=None, threshold=None, **kwargs):
    r"""
    Analyze the data to find what is the most suitable distribution type using
    Kolmogorov-Smirnov test.

    Arguments:
        values (np.array): List of values
        percentile (float): The percentile of cum sum down to which compute
            threshold. This argument is mutually exclusive with `threshold`.
        threshold (float): The value from which to select elements from the
            distribution. This argument is mutually exclusive with `percentile`.
        (Optional)
        plot (bool): Default is True
        verbose (bool): Default is False

    Return:
        Dictionary with keys: 'name' of the distribution, the
            'p_value' obtained in the test, the 'dist' itself as Callable, and
            the 'params' of the distribution. If parameter percentile is passed,
            the value from which the accumulated sum of values represent the
            percentage passed, under the key 'threshold'.
    """
    assert not (
        percentile is not None and threshold is not None
    ), "Both percentile and threshold cannot be specified at the same time."
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    verbose = kwargs.get('verbose', False)
    d = dict()
    d["threshold"], d["percentile"], d['pos'] = get_boundaries(
        values, percentile, threshold, **kwargs
    )

    best_pvalue = 0.0
    for dist_name in [
        "norm",
        "exponweib",
        "weibull_max",
        "weibull_min",
        "pareto",
        "genextreme",
    ]:
        dist = getattr(stats, dist_name)
        param = dist.fit(values)
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(values, dist_name, args=param)
        if p > best_pvalue:
            best_pvalue = p
            d['name'] = dist_name
            d['p_value'] = p
            d['dist'] = dist
            d['params'] = param
    if verbose:
        print(
            f"Best fitting distribution (p_val:{d['p_value']:.2f}): {d['name']}")
    return d
