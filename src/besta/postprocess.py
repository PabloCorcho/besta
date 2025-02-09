"""
Module dedicated to the post-processing of data produced by BESTA.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from astropy.io import fits
from scipy import stats

pct_cmap = plt.get_cmap("rainbow").copy()


def weighted_sample_mean(x, weights):
    """Compute the weighted mean of an input sample.

    Parameters
    ----------
    x : np.ndarray
        Array of sample values. The last dimension must correspond
        to the sample dimension.
    weights : np.ndarray
        1D array of weights associated to the sample ``x``.

    Returns
    -------
    mean : np.array
        Weighted mean of the sample
    """
    return np.sum(x * weights, axis=-1)


def weighted_sample_covariance(x, weights, unbiased=False):
    """Compute the covariance matrix of a sample with weights.

    Parameters
    ----------
    x : np.ndarray
        Array of sample values. The last dimension must correspond
        to the sample dimension.
    weights : np.ndarray
        1D array of weights associated to the sample ``x``.

    Returns
    -------
    covariance : np.array
        Weighted mean of the sample
    """
    mean = weighted_sample_mean(x, weights)
    x_mean = x - mean[:, np.newaxis]
    covariance = np.sum(
        weights[np.newaxis, np.newaxis] * x_mean[np.newaxis] * x_mean[:, np.newaxis],
        axis=-1,
    )
    if unbiased:
        covariance /= 1 - np.sum(weights**2)
    return covariance


def weighted_1d_cmf(x, weights):
    """Compute the cumulative probability distribution from a sample."""
    sort_idx = np.argsort(x)
    return x[sort_idx], sort_idx, np.cumsum(weights[sort_idx])


def read_results_file(path):
    """Read the results produced during a cosmosis run.

    Parameters
    ----------
    path : str
        Path to the file containing the cosmosis results

    Returns
    -------
    table : :class:`astropy.table.Table`
        Table containing the results.
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip("#")
        columns = header.replace("\n", "").split("\t")
    matrix = np.atleast_2d(np.loadtxt(path))
    table = Table()
    if matrix.size > 1:
        for ith, c in enumerate(columns):
            table.add_column(matrix.T[ith], name=c.lower())
    return table


def compute_fraction_from_map(distribution, xedges=None, yedges=None):
    """Compute the enclosed probability map associated to a given 2D distribution.

    Parameters
    ----------
    distribution : np.ndarray
        Input 2D probability distribution.
    xedges : np.ndarray
        Edges along the second axis (columns)
    yedges : np.ndarray
        Edges along the first axis (rows)
    """
    if xedges is None:
        distribution_mass = distribution
    else:
        distribution_mass = (
            distribution
            * np.diff(xedges)[:, np.newaxis]
            * np.diff(yedges)[np.newaxis, :]
        )
    sorted_flat = np.argsort(distribution, axis=None)
    sorted_2D = np.unravel_index(sorted_flat, distribution.shape)
    density_sorted = distribution.flatten()[sorted_flat]
    cumulative_mass = np.cumsum(distribution_mass[sorted_2D])
    fraction_sorted = cumulative_mass / cumulative_mass[-1]
    fraction = np.interp(distribution, density_sorted, fraction_sorted)
    return fraction


def compute_pdf_from_results(
    table,
    output_filename=None,
    parameter_prefix="parameters",
    posterior_key="post",
    parameter_keys=None,
    pdf_1d=True,
    percentiles=[0.05, 0.16, 0.5, 0.84, 0.95],
    pdf_2d=False,
    parameter_key_pairs=None,
    pdf_size=100,
    plot=False,
    show=False,
    real_values=None,
    extra_info={},
):
    """Compute the PDF from a given results table.

    Parameters
    ----------
    output_filename : str
        Filename of the output FITS file.
    parameter_prefix : str, optional
    posterior_key : str, optional
    weights_key : str, optional
    parameter_keys : list, optional
    pdf_1d : bool, optional
    percentiles : list, optional
    pdf_2d : bool, optional
    parameter_key_pairs : list, optional
    pdf_size : int, optional
    plot : bool, optional
    real_values : dict, optional
    extra_info : dict, optional
    """
    # If not provided, select the keys that correspond to sampled parameters
    if parameter_keys is None:
        print(
            "No keys provided"
            + f"\nSelecting every column that includes prefix *{parameter_prefix}*"
        )
        parameter_keys = [key for key in list(table.keys()) if parameter_prefix in key]
        print("Number of keys found ", len(parameter_keys))

    assert len(parameter_keys) > 0, "No keys available"

    # Create the array of parameter values (theta, samples)
    values = np.array([table[key].value for key in parameter_keys])

    output_hdul = []
    header = fits.Header()

    max_logpost, max_post_idx = (
        np.nanmax(table[posterior_key]),
        np.nanargmax(table[posterior_key]),
    )
    # Linear posterior renormalized to the maximum value
    logpost = table[posterior_key].value
    posterior = np.exp(logpost - max_logpost)
    posterior /= np.nansum(posterior)
    header["hierarch max_logpost"] = max_logpost, "Maximum of the logposterior"
    maxpost_values = values[:, max_post_idx]
    # Mean and covariance
    mean_values = weighted_sample_mean(values, posterior)
    covariance_matrix = weighted_sample_covariance(values, posterior)
    # Store everything on the header
    for axis, mean, maxpost, key in zip(
        range(len(parameter_keys)), mean_values, maxpost_values, parameter_keys
    ):
        kname = key.replace(parameter_prefix + "--", "")
        header[f"hierarch axis_{axis}"] = kname, "parameter"
        header[f"hierarch mean_{kname}"] = mean, "post-weighted mean"
        header[f"hierarch maxpost_{kname}"] = maxpost, "max-post value"

    covariance_hdu = fits.ImageHDU(
        data=covariance_matrix, name="COVARIANCE", header=header
    )
    output_hdul.append(covariance_hdu)

    # PDF analysis
    if pdf_1d:
        table_1d_pdf = Table()
        table_1d_percentiles = Table()
        table_1d_percentiles.add_column(percentiles, name="percentiles")
        table_1d_pct_hdr = fits.Header()

        for key in parameter_keys:
            value = table[key].value
            mask = np.isfinite(value)
            value_sorted, sort_idx, cmf = weighted_1d_cmf(value[mask], posterior[mask])

            value_pct = np.interp(percentiles, cmf, value_sorted)
            logpost_pct = np.interp(percentiles, cmf, logpost[sort_idx])
            # TODO: duplicated
            value_mean = np.sum(posterior[mask] * value[mask])

            pdf_binedges = np.linspace(value_sorted[0], value_sorted[-1], pdf_size + 1)
            pdf_bins = (pdf_binedges[1:] + pdf_binedges[:-1]) / 2
            interp_cmf = np.interp(pdf_binedges, value_sorted, cmf)
            pdf = (interp_cmf[1:] - interp_cmf[:-1]) / (
                pdf_binedges[1:] - pdf_binedges[:-1]
            )

            key_name = key.replace(parameter_prefix + "--", "")
            try:
                kde = stats.gaussian_kde(value[mask], weights=posterior[mask])
                kde_pdf = kde(pdf_bins)
            except Exception as e:
                print("There was an error during KDE estimation: ", e)
                kde_pdf = np.full_like(pdf, fill_value=np.nan)

            table_1d_pdf.add_column(pdf_bins, name=f"{key_name}_bin")
            table_1d_pdf.add_column(pdf, name=f"{key_name}_pdf")
            table_1d_pdf.add_column(kde_pdf, name=f"{key_name}_pdf_kde")

            table_1d_percentiles.add_column(value_pct, name=f"{key_name}_pct")
            table_1d_percentiles.add_column(logpost_pct, name=f"{key_name}_post_pct")

            if real_values is not None and key in real_values:
                integral_to_real = np.interp(real_values[key], value_sorted, cmf)
                table_1d_pct_hdr[f"hierarch {key_name}_real"] = np.nan_to_num(
                    real_values[key]
                )
                table_1d_pct_hdr[f"hierarch {key_name}_int_to_real"] = np.nan_to_num(
                    integral_to_real
                )

            if plot:
                fig, ax = plt.subplots()
                ax.set_title(key)
                ax.plot(pdf_bins, pdf, label="Original")
                ax.plot(pdf_bins, kde_pdf, label="KDE")
                ax.axvline(value_mean, label="Mean")
                for p, v in zip(percentiles, value_pct):
                    ax.axvline(v, label=f"P{p*100}", color=pct_cmap(p))

                if real_values is not None and key in real_values:
                    plt.axvline(real_values[key], c="k", label="Real")
                ax.legend()
                ax.set_xlabel(key_name)
                ax.set_ylabel(f"PDF [1/{key_name} units]")

                fig.savefig(
                    os.path.join(
                        os.path.dirname(output_filename),
                        f"stat_analysis_pdf_{key_1}.png",
                    ),
                    dpi=200,
                    bbox_inches="tight",
                )
                if show:
                    plt.show()
                else:
                    plt.close()

        output_hdul.extend(
            [
                fits.BinTableHDU(table_1d_pdf, name="PDF-1D"),
                fits.BinTableHDU(
                    table_1d_percentiles, name="PERCENTILES", header=table_1d_pct_hdr
                ),
            ]
        )
    if pdf_2d:
        if parameter_key_pairs is None:
            raise ValueError("2D PDF analysis requires input parameter_key_pairs")

        for key_1, key_2 in parameter_key_pairs:
            print("2D post. PDF of", key_1, "vs", key_2)

            value_1 = table[key_1].value
            value_2 = table[key_2].value
            mask = np.isfinite(value_1) & np.isfinite(value_2)

            binedges_1 = np.linspace(
                value_1[mask].min(), value_1[mask].max(), pdf_size + 1
            )
            bins_1 = (binedges_1[:-1] + binedges_1[1:]) / 2
            binedges_2 = np.linspace(value_1[mask].min(), value_1[mask].max(), pdf_size)
            bins_2 = (binedges_2[:-1] + binedges_2[1:]) / 2

            try:
                v1_grid, v2_grid = np.meshgrid(bins_1, bins_2, indexing="ij")
                kde = stats.gaussian_kde(
                    np.array([value_1[mask], value_2[mask]]), weights=posterior[mask]
                )
                pdf = kde(np.array([v1_grid.flatten(), v2_grid.flatten()]))
                pdf = pdf.reshape(v1_grid.shape)
            except Exception as e:
                print(
                    "There was an error during KDE estimation: ",
                    e,
                    "\nComputing PDF from histogram",
                )
                pdf, _, _ = np.histogram2d(
                    value_1[mask],
                    value_2[mask],
                    weights=posterior[mask],
                    density=True,
                    bins=[binedges_1, binedges_2],
                )

            hdr = fits.Header()
            hdr["AXIS0"] = key_1
            hdr["AXIS1"] = key_2

            hdr["A0_INI"] = bins_1[0]
            hdr["A0_END"] = bins_1[-1]
            hdr["A0_DELTA"] = bins_1[1] - bins_1[0]

            hdr["A1_INI"] = bins_2[0]
            hdr["A1_END"] = bins_2[-1]
            hdr["A1_DELTA"] = bins_2[1] - bins_2[0]

            k1 = key_1.replace(parameter_prefix + "--", "")
            k2 = key_2.replace(parameter_prefix + "--", "")
            output_hdul.append(fits.ImageHDU(data=pdf, header=hdr, name=f"{k1}_{k2}"))

            if plot:
                fraction = compute_fraction_from_map(pdf)
                fig, ax = plt.subplots()
                ax.pcolormesh(bins_2, bins_1, pdf, cmap="Greys")
                ax.contour(bins_1, bins_2, fraction, levels=[0.1, 0.5, 0.84])
                ax.set_xlabel(k2)
                ax.set_ylabel(k1)

                if real_values is not None and key_1 in real_values:
                    ax.axhline(real_values[key_1], c="r")
                if real_values is not None and key_2 in real_values:
                    ax.axvline(real_values[key_2], c="r")

                fig.savefig(
                    os.path.join(
                        os.path.dirname(output_filename),
                        f"stat_analysis_pdf_{key_1}_{key_2}.png",
                    ),
                    dpi=200,
                    bbox_inches="tight",
                )
                if show:
                    plt.show()
                else:
                    plt.close()

    primary = fits.PrimaryHDU()
    for k, v in extra_info.items():
        primary.header[k] = v, "user-provided information"
    output_hdul = fits.HDUList([primary, *output_hdul])
    if output_filename is not None:
        output_hdul.writeto(output_filename, overwrite=True)
    return output_hdul


def make_plot_chains(chain_results, truth_values=None, output="."):
    parameters = [par for par in chain_results.keys() if "parameters" in par]
    if truth_values is None:
        truth_values = [np.nan] * len(parameters)
    all_figs = []
    for par, truth in zip(parameters, truth_values):
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(chain_results[par], ",", c="k")
        ax.axhline(truth, c="r")
        inax = ax.inset_axes((1.05, 0, 0.5, 1))
        inax.hist(chain_results[par], weights=chain_results["weight"], bins=100)
        inax.axvline(truth, c="r")
        plt.show()
        all_figs.append(fig)
    return all_figs


def compute_chain_percentiles(chain_results, pct=[0.5, 0.16, 0.50, 0.84, 0.95]):
    parameters = [par for par in chain_results.keys() if "parameters" in par]
    pct_resutls = {}
    for par in parameters:
        sort_pos = np.argsort(chain_results[par])
        cum_distrib = np.cumsum(chain_results["weight"][sort_pos])
        cum_distrib /= cum_distrib[-1]
        pct_resutls[par] = np.interp(pct, cum_distrib, chain_results[par][sort_pos])
    return pct_resutls


if __name__ == "__main__":
    table = read_results_file(
        "/home/pcorchoc/Develop/besta/output/photometry/illustris_dust_and_redshift/subhalo_484448/SFH_results.txt"
    )
    compute_pdf_from_results(
        table,
        real_values={
            "parameters--a_v": 0.15,
            "parameters--z_today": 0.02869,
            "parameters--logssfr_over_10.00_yr": np.log10(9.33e-11),
            "parameters--logssfr_over_9.70_yr": np.log10(1.40e-11),
            "parameters--logssfr_over_9.48_yr": np.log10(1.78e-12),
            "parameters--logssfr_over_9.00_yr": np.log10(1.49e-13),
            "parameters--logssfr_over_8.70_yr": np.log10(5.84e-14),
            "parameters--logssfr_over_8.48_yr": np.log10(1e-14),
            "parameters--logssfr_over_8.00_yr": np.log10(1e-14),
        },
        # parameter_keys=['parameters--logssfr_over_9.48_yr',
        #                 'parameters--logssfr_over_9.00_yr',
        #                 'parameters--logssfr_over_8.70_yr',
        #                 'parameters--logssfr_over_8.48_yr'],
        parameter_key_pairs=[
            ["parameters--logssfr_over_8.48_yr", "parameters--logssfr_over_9.00_yr"]
        ],
        pdf_2d=True,
        plot=False,
        pdf_size=30,
        output_filename="/home/pcorchoc/Research/Euclid/test.table.fits",
    )
