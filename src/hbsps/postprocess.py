import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from astropy.io import fits
from scipy import stats

pct_cmap = plt.get_cmap("rainbow").copy()

def read_results_file(path):
    with open(path, "r") as f:
        header = f.readline().strip("#")
        columns = header.replace("\n", "").split("\t")
    matrix = np.loadtxt(path)
    
    table = Table()
    for ith, c in enumerate(columns):
        table.add_column(matrix.T[ith], name=c.lower())
    return table


def compute_fraction_from_map(H, xedges=None, yedges=None, **kwargs):
    """..."""
    if xedges is None:
        H_mass = H
    else:
        H_mass = (H * np.diff(xedges)[:, np.newaxis]
                   * np.diff(yedges)[np.newaxis, :])
    sorted_flat = np.argsort(H, axis=None)
    sorted_2D = np.unravel_index(sorted_flat, H.shape)
    density_sorted = H.flatten()[sorted_flat]
    cumulative_mass = np.cumsum(H_mass[sorted_2D])
    fraction_sorted = cumulative_mass/cumulative_mass[-1]
    fraction = np.interp(H, density_sorted, fraction_sorted)
    return fraction

def compute_pdf_from_results(table,
                             output_filename=None,
                             parameter_prefix='parameters',
                             posterior_key='post', weights_key=None, #TODO
                             parameter_keys=None,                            
                             pdf_1d=True,
                             percentiles=[0.05, 0.16, 0.5, 0.84, 0.95],
                             pdf_2d=True,
                             pdf_size=100, 
                             plot=True,
                             real_values=None):

    posterior = np.exp(
        table[posterior_key].value - np.nanmax(table[posterior_key].value)) 
    posterior /= np.nansum(posterior)

    # Select the keys that correspond to sampled parameters
    if parameter_keys is None:
        parameter_keys = [key for key in list(table.keys())
                          if parameter_prefix in key]
        
    output_hdul = []
    if pdf_1d:
        table_1d_pdf = Table()
        table_1d_percentiles = Table()
        table_1d_percentiles.add_column(percentiles, name='percentiles')
        table_1d_pct_hdr = fits.Header()

        for key in parameter_keys:
            value = table[key].value
            mask = np.isfinite(value)
            sort_pos = np.argsort(value[mask])
            cmf = np.cumsum(posterior[mask][sort_pos])

            value_pct = np.interp(percentiles, cmf, value[mask][sort_pos])

            dummy_value = np.linspace(value[mask].min(), value[mask].max(),
                                      pdf_size + 1)
            interp_cmf = np.interp(dummy_value, value[sort_pos], cmf)
            pdf = (interp_cmf[1:] - interp_cmf[:-1]) / (dummy_value[1:] - dummy_value[:-1])
            dummy_bins = (dummy_value[1:] + dummy_value[:-1]) / 2

            key_name = key.replace(parameter_prefix + "--", "")
            kde = stats.gaussian_kde(value[mask], weights=posterior[mask])

            table_1d_pdf.add_column(dummy_bins,
                                    name=f"{key_name}_bin")
            table_1d_pdf.add_column(pdf, name=f"{key_name}_pdf")
            table_1d_pdf.add_column(kde(dummy_bins), name=f"{key_name}_pdf_kde")

            table_1d_percentiles.add_column(value_pct, name=f"{key_name}_pct")

            if real_values is not None and key in real_values:
                integral_to_real = np.interp(
                    real_values[key], value[sort_pos], cmf)
                table_1d_pct_hdr[f"hierarch {key_name}_real"] = np.nan_to_num(real_values[key])
                table_1d_pct_hdr[f"hierarch {key_name}_int_to_real"] = np.nan_to_num(integral_to_real)

            if plot:
                fig, ax = plt.subplots()
                ax.set_title(key)
                ax.plot(dummy_bins, pdf, label='Original')
                ax.plot(dummy_value, kde(dummy_value), label='KDE')
                for p, v in zip(percentiles, value_pct):
                    ax.axvline(v, label=f"P{p*100}", color=pct_cmap(p))

                if real_values is not None and key in real_values:
                    plt.axvline(real_values[key], c='k', label='Real')
                ax.legend()
                ax.set_xlabel(key_name)
                ax.set_ylabel(f"PDF [1/{key_name} units]")

                plt.show() # TODO: save plots
                #plt.close()

        output_hdul.extend(
            [fits.BinTableHDU(table_1d_pdf, name='PDF-1D'),
             fits.BinTableHDU(table_1d_percentiles, name='PERCENTILES',
                              header=table_1d_pct_hdr)])
    if pdf_2d:
        for ith, key_1 in enumerate(parameter_keys):
            for key_2 in parameter_keys[ith + 1:]:
                print("Computing 2D posterior distribution of\n",
                      key_1, "versus", key_2)
                
                value_1 = table[key_1].value
                mask_1 = np.isfinite(value_1)

                value_2 = table[key_2].value
                mask_2 = np.isfinite(value_2)
                mask = mask_1 & mask_2

                # dummy_value_1 = np.linspace(
                #     value_1[mask].min(), value_1[mask].max(), pdf_size + 1)
                # dummy_value_2 = np.linspace(
                #     value_2[mask].min(), value_2[mask].max(), pdf_size + 1)
                
                # dd1, dd2 = np.meshgrid(dummy_value_1, dummy_value_2)

                # kde = stats.gaussian_kde(np.array([value_1[mask], value_2[mask]]),
                #                    weights=posterior[mask])
                # pdf = kde(np.vstack([dd1.ravel(), dd2.ravel()]))
                # pdf = pdf.reshape(dd1.shape)

                pdf, xedges, yedges = np.histogram2d(
                    value_1[mask], value_2[mask], weights=posterior[mask],
                    density=True, bins=pdf_size)

                dummy_value_1 = (xedges[:-1] + xedges[1:]) / 2
                dummy_value_2 = (yedges[:-1] + yedges[1:]) / 2

                hdr = fits.Header()
                hdr['AXIS0'] = key_1
                hdr['AXIS1'] = key_2

                hdr['A0_INI'] = dummy_value_1[0]
                hdr['A0_END'] = dummy_value_1[-1]

                hdr['A1_INI'] = dummy_value_2[0]
                hdr['A1_END'] = dummy_value_2[-1]

                k1 = key_1.replace(parameter_prefix + "--", "")
                k2 = key_2.replace(parameter_prefix + "--", "")
                output_hdul.append(
                    fits.ImageHDU(data=pdf, header=hdr, name=f"{k1}_{k2}")
                            )

                key_1_name = key_1.replace(parameter_prefix + "--", "")
                key_2_name = key_2.replace(parameter_prefix + "--", "")

                if plot:
                    fraction = compute_fraction_from_map(pdf)

                    fig, ax = plt.subplots()
                    ax.pcolormesh(dummy_value_1, dummy_value_2, pdf,
                                cmap='Greys')
                    ax.contour(dummy_value_1, dummy_value_2, fraction,
                            levels=[.1, .5, .84])
                    ax.set_xlabel(key_1_name)
                    ax.set_ylabel(key_2_name)

                    if real_values is not None and key_1 in real_values:
                        ax.axvline(real_values[key_1], c='r')
                    if real_values is not None and key_2 in real_values:
                        ax.axhline(real_values[key_2], c='r')
                    
                    if output_filename is None:
                        fig.savefig(f"stat_analysis_pdf_{key_1}_{key_2}.png",
                                    pdi=200, bbox_inches='tight')
                    else:
                        fig.savefig(os.path.join(
                            os.path.dirname(output_filename),
                            f"stat_analysis_pdf_{key_1}_{key_2}.png"),
                                    pdi=200, bbox_inches='tight')
                    plt.close()

    output_hdul = fits.HDUList([fits.PrimaryHDU(), *output_hdul])
    if output_filename is None:
        output_hdul.writeto("stat_analysis.fits", overwrite=True)
    else:
        output_hdul.writeto(output_filename, overwrite=True)


                







if __name__ == "__main__":
    table = read_results_file(
        "/home/pcorchoc/Develop/HBSPS/output/photometry/illustris/subhalo_167396/SFH_results.txt")
    compute_pdf_from_results(
        table, real_values={'parameters--a_v': 0.50,
                            'parameters--logssfr_over_10.00_yr': np.log10(9.31e-11),
                            'parameters--logssfr_over_9.70_yr': np.log10(9.05e-11),
                            'parameters--logssfr_over_9.48_yr': np.log10(8.90e-11),
                            'parameters--logssfr_over_9.00_yr': np.log10(8.46e-11),
                            'parameters--logssfr_over_8.70_yr': np.log10(1.06e-10),
                            'parameters--logssfr_over_8.48_yr': np.log10(9.60e-11),
                            'parameters--logssfr_over_8.00_yr': np.log10(9.77e-11),
                            },
                            # parameter_keys=['parameters--logssfr_over_9.48_yr',
                            #                 'parameters--logssfr_over_9.00_yr',
                            #                 'parameters--logssfr_over_8.70_yr',
                            #                 'parameters--logssfr_over_8.48_yr'],
                            pdf_2d=False,
                            plot=True
                            )