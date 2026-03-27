from MDAnalysis.analysis import rdf
from scipy.integrate import simpson, cumulative_trapezoid
import numpy as np
from scipy.signal import find_peaks




class ComputeRDF:
    """
    Class for calculating Radial Distribution Functions (RDFs).
    """

    def get_rdf(self, sel1, sel2, nbins=200, r_range=(0.0, 15.0), step=1, kirkwood_buff=False, coordination_number=False):
        """
        Calculates the Radial Distribution Function (RDF) between two selections.

        Parameters
        ----------
        sel1 : str
            Selection string for the first atom group.
        sel2 : str
            Selection string for the second atom group.
        nbins : int, optional
            Number of bins for the RDF calculation. Default is 200.
        r_range : tuple, optional
            Range of distances to calculate the RDF for. Default is (0.0, 15.0).
        step : int, optional
            Step size for the RDF calculation. Default is 1.
        kirkwood_buff : bool, optional
            Whether to calculate the Kirkwood-Buff integral. Default is False.

        Returns
        -------
        dict
            Dictionary always containing the RDF bins, RDF values, 
            and optional Kirkwood-Buff integral and coordination number which
            are called by 'kirkwood_buff' and 'coordination_number' flags (and 'first_shell_radius' if wanted).
        """
        mda_sel1 = self._build_selection_string(sel1)
        mda_sel2 = self._build_selection_string(sel2)
        
        ag1 = self.universe.select_atoms(mda_sel1)
        ag2 = self.universe.select_atoms(mda_sel2)
        
        print(f"Calculating RDF between {mda_sel1} ({len(ag1)} atoms) and {mda_sel2} ({len(ag2)} atoms)...")
        
        rdf_analyzer = rdf.InterRDF(ag1, ag2, nbins=nbins, range=r_range)
        rdf_analyzer.run(step=step)
        bins = rdf_analyzer.results.bins
        g_r = rdf_analyzer.results.rdf

        results = {
            "bins": bins,
            "g_r": g_r,
        }

        if coordination_number:
            rho = len(ag2) / self.universe.dimensions[0]**3
            cn = 4 * np.pi * rho * cumulative_trapezoid(y=(g_r * (bins**2)), x=bins, initial=0)
            peaks, _ = find_peaks(g_r, height=1.2)
            if len(peaks) > 0:
                first_peak_idx = peaks[0]
                minima, _ = find_peaks(-g_r[first_peak_idx:])
            else:
                minima, _ = find_peaks(-g_r)
            first_min_idx = first_peak_idx + minima[0]
            shell_radius = bins[first_min_idx]
            shell_cn = cn[first_min_idx]
            results["coordination_number"] = shell_cn
            results["first_shell_radius"] = shell_radius

        # KBI has units of volume per molecule, quantifies the excess (or deficiency) of particle j around particle i.
        if kirkwood_buff: # here we're assuming spherical symmetry, and kb is defined as 4 * pi * integral_0^inf (g(r) - 1) r^2 dr
            kb_integral = 4 * np.pi * simpson(y=(g_r - 1) * bins**2, x=bins)
            results["kirkwood_buff"] = kb_integral
        
        return results

    