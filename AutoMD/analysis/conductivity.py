import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import msd
from scipy.stats import linregress



class ComputeHelfand:
    """
    Class for calculating ionic conductivity via Einstein-Helfand relation (equivalent to GK).
    """


    def get_gk_conductivity(self, step = 1, include_reg = False, fit_params=None):
        """
        Compute ionic conductivity via Einstein-Helfand relation (equivalent to GK).

        Parameters:
            step: Integer, step size for frames to process.
            include_reg: Boolean, whether to include linear regression in the output.
            fit_params: Tuple, (start, end) indices for linear regression IN NANOSECONDS.

        Returns:
            time_ns, msd, conductivity_S_per_m
        """

        KB = 1.380649e-23
        E_CHARGE = 1.60217663e-19
        ANGSTROM_TO_M = 1e-10

        u = self.universe
        ag = u.atoms

        if ag.charges is None:
            raise ValueError("Missing charges")

        time_per_frame_ns = self.dt * self.dump_interval * step * 1e9
        traj = u.trajectory[::step]
        n_frames = len(traj)

        M = np.zeros((n_frames, 3), dtype=np.float64)

        for i, ts in enumerate(traj):
            M[i] = np.sum(ag.charges[:, None] * ag.positions, axis=0)


        #M *= (E_CHARGE * ANGSTROM_TO_M)

        dummy = mda.Universe.empty(n_atoms=1, trajectory=True)
        dummy.load_new(M[:, None, :], format="memory")

        msd_calc = msd.EinsteinMSD(dummy.atoms, fft=True)
        msd_calc.run()

        msd_vals = msd_calc.results.timeseries
        time = np.arange(len(msd_vals)) * time_per_frame_ns
        time_s = time * 1e-9

        lx, ly, lz = u.dimensions[:3]
        vol_m3 = lx * ly * lz * (ANGSTROM_TO_M**3)
        if  fit_params is None:
            start = len(time_s) // 3
            end = len(time_s) * 3 // 4
        else:
            start, end = fit_params
            start = int(start / time_per_frame_ns)
            end = int(end / time_per_frame_ns)

        slope, intercept, r_value, p_value, std_err = linregress(time_s[start:end], msd_vals[start:end])

        sigma = slope / (6 * KB * self.temperature * vol_m3) * (E_CHARGE * ANGSTROM_TO_M)**2
        if include_reg:
            return time, msd_vals, sigma, (slope, intercept, r_value, p_value, std_err)
        else:
            return time, msd_vals, sigma
    