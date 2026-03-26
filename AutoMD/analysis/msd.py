import MDAnalysis.analysis.msd as msd

class ComputeMSD:
    """
    Class for calculating Mean Square Displacements (MSDs).
    """
    def get_msd(self, sel, fft=True, step=1, com=False):
        """
        Calculates the Mean Squared Displacement (MSD) for a selection of atoms or molecules.
        Parameters:
        - sel: Selection string, tuple, or molecule name.
        - fft: Boolean, whether to use FFT to speed up MSD calculations.
        - step: Integer, step size for frames to process.
        - com: Boolean. If True, computes the MSD of the Center of Mass of the selected molecules/residues.
               If False, computes the standard combined/averaged atom MSD.
        """
        mda_sel = self._build_selection_string(sel)
        ag = self.universe.select_atoms(mda_sel)
        if len(ag) == 0:
            raise ValueError(f"Empty selection, check selection string: {sel}")
        
        # compute MSD for center of mass of molecules if com is True
        if com:
            residues = ag.residues
            # only load every step:th frame in case it is not 1
            traj_slice = self.universe.trajectory[::step]
            n_frames = len(traj_slice)
            print(f"Calculating COM MSD for {mda_sel} ({len(residues)} molecules) over {n_frames} frames...")
            com_traj = np.zeros((n_frames, len(residues), 3), dtype=np.float32)
            current_frame = self.universe.trajectory.frame
            # calculate center of mass for each residue
            for i, ts in enumerate(traj_slice):
                com_traj[i] = residues.center_of_mass(compound='residues')
            self.universe.trajectory[current_frame]
            dummy_u = mda.Universe.empty(len(residues), trajectory=True)
            dummy_u.load_new(com_traj, format="memory")
            msd_analyzer = msd.EinsteinMSD(dummy_u.atoms, fft=fft)
            # already sliced the trajectory so step is 1
            msd_analyzer.run(step=1)
        else:
            
            # standard atom-averaged MSD
            print(f"Calculating Atom-averaged MSD for {mda_sel} ({len(ag)} atoms)...")
            msd_analyzer = msd.EinsteinMSD(ag, fft=fft)
            msd_analyzer.run(step=step)
        # time array in ns
        time_per_frame_ns = self.dt * self.dump_interval * step * 1e9 
        frames = len(msd_analyzer.results.timeseries)
        time_array = np.arange(frames) * time_per_frame_ns
        # Returns t in ns and y MSD in Ångström^2
        return time_array, msd_analyzer.results.timeseries