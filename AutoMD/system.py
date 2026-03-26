import os
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
from MDAnalysis.analysis import msd, rdf
from .molecules import load_molecule_definitions
from scipy.integrate import cumulative_trapezoid
from scipy.stats import linregress

class System:
    '''
    A class to store information about a LAMMPS simulation system
    '''
    def __init__(self, sysdir, system_name=None, cations=None, anions=None, n_molecules=None, 
                 ff_file_name="data.lmp", 
                 data_file_name="system_after_equilibration.data", 
                 dump_file_name="dump_prod.lammpstrj",
                 green_kubo_file_name=None,
                 input_file_name="in_append_tmp.lmp",
                 temperature=None):
        
        self.sysdir = sysdir
        self.system_name = system_name
        self.cations = [cations] if isinstance(cations, str) else (cations if cations is not None else [])
        self.anions = [anions] if isinstance(anions, str) else (anions if anions is not None else [])
        self.n_molecules = n_molecules
        
        # file paths
        self.ff_file = os.path.join(sysdir, ff_file_name)
        self.data_file = os.path.join(sysdir, data_file_name)
        self.dump_file = os.path.join(sysdir, dump_file_name)
        self.input_file = os.path.join(sysdir, input_file_name)
        self.green_kubo_file = os.path.join(sysdir, green_kubo_file_name) if green_kubo_file_name is not None else None
        
        # parameters
        self.atom_types = {}
        self.box_size = None
        self.dt = 1.0e-15      # default fallback (1 fs)
        self.dump_interval = 1000 # (usually use this value for production trajectories)
        self.temperature = None
        
        # parse stuff
        self._parse_atom_types()
        self._parse_box_size()
        self._parse_input_script()
        
        # initialize universe
        self.universe = mda.Universe(
            self.data_file,
            self.dump_file,
            topology_format="DATA",
            format="LAMMPSDUMP"
        )
        
        # assign resnames using JSON definitions
        self._assign_molecule_names()

    def _parse_atom_types(self):
        """Extracts atom types from the fftool generated data file."""
        if not os.path.exists(self.ff_file):
            return
            
        with open(self.ff_file, 'r') as f:
            lines = f.readlines()
            
        in_masses_section = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Masses"):
                in_masses_section = True
                continue
                
            if in_masses_section:
                # stop when reaching end of atom types
                if line.startswith("Bond Coeffs") or line.startswith("Pair Coeffs"):
                    break
                
                # looks for lines formatted by fftool "1 6.941 # Li"
                if '#' in line:
                    data_part, comment_part = line.split('#', 1)
                    columns = data_part.split()
                    
                    if len(columns) >= 2:
                        type_number = int(columns[0])
                        atom_name = comment_part.strip()
                        self.atom_types[atom_name] = type_number
                        
                        # detect cation
                        if atom_name in ['Li', 'K', 'Na', 'Rb', 'Cs'] and atom_name not in self.cations:
                            self.cations.append(atom_name)

    def _parse_box_size(self):
        """
        Extracts box dimensions from the equilibrated data file
        """
        if not os.path.exists(self.data_file):
            return
            
        with open(self.data_file, 'r') as f:
            for line in f:
                if "xlo xhi" in line:
                    parts = line.split()
                    xlo, xhi = float(parts[0]), float(parts[1])
                    self.box_size = xhi - xlo 
                    break

    def _parse_input_script(self):
        """
        Extracts temperature, timestep, and dump interval from the LAMMPS input script.
        """
        if not os.path.exists(self.input_file):
            return
            
        target_dump_file = os.path.basename(self.dump_file)
        
        with open(self.input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("variable T_prod equal"):
                    self.temperature = float(line.split()[3]) # get production temperature
                
                # (convert fs to s)
                elif line.startswith("timestep"):
                    try:
                        ts_fs = float(line.split()[1])
                        self.dt = ts_fs * 1.0e-15
                    except ValueError:
                        pass
                
                elif line.startswith("dump ") and target_dump_file in line:
                    parts = line.split()
                    try:
                        file_index = parts.index(target_dump_file)
                        self.dump_interval = int(parts[file_index - 1])
                        print(f"Dump interval: {self.dump_interval}")
                    except (ValueError, IndexError):
                        pass

    def to_dict(self):
        """
        Helper to convert the system object into a dictionary for Pandas.

        Returns:
            dict: Dictionary containing system information.
        """
        return {
            "system_name": self.system_name,
            "sysdir": self.sysdir,
            "cations": self.cations,
            "anions": self.anions,
            "n_molecules": self.n_molecules,
            "atom_types": self.atom_types,
            "dt": self.dt,
            "dump_interval": self.dump_interval,
            "temperature": self.temperature,
            "box_size": self.box_size,
            "universe": self.universe
        }

    def _assign_molecule_names(self):
        """
        Assigns standard names string to MDAnalysis residues based on atom count
        and presence of specific atom types defined in AutoMD/molecules/*.json.

        """

        molecules_dir = os.path.join(os.path.dirname(__file__), 'molecules')
        molecule_defs = load_molecule_definitions(molecules_dir)
        
        if not molecule_defs:
            return
            
        inv_atom_types = {str(v): k for k, v in self.atom_types.items()}
        assigned_counts = {}
        
        # add atom names so selections like "name O*" work
        names_array = np.array([inv_atom_types.get(nt, nt) for nt in self.universe.atoms.types], dtype=object)
        try:
            self.universe.add_TopologyAttr('name', names_array)
        except Exception:
            self.universe.atoms.names = names_array
            
        resnames_array = np.array(["UNK"] * len(self.universe.residues), dtype=object)
        self.molecule_residues = {}
        
        for i, res in enumerate(self.universe.residues): # loop through all residues to assign them names
            n_atoms = res.atoms.n_atoms
            numeric_types = np.unique(res.atoms.types)
            string_types = set([inv_atom_types.get(nt, nt) for nt in numeric_types])
            
            assigned_name = None
            
            for mol_name, sig in molecule_defs.items():
                if n_atoms == sig.get('n_atoms'): # residue size
                    # important! Here we differentiate between different atom types 
                    # in case there are multiple molecules with the same size
                    target_types = set([str(t) for t in sig.get('atom_types', [])]) 
                    if string_types.issubset(target_types):
                        assigned_name = mol_name
                        break
                        
            if assigned_name:
                resnames_array[i] = assigned_name
                assigned_counts[assigned_name] = assigned_counts.get(assigned_name, 0) + 1
                if assigned_name not in self.molecule_residues:
                    self.molecule_residues[assigned_name] = []
                self.molecule_residues[assigned_name].append(res)
            else:
                if n_atoms == 1:
                    single_type = list(string_types)[0]
                    if single_type.lower() in ['li', 'k', 'na']:
                        resnames_array[i] = single_type
                        assigned_counts[resnames_array[i]] = assigned_counts.get(resnames_array[i], 0) + 1
                        if resnames_array[i] not in self.molecule_residues:
                            self.molecule_residues[resnames_array[i]] = []
                        self.molecule_residues[resnames_array[i]].append(res)
                        
        try:
            self.universe.add_TopologyAttr('resname', resnames_array)
        except Exception:
            for i, res in enumerate(self.universe.residues):
                res.resname = resnames_array[i]
                
        print(f"Assigned residues: {assigned_counts}")

    def _build_selection_string(self, sel):
        """
        Helper method to build selection strings dynamically.
        Supports:
        - "TFSI" -> "resname TFSI" (Whole molecule)
        - "OBT" -> "name OBT" (All OBT atoms everywhere)
        - ("TFSI", "OBT") -> "resname TFSI and name OBT" (Only OBT inside TFSI)
        - "type 1 or type 2" -> Custom MDA strings returned as-is
        """
        # Handle tuples to select specific atom types from a specific molecule
        # e.g. if putting in ("TFSI", "OBT") -> gets converted to "resname TFSI and name OBT" (Only OBT inside TFSI)
        if isinstance(sel, tuple) or isinstance(sel, list):
            if len(sel) == 2:
                mol_name, atom_name = sel
                return f"resname {mol_name} and name {atom_name}"
            else:
                raise ValueError("Tuple selections must be formatted as (molecule_name, atom_name)")

        # allow string inputs
        if isinstance(sel, str):
            # allow for MDAnalysis selection strings
            if any(keyword in sel.lower() for keyword in [' or ', ' and ', 'type ', 'name ', 'resname ', 'all', 'not ']):
                return sel
            # allow for whole molecule selections
            if hasattr(self, 'molecule_residues') and sel in self.molecule_residues:
                return f"resname {sel}"
            
            # if it matches an atom type string (e.g., "OBT", "F1") which are loaded in from the json files
            if sel in self.atom_types:
                return f"name {sel}"
                
            # o/w just use the types
            if sel in self.atom_types:
                return f"type {self.atom_types[sel]}"

        raise ValueError(f"Unrecognized selection format: {sel}")

    def get_rdf(self, sel1, sel2, nbins=200, r_range=(0.0, 15.0), step=1):
        """
        Calculates the Radial Distribution Function (RDF) between two selections.
        """
        # convert input (e.g., 'Li') to MDA selection strings (e.g., 'type 1')
        mda_sel1 = self._build_selection_string(sel1)
        mda_sel2 = self._build_selection_string(sel2)
        
        ag1 = self.universe.select_atoms(mda_sel1)
        ag2 = self.universe.select_atoms(mda_sel2)
        
        if len(ag1) == 0 or len(ag2) == 0:
            raise ValueError(f"Empty selection! Check your selection strings: {sel1}, {sel2}")

        print(f"Calculating RDF between {mda_sel1} ({len(ag1)} atoms) and {mda_sel2} ({len(ag2)} atoms)...")
        
        rdf_analyzer = rdf.InterRDF(ag1, ag2, nbins=nbins, range=r_range)
        rdf_analyzer.run(step=step)
        
        # Returns r (distance in Ångström) and g(r) (radial probability density)
        return rdf_analyzer.results.bins, rdf_analyzer.results.rdf

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



    def get_gk_conductivity(self, step = 1):
        """
        Compute ionic conductivity via Einstein-Helfand relation (equivalent to GK).

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


        M *= (E_CHARGE * ANGSTROM_TO_M)

        dummy = mda.Universe.empty(n_atoms=1, trajectory=True)
        dummy.load_new(M[:, None, :], format="memory")

        msd_calc = msd.EinsteinMSD(dummy.atoms, fft=True)
        msd_calc.run()

        msd_vals = msd_calc.results.timeseries
        time = np.arange(len(msd_vals)) * time_per_frame_ns
        time_s = time * 1e-9

        lx, ly, lz = u.dimensions[:3]
        vol_m3 = lx * ly * lz * (ANGSTROM_TO_M**3)
        start = len(time_s) // 3
        end = len(time_s) * 3 // 4

        slope, _, _, _, _ = linregress(time_s[start:end], msd_vals[start:end])

        sigma = slope / (6 * KB * self.temperature * vol_m3)

        return time, msd_vals, sigma