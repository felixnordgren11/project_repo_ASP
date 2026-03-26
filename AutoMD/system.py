import os
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
from MDAnalysis.analysis import msd, rdf
from .molecules import load_molecule_definitions
from scipy.integrate import cumulative_trapezoid
from scipy.stats import linregress
from .analysis.msd import ComputeMSD
from .analysis.rdf import ComputeRDF
from .analysis.conductivity import ComputeHelfand

class System(ComputeMSD, ComputeRDF, ComputeHelfand):
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