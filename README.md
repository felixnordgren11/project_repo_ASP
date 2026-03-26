# AutoMD

**AutoMD** is a Python package for automating the analysis of molecular dynamics (MD) simulations. Helps simplify parsing trajectory files and molecules present in the system, giving me a more streamlined workflow.

## Features

AutoMD parses atom types from data files created using fftool and assigns them to molecules present in the system based on a JSON file. Specifically designed for molten salt electrolyte systems (only because of the currently available json files). Features include:

- **Trajectory Processing**: Reads and parses LAMMPS trajectory files (`.lammpstrj`) with automatic timestep, dump frequency and temperature extraction.
- **Molecule Management**: Uses a JSON-based molecule bank to easily identify and track different chemical species in your simulation.
- **Standard Analyses**:
  - **MSD Plots**: Mean Squared Displacement for diffusion coefficient calculation.
  - **Diffusion Constants**: Calculated from MSD with linear regression.
  - **Ionic Conductivities**: Using the Nernst-Einstein relation from diffusion constants (want to extend to Green-Kubo ionic conductivity as well).
  - **Radial Distribution Functions (RDFs)**: Pairwise analysis of atomic correlations.
  - **Coordination/Solvation Numbers**: Analyzing local environments around specific ions.
  - **Kirkwood-Buff Integrals**: Analyzing thermodynamic properties and preferential solvation based on RDFs.
- **Automation**: Makes it easier to calculate RDFs and MSDs for specific atom types present in only specific molecules in case the molecules happen to share atom types. For example, if you have TFSI- and FSI- in your system, you can easily calculate the RDFs and MSDs for example of the oxygens in TFSI- and FSI- separately. Residues are also assigned based on the JSON file, not only the size of the molecule, making it more robust in case we have molecules of the same size.

## Installation

clone the repo then if inside the repo, run:

pip install -e .

Otherwise if outside in parent directory, run:

pip install AutomMD

