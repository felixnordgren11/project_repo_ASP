from MDAnalysis.analysis import rdf




class ComputeRDF:
    """
    Class for calculating Radial Distribution Functions (RDFs).
    """

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