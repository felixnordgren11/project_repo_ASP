import os
import json

def load_molecule_definitions(molecules_dir):
    """
    Loads all .json files in the molecules_dir.
    Returns a dictionary of molecule names mapped to their signatures.
    """
    molecules = {}
    if not os.path.isdir(molecules_dir):
        return molecules
        
    for filename in os.listdir(molecules_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(molecules_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Use filename without extension as the molecule name (uppercase)
                    mol_name = data["name"]
                    molecules[mol_name] = data
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return molecules
