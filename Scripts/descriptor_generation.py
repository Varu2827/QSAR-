import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

# Folder containing all the SDF files
sdf_folder = "ligands.sdf"

# Store results
data = []

for filename in os.listdir(sdf_folder):
    if filename.endswith(".sdf"):
        sdf_path = os.path.join(sdf_folder, filename)
        suppl = Chem.SDMolSupplier(sdf_path)
        for mol in suppl:
            if mol is None:
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else filename
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            data.append({
                "Name": name,
                "File": filename,
                "MolWt": mw,
                "LogP": logp,
                "HDonors": hbd,
                "HAcceptors": hba
            })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("ligand_descriptors.csv", index=False)
print("✅ Descriptors saved to ligand_descriptors.csv")
