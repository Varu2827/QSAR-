Computational Identification of Pseudomonas oleovorans AlkB Enzyme Activators
📌 Project Overview

This project focuses on the computational identification of AlkB enzyme activators in Pseudomonas oleovorans to enhance plastic biodegradation. Using QSAR modeling and molecular docking, natural/marine ligands were screened for their ability to improve AlkB activity, offering potential eco-friendly solutions to plastic waste degradation.

🧬 Workflow

Sequence & Structure Retrieval

UniProt ID: P12691

AlphaFold-predicted structure downloaded from UniProt.

Ligand Library Preparation

Marine/natural product SDF library curated.

Ligands converted to PDBQT format using OpenBabel/AutoDock tools.

Molecular Docking

Docking performed with AutoDock Vina.

Top-scoring ligands identified based on binding energy.

QSAR Model Building

Molecular descriptors calculated (PaDEL/Dragon).

Regression/ML models applied for activity prediction.

Model validated with R², Q², and external test sets.

Visualization & Analysis

Docking poses analyzed in PyMOL/Discovery Studio.

2D interaction maps generated.

📂 Project Structure
project/
│── data/
│   ├── ligands.sdf
│   ├── alkB.pdb (AlphaFold structure)
│── Scripts
│── figures/
│   ├── workflow.png
│   ├── docking_interactions.png
│── README.md

⚙️ Tools & Software

AutoDock Vina – Molecular docking

PyMOL / Discovery Studio – Visualization

PaDEL / Dragon – Descriptor calculation

Scikit-learn (Python) – QSAR modeling

UniProt / AlphaFold – Sequence & structure

🚀 Results & Outcome

Top ligand candidates predicted as AlkB activators.

QSAR model validated with acceptable predictive power.

Provides potential green biotechnology strategy for plastic degradation.

⚙️ Installation & Requirements
# Install dependencies
conda create -n alkb_project python=3.10 -y
conda activate alkb_project
pip install scikit-learn pandas numpy matplotlib seaborn rdkit-pypi
sudo apt install autodock-vina openbabel

🔹 Step 1: Protein Retrieval

Download the AlkB enzyme structure (AlphaFold model):
👉 Direct UniProt Link

Save as alkB.pdb.

🔹 Step 2: Ligand Preparation

Convert ligand library from SDF → PDBQT using OpenBabel:

obabel ligands.sdf -O ligands.pdbqt --gen3d

🔹 Step 3: Docking with AutoDock Vina

Prepare receptor and ligand:

# Receptor preparation
prepare_receptor4.py -r alkB.pdb -o alkB.pdbqt

# Example: Dock one ligand
vina --receptor alkB.pdbqt --ligand ligand1.pdbqt --center_x 10 --center_y 15 --center_z 20 --size_x 20 --size_y 20 --size_z 20 --out out_ligand1.pdbqt --log log1.txt


Batch docking (all ligands):

for i in *.pdbqt; do
    vina --receptor alkB.pdbqt --ligand $i --center_x 10 --center_y 15 --center_z 20 \
    --size_x 20 --size_y 20 --size_z 20 --out docked_$i --log log_$i.txt
done

🔹 Step 4: Descriptor Calculation (QSAR)

Use PaDEL for descriptor extraction:

java -jar PaDEL-Descriptor.jar -dir ligands/ -file descriptors.csv

🔹 Step 5: QSAR Modeling in Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load descriptors
data = pd.read_csv("descriptors.csv")

X = data.drop(["Activity"], axis=1)   # descriptors
y = data["Activity"]                  # experimental activity values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R²:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

🔹 Step 6: Visualization

Docking interaction visualization (PyMOL):

pymol alkB.pdb docked_ligand1.pdbqt


QSAR scatter plot:

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Experimental Activity")
plt.ylabel("Predicted Activity")
plt.title("QSAR Model Performance")
plt.show()

🚀 Results

Top-scoring ligands identified as potential AlkB activators.

QSAR model achieved strong predictive accuracy (R² > 0.7).

Provides an eco-friendly biotechnology approach for plastic degradation.

👩‍💻 Author

Varsha L – MSc Bioinformatics, Garden City University
contact - varshabiotech2021.bioinfo2024@gmail.com
