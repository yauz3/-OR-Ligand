import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import rdFingerprintGenerator

def smiles_to_all_fingerprints(smiles_list, nBits_avalon=512, nBits_morgan=512):
    """Compute Avalon, Morgan and MACCS fingerprints and return as combined DataFrame."""
    avalon_fps = []
    morgan_fps = []
    maccs_fps = []

    # Initialize Morgan generator once
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nBits_morgan)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            avalon_fps.append(np.zeros(nBits_avalon))
            morgan_fps.append(np.zeros(nBits_morgan))
            maccs_fps.append(np.zeros(167))
            continue

        # Avalon
        avalon_fp = GetAvalonFP(mol, nBits=nBits_avalon)
        avalon_arr = np.zeros((nBits_avalon,), dtype=int)
        ConvertToNumpyArray(avalon_fp, avalon_arr)
        avalon_fps.append(avalon_arr)

        # Morgan (using MorganGenerator to avoid deprecation)
        morgan_fp = morgan_gen.GetFingerprint(mol)
        morgan_arr = np.zeros((nBits_morgan,), dtype=int)
        ConvertToNumpyArray(morgan_fp, morgan_arr)
        morgan_fps.append(morgan_arr)

        # MACCS
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=int)
        ConvertToNumpyArray(maccs_fp, maccs_arr)
        maccs_fps.append(maccs_arr)

    # Create DataFrames
    df_avalon = pd.DataFrame(avalon_fps, columns=[f"Avalon_FP_{i}" for i in range(nBits_avalon)])
    df_morgan = pd.DataFrame(morgan_fps, columns=[f"Morgan_FP_{i}" for i in range(nBits_morgan)])
    df_maccs = pd.DataFrame(maccs_fps, columns=[f"MACCS_FP_{i}" for i in range(167)])

    # (Optional) Print column names
    print("Avalon Columns:", list(df_avalon.columns))
    print("Morgan Columns:", list(df_morgan.columns))
    print("MACCS Columns:", list(df_maccs.columns))

    # Combine all
    return pd.concat([df_avalon, df_morgan, df_maccs], axis=1)


def example_usage():
    # 📌 Load original datasets
    train_df = pd.read_csv("all_data.csv")

    # 📌 Extract SMILES
    train_smiles = train_df["smiles"]

    # 📌 Generate fingerprints
    train_fps = smiles_to_all_fingerprints(train_smiles)

    # 📌 Merge with original data
    train_final = pd.concat([train_df.reset_index(drop=True), train_fps.reset_index(drop=True)], axis=1)

    # 📌 Save to CSV
    train_final.to_csv("train_avalon_morgan_maccs.csv", index=False)

    print("✅ Avalon + Morgan + MACCS fingerprint have been done!!!!")

example_usage()