#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 26/03/2025
# Author: Sadettin Y. Ugurlu

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from PyBioMed.PyMolecule import moe
from PyBioMed.PyInteraction import PyInteraction
from PyBioMed import Pyprotein
from uti import prepare_mathfeature

"""inactive_targets = {
    "9MQH": "TIMALYSIVCVVGLFGNFLVMYVIVRYTKMKTATNIYIFNLALADALATSTLPFQSVNYLMGTWPFGTILCKIVISIDYYNMFTSIFTLCTMSVDRYIAVCHPVKALDFRTPRNAKIINVCNWILSSAIGLPVMFMATTKYRQGSIDCTLTFSTWYWENLLKICVFIFAFIMPVLIITVCYGLMILRLKSVRLLSGSREKDRNLRRITRMVLVVVAVFIVCWTPIHIYVIIKALVTIPETTFQTVSWHFCIALGYTNSCLNPVLYAFLDENFKRCFREFCI"
}"""

inactive_targets = {
    "5C1M": "GSHSLPQTGSPSMVTAITIMALYSIVCVVGLFGNFLVMYVIVRYTKMKTATNIYIFNLALADALATSTLPFQSVNYLMGTWPFGNILCKIVISIDYYNMFTSIFTLCTMSVDRYIAVCHPVKALDFRTPRNAKIVNVCNWILSSAIGLPVMFMATTKYRQGSIDCTLTFSHPTWYWENLLKICVFIFAFIMPVLIITVCYGLMILRLKSVRMLSGSKEKDRNLRRITRMVLVVVAVFIVCWTPIHIYVIIKALITIPETTFQTVSWHFCIALGYTNSCLNPVLYAFLDENFKRCFQVQLVESGGGLVRPGGSLRLSCVDSERTSYPMGWFRRAPGKEREFVASITWSGIDPTYADSVADRFTTSRDVANNTLYLQMNSLKHEDTAVYYCAARAPVDYDYWGQGTQVTVSSAAA"
}

def smiles_to_rdkit_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol


def prepare_ligand_protein_interactions(smiles, protein_seq, pdb_id):
    mol = smiles_to_rdkit_mol(smiles)
    mol_des = moe.GetMOE(mol)
    prot_des = Pyprotein.PyProtein(protein_seq).GetALL()
    interaction1 = PyInteraction.CalculateInteraction1(mol_des, prot_des)
    interaction2 = PyInteraction.CalculateInteraction2(mol_des, prot_des)
    df1 = pd.DataFrame([{"inter_1_" + k: v for k, v in interaction1.items()}])
    df2 = pd.DataFrame([{"inter_2_" + k: v for k, v in interaction2.items()}])
    result = pd.concat([df1, df2], axis=1)
    result.insert(0, 'smiles', smiles)
    result.insert(1, 'class', 1 if pdb_id in active_targets else 0)
    result.insert(2, 'pdb_id', pdb_id)
    return result


def process_all_ligands_in_chunks(input_csv, output_prefix, chunk_size=100):
    df_input = pd.read_csv(input_csv)
    total_chunks = (len(df_input) + chunk_size - 1) // chunk_size
    all_chunk_paths = []

    for i in range(total_chunks):
        chunk = df_input.iloc[i*chunk_size:(i+1)*chunk_size]
        chunk_results = []

        for idx, row in chunk.iterrows():
            smiles = row['smiles']
            label = row['class']
            targets = active_targets if label == 1 else inactive_targets

            for pdb_id, sequence in targets.items():
                try:
                    interaction_df = prepare_ligand_protein_interactions(smiles, sequence, pdb_id)
                    chunk_results.append(interaction_df)
                except Exception as e:
                    print(f"❌ Failed: {smiles}, {pdb_id} → {e}")

        if chunk_results:
            chunk_df = pd.concat(chunk_results, axis=0, ignore_index=True)
            chunk_path = f"{output_prefix}_chunk_{i+1}.csv"
            chunk_df.to_csv(chunk_path, index=False)
            all_chunk_paths.append(chunk_path)
            print(f"✅ Saved chunk {i+1} → {chunk_path}")

    # Merge all
    if all_chunk_paths:
        final_df = pd.concat([pd.read_csv(p) for p in all_chunk_paths], axis=0, ignore_index=True)
        final_output = f"{output_prefix}_all.csv"
        final_df.to_csv(final_output, index=False)
        print(f"✅ Merged all chunks into {final_output}")


# 🚀 Example usage
if __name__ == "__main__":
    input_file = "all_data.csv"
    output_prefix = "ligand_interactions"
    process_all_ligands_in_chunks(input_file, output_prefix, chunk_size=100)

