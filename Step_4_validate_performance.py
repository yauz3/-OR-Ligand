import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    recall_score, precision_score, balanced_accuracy_score
)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import os
import random
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

random.seed(42)
np.random.seed(42)

# Load data
train_df = pd.read_csv("training_gnn_fixed.csv", index_col="smiles")
test_df = pd.read_csv("external_validation_inxight_gnn_fixed.csv", index_col="smiles")
y_train = train_df["class"]
y_test = test_df["class"]
# Ligand features
ligand_feats=['nRot', 'n10FaHRing', 'ATS4dv', 'Kier2', 'AATSC1v', 'NssBH', 'AATS3i', 'GATS1m', 'AATS5s', 'GATS6d', 'AATSC0c', 'AATS6s', 'SRW10', 'NddssSe', 'Xch-3d', 'n11FHRing', 'MATS2p', 'SpAD_DzZ', 'SdCH2', 'GATS5Z', 'MATS4dv', 'LogEE_Dzm', 'SaasN', 'ATSC1d', 'CIC1', 'MATS1s', 'BCUTv-1l', 'MAXssO', 'MAXssCH2', 'AATS0s', 'C2SP3', 'ATS0v', 'ATSC3Z', 'NsssNH', 'MINsCH3', 'SpAD_Dzv', 'MAXaasN', 'NsSnH3', 'AATSC4Z', 'AATS4dv', 'SsCl', 'AATSC7d', 'nG12FAHRing', 'LabuteASA', 'AXp-3dv', 'GATS8m', 'MINssS', 'GATS6s', 'VE2_A', 'NsOH', 'ZMIC2', 'n6HRing', 'Xch-7dv', 'VSA_EState5', 'VE1_D', 'SpDiam_Dzi', 'ATSC3s', 'n6ARing', 'GATS2i', 'ATS4p', 'AXp-4dv', 'SssssC', 'AATS7dv', 'NssO', 'IC3', 'IC4', 'AATS5pe', 'SsssNH', 'AATS8v', 'AATS1i', 'MINaaN', 'NsssssP', 'n8FRing', 'VR2_DzZ', 'GATS6se', 'AATS8m', 'n4HRing', 'ATS2Z', 'SaaO', 'NaaN', 'AATS7pe', 'Xc-3d', 'MATS6d']
# Molecular Interactions
interaction_feats=['inter_2_VSAEstate7*_PolarityD1050', 'inter_2_VSAEstate4*712', 'inter_2_slogPVSA1*542', 'inter_2_PEOEVSA10*_PolarizabilityD1100', 'inter_2_VSAEstate5*_PolarityC3', 'inter_2_VSAEstate2*MoreauBrotoAuto_Polarizability26', 'inter_2_VSAEstate3*MVT', 'inter_2_VSAEstate6*MoreauBrotoAuto_AvFlexibility1', 'inter_2_EstateVSA9*GearyAuto_Mutability8', 'inter_2_slogPVSA0*QSOSW13', 'inter_2_EstateVSA9*FLD', 'inter_2_PEOEVSA10*VG', 'inter_2_PEOEVSA12*VSS', 'inter_2_slogPVSA11*QSOgrant48', 'inter_2_EstateVSA9*I', 'inter_2_VSAEstate5*266', 'inter_2_VSAEstate8*APAAC13', 'inter_2_slogPVSA5*APV', 'inter_2_EstateVSA10*MoranAuto_Mutability18', 'inter_2_PEOEVSA11*KMK', 'inter_2_MRVSA0*ISI', 'inter_2_EstateVSA0*LNP', 'inter_2_PEOEVSA12*APAAC8', 'inter_2_MRVSA2*RML', 'inter_2_VSAEstate5*PAAC38', 'inter_2_PEOEVSA8*_PolarityD1075', 'inter_2_MRVSA0*346', 'inter_2_slogPVSA6*MoranAuto_Polarizability21', 'inter_2_PEOEVSA0*MoranAuto_Steric18', 'inter_2_PEOEVSA1*taugrant30', 'inter_2_VSAEstate8*taugrant33', 'inter_2_PEOEVSA10*WEN', 'inter_2_PEOEVSA1*MoreauBrotoAuto_ResidueASA30', 'inter_2_slogPVSA3*SVR', 'inter_2_PEOEVSA4*TWY', 'inter_2_VSAEstate3*MoreauBrotoAuto_ResidueVol23', 'inter_2_VSAEstate3*MoranAuto_FreeEnergy6', 'inter_2_PEOEVSA9*GearyAuto_ResidueVol13', 'inter_2_PEOEVSA12*QSOgrant23', 'inter_2_EstateVSA8*535', 'inter_2_MRVSA6*RR', 'inter_2_PEOEVSA6*LTF', 'inter_2_PEOEVSA5*RP', 'inter_2_MRVSA9*taugrant45', 'inter_2_VSAEstate1*QSOgrant47', 'inter_2_PEOEVSA12*QSOgrant20', 'inter_2_EstateVSA0*MoranAuto_Steric26', 'inter_2_PEOEVSA1*611', 'inter_2_slogPVSA3*GearyAuto_FreeEnergy9', 'inter_2_VSAEstate0*VM', 'inter_2_EstateVSA8*QT', 'inter_2_MRVSA3*MoreauBrotoAuto_Hydrophobicity12', 'inter_2_MRVSA3*MoreauBrotoAuto_AvFlexibility9', 'inter_2_PEOEVSA6*464', 'inter_2_VSAEstate3*MoreauBrotoAuto_Steric16', 'inter_2_PEOEVSA1*IT', 'inter_2_PEOEVSA10*_PolarityD1001', 'inter_2_PEOEVSA8*MoreauBrotoAuto_Polarizability2', 'inter_2_PEOEVSA2*FN', 'inter_2_VSAEstate5*YR', 'inter_2_MTPSA*MoranAuto_AvFlexibility24', 'inter_2_VSAEstate0*VY', 'inter_2_slogPVSA4*IKA', 'inter_2_TPSA1*YPM', 'inter_2_slogPVSA7*MoreauBrotoAuto_Hydrophobicity18', 'inter_2_VSAEstate2*MoreauBrotoAuto_ResidueASA22', 'inter_2_VSAEstate0*466', 'inter_2_EstateVSA0*MoreauBrotoAuto_ResidueVol4', 'inter_2_EstateVSA8*GearyAuto_FreeEnergy12', 'inter_2_PEOEVSA10*TFS', 'inter_2_EstateVSA7*_HydrophobicityD1075', 'inter_2_PEOEVSA0*GearyAuto_ResidueASA3', 'inter_2_PEOEVSA0*GearyAuto_ResidueASA27', 'inter_2_MRVSA2*APAAC6', 'inter_2_VSAEstate5*423', 'inter_2_PEOEVSA11*AS', 'inter_2_MRVSA2*GearyAuto_Hydrophobicity17', 'inter_2_VSAEstate4*CA', 'inter_2_slogPVSA3*QSOSW35', 'inter_2_LabuteASA*MoreauBrotoAuto_Steric27', 'inter_2_VSAEstate2*MoreauBrotoAuto_Polarizability27', 'inter_2_VSAEstate7*SE', 'inter_2_PEOEVSA1*341', 'inter_2_VSAEstate3*MoranAuto_ResidueVol9', 'inter_2_MRVSA5*MoranAuto_FreeEnergy24', 'inter_2_VSAEstate6*MoreauBrotoAuto_FreeEnergy12', 'inter_2_PEOEVSA1*MoreauBrotoAuto_AvFlexibility6', 'inter_2_slogPVSA5*466', 'inter_2_VSAEstate6*LMI', 'inter_2_VSAEstate4*taugrant23', 'inter_2_PEOEVSA1*GearyAuto_Mutability24', 'inter_2_PEOEVSA9*MoreauBrotoAuto_Hydrophobicity11', 'inter_2_PEOEVSA6*ALY', 'inter_2_MRVSA9*MoranAuto_Hydrophobicity28', 'inter_2_PEOEVSA12*MoranAuto_Mutability2', 'inter_2_VSAEstate9*MoreauBrotoAuto_Hydrophobicity24', 'inter_2_MRVSA1*GearyAuto_Mutability12', 'inter_2_PEOEVSA9*_PolarityD3025', 'inter_2_slogPVSA1*KY', 'inter_2_VSAEstate2*GearyAuto_ResidueVol22', 'inter_2_PEOEVSA8*QSOSW37', 'inter_2_EstateVSA0*P', 'inter_2_MRVSA8*TL', 'inter_2_PEOEVSA7*MoreauBrotoAuto_ResidueASA8', 'inter_2_slogPVSA5*MoreauBrotoAuto_Mutability29', 'inter_2_PEOEVSA6*DT', 'inter_2_VSAEstate7*MoranAuto_Polarizability5', 'inter_2_PEOEVSA10*353', 'inter_2_VSAEstate3*LQM', 'inter_2_PEOEVSA0*QSOSW10', 'inter_2_PEOEVSA6*GearyAuto_FreeEnergy18', 'inter_2_MRVSA5*KRC', 'inter_2_VSAEstate7*GearyAuto_Steric14', 'inter_2_MTPSA*YG', 'inter_2_MRVSA4*MoreauBrotoAuto_Steric16', 'inter_2_EstateVSA10*_HydrophobicityC3', 'inter_2_PEOEVSA8*GGS', 'inter_2_VSAEstate6*PAAC30', 'inter_2_VSAEstate2*IDP', 'inter_2_TPSA1*GNF', 'inter_2_slogPVSA9*CIA', 'inter_2_slogPVSA0*MoreauBrotoAuto_AvFlexibility26', 'inter_2_VSAEstate5*FLD', 'inter_2_VSAEstate7*GearyAuto_Mutability21', 'inter_2_PEOEVSA5*MoreauBrotoAuto_Steric10', 'inter_2_VSAEstate3*GearyAuto_Polarizability20', 'inter_2_MRVSA0*MoranAuto_Polarizability5', 'inter_2_PEOEVSA8*GearyAuto_FreeEnergy27', 'inter_2_VSAEstate5*_SolventAccessibilityD1001', 'inter_2_MRVSA9*GearyAuto_Polarizability11', 'inter_2_MRVSA5*QSOgrant33', 'inter_2_VSAEstate3*MoreauBrotoAuto_Hydrophobicity4', 'inter_2_slogPVSA7*GearyAuto_Hydrophobicity12', 'inter_2_PEOEVSA7*MoranAuto_Mutability14', 'inter_2_VSAEstate9*VIS', 'inter_2_TPSA1*GearyAuto_Mutability7', 'inter_2_MRVSA5*275', 'inter_2_MRVSA9*KM', 'inter_2_PEOEVSA11*YL', 'inter_2_EstateVSA10*GearyAuto_Hydrophobicity1', 'inter_2_MRVSA5*ITR', 'inter_2_MRVSA9*MoreauBrotoAuto_Polarizability26', 'inter_2_VSAEstate5*QSOSW12', 'inter_2_PEOEVSA1*FAF', 'inter_2_EstateVSA1*GearyAuto_ResidueVol20', 'inter_2_LabuteASA*GearyAuto_ResidueVol27', 'inter_2_PEOEVSA10*744', 'inter_2_PEOEVSA10*LPQ', 'inter_2_slogPVSA4*GearyAuto_Polarizability20', 'inter_2_PEOEVSA7*MoreauBrotoAuto_FreeEnergy27', 'inter_2_MTPSA*GearyAuto_Mutability30', 'inter_2_MRVSA5*_SolventAccessibilityD1100', 'inter_2_slogPVSA11*_SolventAccessibilityD3100', 'inter_2_PEOEVSA10*FAF', 'inter_2_slogPVSA9*_ChargeT23', 'inter_2_PEOEVSA0*LYA', 'inter_2_slogPVSA5*QSOgrant10', 'inter_2_EstateVSA0*_ChargeD2050', 'inter_2_VSAEstate7*216', 'inter_2_slogPVSA1*LQM', 'inter_2_MRVSA6*RFT', 'inter_2_slogPVSA5*GearyAuto_ResidueVol19', 'inter_2_VSAEstate2*LYL', 'inter_2_PEOEVSA8*LMI', 'inter_2_slogPVSA0*_ChargeC3', 'inter_2_MTPSA*QSOSW19', 'inter_2_PEOEVSA1*122', 'inter_2_MRVSA8*_SolventAccessibilityT13', 'inter_2_VSAEstate3*PV', 'inter_2_MTPSA*323', 'inter_2_VSAEstate6*MoranAuto_Mutability2', 'inter_2_MRVSA8*MoranAuto_ResidueVol24', 'inter_2_PEOEVSA4*CIA', 'inter_2_VSAEstate7*427', 'inter_2_slogPVSA11*_PolarizabilityD2025', 'inter_2_MRVSA8*MoranAuto_Hydrophobicity23', 'inter_2_VSAEstate3*MoranAuto_FreeEnergy17', 'inter_2_MRVSA4*YSI', 'inter_2_PEOEVSA0*HP', 'inter_2_MRVSA3*636', 'inter_2_MRVSA5*KDR', 'inter_2_MRVSA1*GearyAuto_Polarizability6', 'inter_2_PEOEVSA6*QSOSW31', 'inter_2_PEOEVSA6*tausw5', 'inter_2_EstateVSA0*I', 'inter_2_slogPVSA1*MoreauBrotoAuto_Mutability16', 'inter_2_MRVSA5*TI', 'inter_2_MRVSA2*_PolarizabilityD1025', 'inter_2_MRVSA9*GearyAuto_Steric25', 'inter_2_PEOEVSA6*DEN', 'inter_2_PEOEVSA3*114', 'inter_2_slogPVSA2*MoranAuto_Mutability27', 'inter_2_LabuteASA*GearyAuto_ResidueASA10', 'inter_2_PEOEVSA0*MoranAuto_Polarizability28', 'inter_2_VSAEstate1*MoranAuto_Mutability2', 'inter_2_VSAEstate5*MoreauBrotoAuto_AvFlexibility29', 'inter_2_slogPVSA6*MoranAuto_FreeEnergy26', 'inter_2_slogPVSA3*MoranAuto_ResidueASA5', 'inter_2_VSAEstate5*GearyAuto_Steric29', 'inter_2_EstateVSA10*MoreauBrotoAuto_Steric3']
# Fingerprints
other_feats=['Morgan_FP_470', 'Avalon_FP_355', 'Morgan_FP_50', 'Avalon_FP_298', 'Avalon_FP_223', 'Avalon_FP_48', 'Avalon_FP_409', 'Avalon_FP_28', 'Morgan_FP_316', 'Morgan_FP_289', 'MACCS_FP_34', 'Avalon_FP_396', 'Avalon_FP_288', 'Morgan_FP_235', 'Morgan_FP_372', 'Avalon_FP_216', 'Avalon_FP_306', 'Morgan_FP_242', 'Avalon_FP_71', 'Morgan_FP_40', 'Avalon_FP_312', 'Morgan_FP_484', 'Morgan_FP_264', 'Avalon_FP_195', 'Morgan_FP_118', 'Avalon_FP_228', 'Avalon_FP_10', 'Morgan_FP_309', 'Avalon_FP_422', 'Morgan_FP_275', 'Morgan_FP_303', 'Avalon_FP_385', 'MACCS_FP_132', 'Avalon_FP_190', 'Morgan_FP_361', 'Morgan_FP_238', 'Avalon_FP_267', 'Morgan_FP_244', 'Morgan_FP_248', 'Avalon_FP_231', 'Avalon_FP_250', 'Morgan_FP_430', 'Morgan_FP_399', 'Morgan_FP_266', 'Morgan_FP_26', 'Morgan_FP_3', 'Avalon_FP_14', 'Avalon_FP_114', 'Morgan_FP_240', 'Morgan_FP_32', 'Avalon_FP_495', 'MACCS_FP_66', 'Morgan_FP_82', 'Avalon_FP_49', 'Avalon_FP_124', 'Morgan_FP_222', 'Avalon_FP_348', 'Morgan_FP_16', 'Avalon_FP_308', 'MACCS_FP_43', 'Avalon_FP_292', 'MACCS_FP_42', 'Avalon_FP_268', 'Morgan_FP_0', 'Morgan_FP_481', 'Avalon_FP_403', 'Morgan_FP_226', 'Morgan_FP_200', 'Morgan_FP_491', 'Avalon_FP_367', 'Avalon_FP_448', 'Morgan_FP_330', 'Morgan_FP_374', 'Morgan_FP_409', 'Avalon_FP_87', 'Avalon_FP_370', 'MACCS_FP_116', 'Morgan_FP_358', 'Morgan_FP_89', 'Morgan_FP_125', 'Morgan_FP_459', 'Morgan_FP_260', 'MACCS_FP_158', 'Avalon_FP_162', 'Avalon_FP_404', 'Morgan_FP_68', 'Avalon_FP_35', 'Morgan_FP_224', 'Morgan_FP_471', 'Morgan_FP_360', 'Morgan_FP_433', 'Morgan_FP_141', 'Morgan_FP_186', 'MACCS_FP_18', 'Morgan_FP_332', 'Morgan_FP_436', 'Morgan_FP_402', 'Morgan_FP_64', 'Avalon_FP_478', 'MACCS_FP_163', 'Morgan_FP_342', 'Avalon_FP_215', 'MACCS_FP_72', 'Morgan_FP_95', 'Avalon_FP_505', 'Avalon_FP_182', 'Avalon_FP_433', 'Morgan_FP_149', 'Morgan_FP_105', 'Avalon_FP_376', 'Morgan_FP_230', 'MACCS_FP_115', 'Avalon_FP_371', 'Avalon_FP_339', 'Avalon_FP_374', 'Morgan_FP_42', 'Morgan_FP_427', 'Avalon_FP_452', 'Morgan_FP_474', 'Avalon_FP_209', 'Avalon_FP_499', 'MACCS_FP_126', 'Morgan_FP_218', 'Morgan_FP_72', 'MACCS_FP_111', 'Avalon_FP_335', 'MACCS_FP_93', 'Morgan_FP_25', 'MACCS_FP_8', 'Avalon_FP_472', 'Avalon_FP_364', 'Morgan_FP_153', 'Morgan_FP_454', 'Morgan_FP_252', 'Morgan_FP_499', 'MACCS_FP_146', 'Morgan_FP_420', 'Avalon_FP_34', 'Morgan_FP_83', 'Avalon_FP_458', 'Morgan_FP_333', 'MACCS_FP_141', 'MACCS_FP_63', 'Morgan_FP_320', 'Avalon_FP_380', 'Avalon_FP_31', 'Morgan_FP_187', 'MACCS_FP_62', 'Avalon_FP_274', 'Morgan_FP_389']
merged_feats=ligand_feats+interaction_feats+other_feats


# Feature processing

def process_features(train_df, test_df, feature_list):
    imp = SimpleImputer(strategy="mean")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(imp.fit_transform(train_df[feature_list]))
    X_test = scaler.transform(imp.transform(test_df[feature_list]))
    return X_train, X_test

X_train_lig, X_test_lig = process_features(train_df, test_df, ligand_feats)
X_train_int, X_test_int = process_features(train_df, test_df, interaction_feats)
X_train_oth, X_test_oth = process_features(train_df, test_df, other_feats)
X_train_mer, X_test_mer = process_features(train_df, test_df, merged_feats)

feature_sets = {
    "Ligand": (X_train_lig, X_test_lig),
    "Interaction": (X_train_int, X_test_int),
    "Other": (X_train_oth, X_test_oth),
    "Merged": (X_train_mer, X_test_mer),
}

# Model
classifiers = {
    "ST": StackingClassifier(
        estimators=[
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('lg', LogisticRegression(max_iter=1000, solver='lbfgs')),
            ('svm', SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)),
        ],
        final_estimator=ExtraTreesClassifier(random_state=42),
        passthrough=True
    )
}

# Kaydetme fonksiyonu
def save_cv_predictions(results_dict):
    os.makedirs("saved_predictions", exist_ok=True)
    for name, content in results_dict.items():
        probs_list = []
        for split_idx, (test_idx, probs) in enumerate(content["probs"]):
            for idx, prob in zip(test_idx, probs):
                probs_list.append({
                    "split": split_idx,
                    "index": idx,
                    "prob": prob
                })
        df_probs = pd.DataFrame(probs_list)
        df_probs.to_csv(f"saved_predictions/{name}_probs.csv", index=False)
    joblib.dump(results_dict, "saved_predictions/all_model_probs.pkl")
    print("✅ Tahminler CSV ve PKL olarak kaydedildi.")

# Stratified CV
total_splits = 50
skf = StratifiedShuffleSplit(n_splits=total_splits, test_size=0.1, random_state=42)

for model_name, clf in classifiers.items():
    print(f"\n\U0001F4A1 MODEL: {model_name}")

    results = {"Ligand": {"probs": []}, "Interaction": {"probs": []}, "Other": {"probs": []}, "Merged": {"probs": []}}

    for name, (X_tr, X_te) in feature_sets.items():
        print(f"\n\U0001F4CA {name.upper()} FEATURES")
        metrics = {
            "agonist_recall": [], "antagonist_recall": [], "agonist_precision": [], "antagonist_precision": [],
            "f1_antagonist": [], "balanced_accuracy": [], "roc_auc": []
        }

        for train_idx, val_idx in skf.split(X_tr, y_train):
            clf.fit(X_tr[train_idx], y_train.iloc[train_idx])
            preds = clf.predict(X_tr[val_idx])
            probas = clf.predict_proba(X_tr[val_idx])[:, 1]
            y_true = y_train.iloc[val_idx]

            results[name]["probs"].append((val_idx, probas))

            metrics["agonist_recall"].append(recall_score(y_true, preds, pos_label=1))
            metrics["antagonist_recall"].append(recall_score(y_true, preds, pos_label=0))
            metrics["agonist_precision"].append(precision_score(y_true, preds, pos_label=1, zero_division=0))
            metrics["antagonist_precision"].append(precision_score(y_true, preds, pos_label=0, zero_division=0))
            metrics["f1_antagonist"].append(f1_score(y_true, preds, pos_label=0))
            metrics["balanced_accuracy"].append(balanced_accuracy_score(y_true, preds))
            metrics["roc_auc"].append(roc_auc_score(y_true, probas))

        print("\n\U0001F4CC 50-FOLD CROSS-VALIDATION")
        for k, v in metrics.items():
            mean = np.mean(v) * 100
            std = np.std(v) * 100
            print(f"{k.replace('_', ' ').title():<22}: {mean:.1f} ± {std:.1f}%")

        clf.fit(X_tr, y_train)
        preds_test = clf.predict(X_te)
        probas_test = clf.predict_proba(X_te)[:, 1]

        print("\n\U0001F4CC TEST SET PERFORMANCE")
        print(f"Agonist Recall         : {recall_score(y_test, preds_test, pos_label=1):.4f}")
        print(f"Antagonist Recall       : {recall_score(y_test, preds_test, pos_label=0):.4f}")
        print(f"Agonist Precision       : {precision_score(y_test, preds_test, pos_label=1, zero_division=0):.4f}")
        print(f"Antagonist Precision    : {precision_score(y_test, preds_test, pos_label=0, zero_division=0):.4f}")
        print(f"Balanced Accuracy       : {balanced_accuracy_score(y_test, preds_test):.4f}")
        print(f"Hold-out ROC AUC        : {roc_auc_score(y_test, probas_test):.4f}")
        print(f"Hold-out Average Prec.  : {average_precision_score(y_test, probas_test):.4f}")
        print(f"Hold-out F1 Score       : {f1_score(y_test, preds_test):.4f}")

    save_cv_predictions(results)

    # Ensemble with stored CV results
    weights = {"Ligand": 1, "Interaction": 1, "Other": 1, "Merged": 3}
    ensemble_metrics = []
    for split_idx in range(total_splits):
        combined_probs = None
        y_val = None
        for name, weight in weights.items():
            test_idx, probs = results[name]["probs"][split_idx]
            if combined_probs is None:
                combined_probs = weight * np.array(probs)
                y_val = y_train.iloc[test_idx]
            else:
                combined_probs += weight * np.array(probs)
        y_pred = (combined_probs > 0.5).astype(int)
        f1_an = f1_score(y_val, y_pred, pos_label=0)
        roc = roc_auc_score(y_val, combined_probs)
        ensemble_metrics.append((f1_an, roc))

    f1s, rocs = zip(*ensemble_metrics)
    print("\n\U0001F4CC 50-FOLD ENSEMBLE CV")
    print(f"F1 (Antagonist): {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}%")
    print(f"ROC AUC        : {np.mean(rocs)*100:.2f} ± {np.std(rocs)*100:.2f}%")

    # Final ensemble on test set
    prob_ens = np.zeros(len(y_test))
    for feat_name, weight in weights.items():
        X_train_current, X_test_current = feature_sets[feat_name]
        clf.fit(X_train_current, y_train)
        prob_ens += weight * clf.predict_proba(X_test_current)[:, 1]

    pred_ens = (prob_ens > 0.5).astype(int)

    print(f"\n\U0001F4CC WEIGHTED ENSEMBLE WITH {model_name}")
    print(f"Agonist Recall         : {recall_score(y_test, pred_ens, pos_label=1):.4f}")
    print(f"Antagonist Recall       : {recall_score(y_test, pred_ens, pos_label=0):.4f}")
    print(f"Agonist Precision       : {precision_score(y_test, pred_ens, pos_label=1, zero_division=0):.4f}")
    print(f"Antagonist Precision    : {precision_score(y_test, pred_ens, pos_label=0, zero_division=0):.4f}")
    print(f"Balanced Accuracy       : {balanced_accuracy_score(y_test, pred_ens):.4f}")
    print(f"Hold-out ROC AUC        : {roc_auc_score(y_test, prob_ens):.4f}")
    print(f"Hold-out Average Prec.  : {average_precision_score(y_test, prob_ens):.4f}")
    print(f"Hold-out F1 Score       : {f1_score(y_test, pred_ens):.4f}")
