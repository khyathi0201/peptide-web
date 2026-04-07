from flask import Flask, render_template, request, send_file, redirect, url_for
import joblib
import pandas as pd
from feature_extractor import extract_features
from Bio import SeqIO
import io
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
import urllib.request
#redeploy
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

app = Flask(__name__)

results_global = []
# -----------------------------
# DOWNLOAD MODELS (FROM DRIVE)
# -----------------------------

download_file("https://drive.google.com/uc?id=1ja2soabCDCjgMJesk1hon1kHOpMKpjhu", "svm_amp_model.pkl")
download_file("https://drive.google.com/uc?id=1xv7R5qVqZ1Y8tSHpAn3pU3XeDspL7d4n", "scaler.pkl")

download_file("https://drive.google.com/uc?id=1T0QotxZWguo_u0h8NSN-7ZrwsJN8-4G0", "rf_toxicity_model.pkl")
download_file("https://drive.google.com/uc?id=1mbFyQK4QsinpQgkvyD8N0JLm2aOAMnqy", "scaler_tox.pkl")

download_file("https://drive.google.com/uc?id=1fmNynDZo4eXDrocJqMlJL2e5FYlDyeRw", "xgboost_mic_model.pkl")
download_file("https://drive.google.com/uc?id=1lsGLnAg9bu-nkBSi4aLdove3uWcGcMSm", "scaler_mic.pkl")

# Load models
svm_model = joblib.load("svm_amp_model.pkl")
toxicity_model = joblib.load("rf_toxicity_model.pkl")
activity_model = joblib.load("xgboost_mic_model.pkl")

scaler_svm = joblib.load("scaler.pkl")
scaler_tox = joblib.load("scaler_tox.pkl")
scaler_mic = joblib.load("scaler_mic.pkl")

# Load feature order for 13 features
feature_columns = pd.read_csv("feature_columns.csv", header=None)
feature_order = feature_columns[0].tolist()


# -----------------------------
# CLEAN SEQUENCE
# -----------------------------
def clean_sequence(seq):
    valid_aa = "ACDEFGHIKLMNPQRSTVWY"
    seq = str(seq).upper()
    seq = "".join([aa for aa in seq if aa in valid_aa])
    return seq


# -----------------------------
# FEATURE EXTRACTION (13)
# -----------------------------
def extract_13_features(sequence):
    analysed_seq = ProteinAnalysis(sequence)

    length = len(sequence)
    molecular_weight = analysed_seq.molecular_weight()
    isoelectric_point = analysed_seq.isoelectric_point()
    instability_index = analysed_seq.instability_index()
    gravy = analysed_seq.gravy()

    pos = sequence.count("K") + sequence.count("R")
    neg = sequence.count("D") + sequence.count("E")
    net_charge = pos - neg

    hydrophobic = sum(sequence.count(aa) for aa in "AILMFWV")
    polar = sum(sequence.count(aa) for aa in "STNQ")
    nonpolar = hydrophobic
    aliphatic = sum(sequence.count(aa) for aa in "AILV")

    hydrophobic_ratio = hydrophobic / length if length > 0 else 0
    polar_ratio = polar / length if length > 0 else 0
    nonpolar_ratio = nonpolar / length if length > 0 else 0
    aliphatic_index = aliphatic / length if length > 0 else 0
    positive_ratio = pos / length if length > 0 else 0

    features = {
        "Amphiphilicity Index": hydrophobic_ratio * positive_ratio,
        "Angle Subtended by the Hydrophobic Residues": hydrophobic_ratio,
        "Disordered Conformation Propensity": instability_index,
        "Isoelectric Point": isoelectric_point,
        "Linear Moment": gravy,
        "Net Charge": net_charge,
        "Normalized Hydrophobic Moment": hydrophobic_ratio,
        "Normalized Hydrophobicity": gravy,
        "Penetration Depth": nonpolar_ratio,
        "Propensity to PPII coil": polar_ratio,
        "Propensity to in vitro Aggregation": hydrophobic_ratio,
        "Tilt Angle": aliphatic_index,
        "molecular_weight": molecular_weight
    }

    return pd.DataFrame([features])


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        results = []
        sequence = request.form.get("sequence")
        file = request.files.get("fasta_file")

        sequences = []

        if file and file.filename != "":
            fasta_sequences = SeqIO.parse(io.TextIOWrapper(file, encoding="utf-8"), "fasta")
            for fasta in fasta_sequences:
                sequences.append((fasta.id, str(fasta.seq)))
        elif sequence:
            sequences.append(("User_Sequence", sequence))

        for seq_id, seq in sequences:

            seq = clean_sequence(seq)
            if len(seq) == 0:
                continue

            # -------- AMP Prediction (534 features) --------
            vec = extract_features(seq)      # (534,)
            vec = vec.reshape(1, -1)         # (1, 534)
            vec_scaled = scaler_svm.transform(vec)
            amp_prob = svm_model.predict_proba(vec_scaled)[0][1]

            # -------- Toxicity & Activity --------
            features_13 = extract_13_features(seq)
            features_13 = features_13[feature_order]

            net_charge = features_13["Net Charge"].values[0]

            features_scaled_tox = scaler_tox.transform(features_13)
            features_scaled_mic = scaler_mic.transform(features_13)

            tox_prob = toxicity_model.predict_proba(features_scaled_tox)[0][1]
            act_prob = activity_model.predict_proba(features_scaled_mic)[0][1]

            # -------- Decision Logic --------
            if net_charge <= 0:
                amp_result = "Not AMP"
                tox_result = "Safe"
                act_result = "Inactive"
                final_result = "Non-Antimicrobial Peptide"
            else:
                amp_result = "AMP" if amp_prob > 0.5 else "Not AMP"
                tox_result = "Toxic" if tox_prob > 0.45 else "Safe"

                if act_prob > 0.7:
                    act_result = "Highly Active"
                elif act_prob > 0.4:
                    act_result = "Active"
                else:
                    act_result = "Inactive"

                if amp_result == "AMP" and tox_result == "Safe" and act_result in ["Active", "Highly Active"]:
                    final_result = "Safe Antimicrobial Peptide"
                elif amp_result == "AMP" and tox_result == "Toxic":
                    final_result = "Toxic Antimicrobial Peptide"
                elif amp_result == "AMP" and act_result == "Inactive":
                    final_result = "Inactive Antimicrobial Peptide"
                elif amp_result == "Not AMP" and act_result in ["Active", "Highly Active"]:
                    final_result = "Active Non-AMP Peptide"
                else:
                    final_result = "Non-Antimicrobial Peptide"

            results.append({
                "id": seq_id,
                "amp_result": amp_result,
                "amp_prob": round(float(amp_prob), 3),
                "tox_result": tox_result,
                "tox_prob": round(float(tox_prob), 3),
                "act_result": act_result,
                "act_prob": round(float(act_prob), 3),
                "final_result": final_result
            })

        global results_global
        results_global = results
        return redirect(url_for("results_page"))

    return render_template("index.html", results=None)


@app.route("/results", methods=["GET"])
def results_page():
    global results_global
    return render_template("index.html", results=results_global)


@app.route("/download_csv")
def download_csv():
    df = pd.DataFrame(results_global)
    df.to_csv("batch_results.csv", index=False)
    return send_file("batch_results.csv", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
