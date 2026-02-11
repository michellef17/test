import os
import sys
import gzip
import argparse
import numpy as np
import pandas as pd

# Use Agg backend so no X server is required (avoids XIO fatal errors)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Bio import SeqIO
from pandarallel import pandarallel

# Initialize Pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=1)

# Import TeloBP functions
from TeloBP import getTeloBoundary, getTeloNPBoundary

# Error codes
errorReturns = {"init": -1, "fusedRead": -10, "strandType": -20, "seqNotFound": -1000}

# Globals
teloNP = False
showGraphsGlobal = False
outputColName = "teloBPLengths"
verbose = False

def vprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)

# ----------------------------
# Row processing functions
# ----------------------------
def rowToTeloBP(row, pdf_local=None):
    if row["seq"] is np.nan:
        return errorReturns["seqNotFound"]

    # If a PDF is passed, save plots into it
    if pdf_local:
        # First, a text page with the read name
        fig_text = plt.figure(figsize=(4,1))
        plt.text(0.5, 0.5, f'Qname: {row["qname"]}', ha='center', va='center', fontsize=8)
        plt.axis('off')
        pdf_local.savefig(fig_text)
        plt.close(fig_text)

        # Then call TeloBP boundary plotting, letting it plot into pdf_local
        result = getTeloBoundary(row["seq"], showGraphs=True, pdf=pdf_local)
    else:
        result = getTeloBoundary(row["seq"], showGraphs=False, pdf=None)

    return result

def rowToTeloNP(row, pdf_local=None):
    if row["seq"] is np.nan:
        return errorReturns["seqNotFound"]

    if pdf_local:
        fig_text = plt.figure(figsize=(4,1))
        plt.text(0.5, 0.5, f'Qname: {row["qname"]}', ha='center', va='center', fontsize=8)
        plt.axis('off')
        pdf_local.savefig(fig_text)
        plt.close(fig_text)

        result = getTeloNPBoundary(row["seq"], showGraphs=True, pdf=pdf_local)
    else:
        result = getTeloNPBoundary(row["seq"], showGraphs=False, pdf=None)

    return result

# ----------------------------
# Sample processing
# ----------------------------
def process_sampleDf(sampleDf, sampleKey, outputDir):
    pdf_local = None
    if showGraphsGlobal:
        os.makedirs(outputDir, exist_ok=True)
        pdf_path = os.path.join(outputDir, f"{sampleKey}_graphs.pdf")
        pdf_local = PdfPages(pdf_path)

    # Apply TeloBP/TeloNP with plots
    if teloNP:
        sampleDf[outputColName] = sampleDf.apply(lambda row: rowToTeloNP(row, pdf_local), axis=1)
    else:
        sampleDf[outputColName] = sampleDf.apply(lambda row: rowToTeloBP(row, pdf_local), axis=1)

    # Add summary histogram page
    if pdf_local:
        fig_hist = plt.figure(figsize=(6,4))
        lengths = sampleDf[sampleDf[outputColName] > 0][outputColName]
        plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Histogram of {outputColName} for {sampleKey}")
        plt.xlabel("Telo Length")
        plt.ylabel("Count")
        pdf_local.savefig(fig_hist)
        plt.close(fig_hist)

        pdf_local.close()

    # Save CSV without negative values
    sampleDf_no_seq = sampleDf.drop(columns=["seq"])
    dfNoNeg = sampleDf_no_seq[sampleDf_no_seq[outputColName] > 0].reset_index(drop=True)
    os.makedirs(outputDir, exist_ok=True)
    dfNoNeg.to_csv(os.path.join(outputDir, f"{sampleKey}.csv"), index=False)

    return sampleDf

# ----------------------------
# Main analysis function
# ----------------------------
def run_analysis(dataDir, fileMode, teloNPIn, outputDir, save_graphs=False, targetQnamesCSV=None):
    global teloNP, showGraphsGlobal, outputColName
    teloNP = teloNPIn
    if teloNP:
        outputColName = "teloNPLengths"

    showGraphsGlobal = save_graphs and (targetQnamesCSV is not None)
    if save_graphs and targetQnamesCSV is None:
        print("WARNING: --save_graphs set but no targetQnamesCSV provided. Graphs disabled.")
        showGraphsGlobal = False

    targetQnames = None
    if targetQnamesCSV:
        targetQnames = pd.read_csv(targetQnamesCSV, header=None)
        targetQnames = set(targetQnames[0].tolist())

    # Collect files
    filenames = []
    if not fileMode:
        for root, _, files in os.walk(dataDir):
            for filename in files:
                if filename.endswith(".fastq") or filename.endswith(".gz"):
                    filenames.append(os.path.join(root, filename))
    else:
        filenames.append(dataDir)

    # Process each file
    for filename in filenames:
        if not (filename.endswith(".fastq") or filename.endswith(".gz")):
            continue

        sampleKey = "_".join(filename.split("/")[-1].split(".")[:-1]).replace(" ", "_")
        qnameTeloValues = []

        if filename.endswith(".fastq.gz"):
            with gzip.open(filename, "rt") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    if targetQnames and record.id not in targetQnames:
                        continue
                    qnameTeloValues.append([record.id, record.seq])
        else:
            for record in SeqIO.parse(filename, "fastq"):
                if targetQnames and record.id not in targetQnames:
                    continue
                qnameTeloValues.append([record.id, record.seq])

        if not qnameTeloValues:
            print(f"No reads found in {filename}")
            continue

        sampleDf = pd.DataFrame(qnameTeloValues, columns=["qname", "seq"])
        process_sampleDf(sampleDf, sampleKey, outputDir)

    print("Analysis complete!")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TeloBP Analysis')
    parser.add_argument('dataDir', type=str, help='Path to data directory or file')
    parser.add_argument('outputDir', type=str, help='Path to output directory')
    parser.add_argument('--fileMode', action='store_true', help='If processing single file')
    parser.add_argument('--teloNP', action='store_true', help='Use teloNP')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--save_graphs', action='store_true', help='Save graphs to PDF')
    parser.add_argument('--targetQnamesCSV', type=str, help='CSV with Qnames to test')

    args = parser.parse_args()
    verbose = args.verbose

    run_analysis(
        args.dataDir,
        args.fileMode,
        args.teloNP,
        args.outputDir,
        save_graphs=args.save_graphs,
        targetQnamesCSV=args.targetQnamesCSV
    )
