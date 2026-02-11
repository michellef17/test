import os
import sys
import gzip
import tempfile
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
from pandarallel import pandarallel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger

# Initialize Pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=1)  # Single worker for PDF safety

# sys.path.insert(0, '../TeloBP')
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
def rowToTeloBP(row, sampleKey, pdf_local=None):
    if row["seq"] is np.nan:
        return errorReturns["seqNotFound"]

    # Save a figure if PDF is provided
    if pdf_local:
        plt.figure()
        plt.text(0.5, 0.5, f'Sample: {sampleKey}, Qname: {row["qname"]}', ha='center', va='center', fontsize=12)
        plt.axis('off')
        pdf_local.savefig()
        plt.close()

    return getTeloBoundary(row["seq"], showGraphs=bool(pdf_local), pdf=pdf_local)

def rowToTeloNP(row, sampleKey, pdf_local=None):
    if row["seq"] is np.nan:
        return errorReturns["seqNotFound"]

    if pdf_local:
        plt.figure()
        plt.text(0.5, 0.5, f'Sample: {sampleKey}, Qname: {row["qname"]}', ha='center', va='center', fontsize=9)
        plt.axis('off')
        pdf_local.savefig()
        plt.close()

    return getTeloNPBoundary(row["seq"], showGraphs=bool(pdf_local), pdf=pdf_local)

# ----------------------------
# Sample processing
# ----------------------------
def process_sampleDf(sampleDf, sampleKey, outputDir):
    pdf_temp_path = None
    pdf_local = None
    if showGraphsGlobal:
        pdf_temp_path = os.path.join(outputDir, f"{sampleKey}_graphs.pdf")
        pdf_local = PdfPages(pdf_temp_path)

    # Apply row function
    if teloNP:
        sampleDf[outputColName] = sampleDf.apply(lambda row: rowToTeloNP(row, sampleKey, pdf_local), axis=1)
    else:
        sampleDf[outputColName] = sampleDf.apply(lambda row: rowToTeloBP(row, sampleKey, pdf_local), axis=1)

    if pdf_local:
        pdf_local.close()

    # Save CSV
    saveDfNoNeg(sampleDf.drop(columns=["seq"]), outputColName, outputDir, sampleKey)

    return sampleDf

# ----------------------------
# CSV saving
# ----------------------------
def saveDfNoNeg(df, outputColName, outputDir, sampleKey):
    dfNoNeg = df[df[outputColName] > 0].reset_index(drop=True)
    os.makedirs(outputDir, exist_ok=True)
    dfNoNeg.to_csv(os.path.join(outputDir, f"{sampleKey}.csv"), index=False)

# ----------------------------
# Main analysis function
# ----------------------------
def run_analysis(dataDir, fileMode, teloNPIn, outputDir, save_graphs=False, targetQnamesCSV=None):
    global teloNP, showGraphsGlobal, outputColName
    teloNP = teloNPIn
    if teloNP:
        outputColName = "teloNPLengths"

    showGraphsGlobal = save_graphs and targetQnamesCSV is not None
    if save_graphs and targetQnamesCSV is None:
        print("WARNING: --save_graphs set but no targetQnamesCSV provided. Graph saving disabled.")
        showGraphsGlobal = False

    targetQnames = None
    if targetQnamesCSV:
        targetQnames = pd.read_csv(targetQnamesCSV, header=None)
        targetQnames = set(targetQnames[0].tolist())

    # Collect filenames
    filenames = []
    if not fileMode:
        for root, _, files in os.walk(dataDir):
            for filename in files:
                if filename.endswith(".fastq") or filename.endswith(".gz"):
                    filenames.append(os.path.join(root, filename))
    else:
        filenames.append(dataDir)

    sampleQnames = {}

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
        sampleQnames[sampleKey] = process_sampleDf(sampleDf, sampleKey, outputDir)

    # ----------- Stats -----------
    totalReads = nonNegativeReads = totalReadLengths = 0
    initErrors = fusedReadErrors = strandTypeErrors = seqNotFound = 0

    for sampleKey, df in sampleQnames.items():
        sampleTotalReads = len(df)
        sampleNonNeg = len(df[df[outputColName] >= 0])
        totalReads += sampleTotalReads
        nonNegativeReads += sampleNonNeg
        totalReadLengths += df[df[outputColName] >= 0][outputColName].sum()
        initErrors += len(df[df[outputColName] == errorReturns["init"]])
        fusedReadErrors += len(df[df[outputColName] == errorReturns["fusedRead"]])
        strandTypeErrors += len(df[df[outputColName] == errorReturns["strandType"]])
        seqNotFound += len(df[df[outputColName] == errorReturns["seqNotFound"]])

    avgTeloBP = totalReadLengths / nonNegativeReads if nonNegativeReads else 0

    print(f"Total reads: {totalReads}")
    print(f"Non-negative reads: {nonNegativeReads}")
    print(f"Average teloBP length: {avgTeloBP}")
    print(f"Initialization errors: {initErrors}")
    print(f"Fused read errors: {fusedReadErrors}")
    print(f"Strand type errors: {strandTypeErrors}")
    print(f"Seq not found errors: {seqNotFound}")

# ----------------------------
# Main CLI
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
