#!/usr/bin/env python3

import argparse
import os
import re
import sys
import json as j
import numpy as np
import pandas as pd

def readConf():
    parser = argparse.ArgumentParser(description="ELPA Result Analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-v', action='store_true', help='Check for correctness')
    group.add_argument('-p', action='store_true', help='Check for performance')
    parser.add_argument('-w', type=str, help='Workload Name in Xe Performance Dashboard - column H', default = 'ELPA')
    parser.add_argument('-i', type=str, help='Input Name in Xe Performance Dashboard - column K', default = 'N.A.')
    parser.add_argument('-r', type=int, help='Number of Ranks', default = 1)
    parser.add_argument("filename", type=str, help="Path to the program output.")
    return parser.parse_args()

def parseMetaData(lines):
    res = {}
    metadataLines = [l for l in lines if re.match("[\w-]+ = -?\d+( -> \w+)?", l)]
    for ml in metadataLines:
        if len(ml) < 3:
            continue
        tokens = entry = ml.split(" ")
        entry = tokens[0]
        rawData = float(tokens[2])
        if len(tokens) >= 5:
            expl = tokens[4]
        else:
            expl = None
        res[entry] = rawData, expl
    return res

def extractPrognameData(progName):
    comps = progName.split("_")
    numberType = f"{comps[1]} {comps[2]}"
    codepath = "1-Stage" if "1stage" in comps else "2-Stage"
    return numberType, codepath

def parseRunData(lines):
    res = {}
    rundataLines = [l for l in lines if re.match("[\w ]+: \d+", l)]
    for rl in rundataLines:
        tokens = rl.split(":")
        entry = tokens[0]
        data = float(tokens[1].strip())
        res[entry] = data
    rundataLines = [l for l in lines if re.match("[\w ]+: \w+", l)]
    for rl in rundataLines:
        tokens = rl.split(":")
        entry = tokens[0]
        data = tokens[1].strip()
        res[entry] = data
    rundataLines = [l for l in lines if l.startswith('Number of processor')][0]
    processorData = re.findall("\d+", rundataLines)
    res['processor rows'] = processorData[0]
    res['processor cols'] = processorData[1]
    res['total processors'] = processorData[2]
    res['number type'], res["codepath"] = extractPrognameData(lines[0].split(" ")[1])
    return res

def getIndentDepth(line):
    initialSpace = re.findall("^( )+\|", line)
    if len(initialSpace) == 0:
        return -1
    else:
        x = 0
        while line[x] == " ":
            x += 1
        return int((x - 1) / 2)

def parseTimeRatioTree(lines):
    res = []
    for l in lines:
        indentLevel = getIndentDepth(l)
        if indentLevel >= 0:
            nums = re.findall("\d+\.?\d*", l)
            if len(nums) == 2:
                title = re.findall("[\w%\(\)][\w\%\(\)]+", l)[0]
                repetitions = 0
                time = float(nums[0])
                ratio = float(nums[1])
                res.append([title, indentLevel, time, ratio])
            elif len(nums) == 3:
                title = re.findall("[\w%\(\)][\w\%\(\)]+", l)[0]
                repetitions = nums[0]
                time = float(nums[1])
                ratio = float(nums[2])
                res.append([title, indentLevel, time, ratio])
    return res

def loadData(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    metaData = parseMetaData(lines)
    runData = parseRunData(lines)
    perfTree = parseTimeRatioTree(lines)
    df = pd.DataFrame(perfTree, columns=["component", "depth", "time", "ratio"])
    df['matrixSize'] = runData['Matrix size']
    df['numEigenvectors'] = runData['Num eigenvectors']
    df['blockSize'] = runData['Blocksize']
    df['mpiRanks'] = runData['Num MPI proc']
    df['filename'] = filename[:-4]
    df['numberType'] = runData["number type"]
    df['code'] = runData["codepath"]
    return df

def verifyCorrectness(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    regex = r"((?:Error Residual)|(?:Maximal error in eigenvector lengths)|(?:Error Orthogonality))(?:\s*:\s*)([0-9\.E-]+)"
    verificationLines = [re.search(regex, l).group(1,2) for l in lines if re.search(regex,l)]
    if len(verificationLines) != 3:
        print(f"Incorrect number of lines for the result verification. Expected 3 but got {len(verificationLines)}")
        return false
    for resVerType, res in verificationLines:
        if float(res) > 1e-10:
            print(f"Incorrect Result: {resVerType}: {res} > 1e-10")
            return False
    print("All is well:")
    for v, r in verificationLines:
        print(f"{v} = {r}")
    return True

def getPerformance(filename):
    df = loadData(filename)
    rList = df[(df.depth == 0)].time.to_list()
    if len(rList) != 1:
        return float('nan')
    else:
        return rList[0]

args = readConf()

if args.v:
    isCorrect = verifyCorrectness(args.filename)
    sys.exit(0 if isCorrect else 1)
else:
    time = getPerformance(args.filename)
    res = {
        "workload" : args.w,
        "input"    : args.i,
        "value"    : time
    }
    if time != time:
        print("Result not found!")
        sys.exit(1)
    else:
        print(j.dumps(res))
        sys.exit(0)
