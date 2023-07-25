#!/usr/bin/env python3

import argparse
import os
import re
import sys
import json as j

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
    perfTree = parseTimeRatioTree(lines)
    return perfTree

def verifyCorrectness(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    regex = r"((?:Error Residual)|(?:Maximal error in eigenvector lengths)|(?:Error Orthogonality))(?:\s*:\s*)([0-9\.E-]+)"
    verificationLines = [re.search(regex, l).group(1,2) for l in lines if re.search(regex,l)]
    if len(verificationLines) != 3:
        print(f"Incorrect number of lines for the result verification. Expected 3 but got {len(verificationLines)}")
        return false
    for resVerType, res in verificationLines:
        if float(res) > 1e-9:
            print(f"Incorrect Result: {resVerType}: {res} > 1e-9")
            return False
    print("All is well:")
    for v, r in verificationLines:
        print(f"{v} = {r}")
    return True

def getPerformance(filename):
    perfTree = loadData(filename)
    for node in perfTree:
        depth = node[1]
        time = node[2]
        if depth == 0:
            return time
    print("Malformed Input!")
    return float('nan')

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
