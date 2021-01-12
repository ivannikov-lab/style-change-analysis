#!/usr/bin/env python

"""Calculates the measures for the PAN17 style breach detection task"""

from __future__ import division

import os
import getopt
import sys
from models.utils.metrics import Metrics, metrics


def getMeasureString(measureName, value):
    """Returns the string represenation of one measure with its value."""
    return "measure{\n  key: \"" + measureName + "\"\n  value: \"" + str(value) + "\"\n}"

def computeMeasures(inputText, groundTruthData, producedData):
    """Computes WindowDiff and WinPR for the given data"""

    (groundTruthWordPositions, producedDataWordPositions, totalWordCount) = getWordPositionsFromCharacterPositions(inputText, groundTruthData["style_breaches"], producedData["style_breaches"])

    groundTruthStrArray = []
    producedStrArray = []

    for i in range(0,totalWordCount-1):
        if i in groundTruthWordPositions: groundTruthStrArray.append("1")
        else: groundTruthStrArray.append("0")

        if i in producedDataWordPositions: producedStrArray.append("1")
        else: producedStrArray.append("0")

    # mark last position
    groundTruthStrArray.append("1")
    producedStrArray.append("1")

    groundTruthString = ''.join(groundTruthStrArray)
    producedString = ''.join(producedStrArray)

    halfSegmentLength = 0
    if len(groundTruthWordPositions) == 0: halfSegmentLength = round(totalWordCount / 2)
    else: halfSegmentLength = round(totalWordCount / (len(groundTruthWordPositions) + 1) / 2)

    metric = Metrics(groundTruthString, producedString, halfSegmentLength)
    winP = metric.precision()
    winR = metric.recall()
    winF = metric.f1score()
    accuracy = metric.accuracy()
    #print("winDiff: ", winDiff)
    #print("winR: ", winR)
    #print("winP: ", winP)
    #print("winF: ", fscore(winR, winP))

    return (winR, winP, winF, accuracy)


def getWordPositionsFromCharacterPositions(text, groundTruthCharPositions, producedCharPositions):
    wordCount = 0
    groundTruthWordPositions = []
    producedWordPositions = []

    for i in range(1,len(text)-1):
        if text[i] == ' ' and text[i-1] != ' ':
            wordCount = wordCount + 1
        if i in groundTruthCharPositions:
            groundTruthWordPositions.append(wordCount)
        if i in producedCharPositions:
            producedWordPositions.append(wordCount)

    return (groundTruthWordPositions, producedWordPositions, wordCount+1)


def style_breach_evaluation(truthDict, predictions, texts):
    problemsCount = 0
    totalWinR = 0
    totalWinP = 0
    totalWinF = 0
    totalAcc = 0
    for truthData, producedData, inputText in zip(truthDict, predictions, texts):
        problemsCount = problemsCount + 1

        if not "style_breaches" in truthData:
            sys.exit("There is no 'positions' key")
        if not "style_breaches" in producedData:
            sys.exit("There is no 'positions' key")

        (winR, winP, winF, acc) = computeMeasures(inputText, truthData, producedData)
        totalWinR += winR
        totalWinP += winP
        totalWinF += winF
        totalAcc += acc

    if problemsCount == 0:
        sys.exit("The input dataset folder contains no style breach detection problems. Be sure to pass the correct folder (option -d).")

    totalWinR = totalWinR / problemsCount
    totalWinP = totalWinP / problemsCount
    totalWinF = totalWinF / problemsCount
    totalAcc = totalAcc / problemsCount

    return totalWinP, totalWinR, totalWinF, totalAcc


def style_change_evaluation(truthDict, predictions):
    winP, winR, winF, acc = metrics(truthDict, predictions)
    return winP, winR, winF, acc


def main(inputDataset, predictions, predictBreaches, config, data_name):
    texts = [text for _, text, _ in inputDataset]
    truthDict = [ans if predictBreaches else ans['style_change'] for _, _, ans in inputDataset]

    if predictBreaches:
        WinP, WinR, WinF, Acc = style_breach_evaluation(truthDict, predictions, texts)
    else:
        predictions = [d['style_change'] for d in predictions]
        WinP, WinR, WinF, Acc = style_change_evaluation(truthDict, predictions)
    outStr = getMeasureString("winP", WinP)
    outStr += "\n" + getMeasureString("winR", WinR)
    outStr += "\n" + getMeasureString("winF", WinF)
    outStr += "\n" + getMeasureString("Acc", Acc)
    name = "-".join([data_name, config["task"], config["model"]["name"],
                     "-".join(str(v) for v in config["model"]["hyperparams"]),
                    str(config["features"]["selection"]), str(config["features"]["pca"])])
    with open(os.path.join(config["resultDir"], name), 'w', encoding='utf-8') as outFile:
        outFile.write(outStr)

    return WinP, WinR, WinF, Acc