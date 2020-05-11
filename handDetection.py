#!/usr/bin/env python

"""
########################################################################################################################
HAND DETECTION FUNCTIONS

WRITTEN BY:     Angus Berg, 440212490
DATE STARTED:   09/10/2018

LAST EDITED:   09/10/2018
########################################################################################################################
"""

###### IMPORTS ######
import cv2
import numpy as np

###### PARAMETERS ######
ED_SP  = 15
HT_CUT = 30

###### PROJECT FUNCTIONS ######
#See whether a hand is present and process it
def handIdentifier(boardImage):
    #Get a colour adjusted hand image
    adjustHand = handColAdjust(boardImage)

    #Get a mask for any hands in the image
    maskedHand = handMasking(adjustHand)

    #Get the bottom and side lines of the board image
    imageAreas = ((boardImage[:, :, 0] + boardImage[:, :, 1] + boardImage[:, :, 2]) != 0)
    imageShape = np.shape(imageAreas)
    bottomLine = np.zeros(imageShape)
    leftLines  = np.zeros(imageShape)
    rightLines = np.zeros(imageShape)

    for i in range(0, imageShape[1]):
        bottomIndx = np.nonzero(imageAreas[:, i])

        if bottomIndx[0].any():
            bottomLine[np.amax(bottomIndx[0]), i] = 1

    for i in range(0, imageShape[0]):
        sidesIndex = np.nonzero(imageAreas[i, :])

        if sidesIndex[0].any():
            leftLines[i, np.amin(sidesIndex[0])]  = 1
            rightLines[i, np.amax(sidesIndex[0])] = 1

    #Get the line between the dealer and player area
    dealerLine = np.zeros(imageShape)

    for i in range(0, imageShape[1]):
        dealerIndx = np.nonzero(imageAreas[:, i])

        if dealerIndx[0].any():
            dealerLine[np.amin(dealerIndx[0]), i] = 1

    #Get the sum of the hand's masking
    leftCross  = np.multiply(leftLines, maskedHand)
    rightCross = np.multiply(rightLines, maskedHand)
    bottmCross = np.multiply(bottomLine, maskedHand)
    handCross  = leftCross + rightCross + bottmCross
    summedArea = np.sum(np.sum(maskedHand, 1), 0)
    handCutoff = np.sum(np.sum(handCross, 1), 0)
    overReach  = np.sum(np.sum(np.multiply(dealerLine, maskedHand), 1), 0)

    #Determine if someone is overreaching
    if overReach > 15:
        return -1, -1

    #Determine if there is a hand in the image
    if (summedArea < 1000) | (handCutoff == 0):
        return 0, 0

    #Identify which side(s) the arm has crossed into from
    leftTotal  = np.sum(np.sum(leftCross, 1), 0)
    rightTotal = np.sum(np.sum(rightCross, 1), 0)
    bottmTotal = np.sum(np.sum(bottmCross, 1), 0)
    sidesCross = ((leftTotal > 15), (rightTotal > 15), (bottmTotal > 15))

    # Identify where the arm is gesturing to (pseudo-centroid)
    crossPoint = np.nonzero(handCross)
    crossPoint = (np.mean(crossPoint[0]), np.mean(crossPoint[1]))
    centroid, handImage = identifyCentroid(maskedHand, crossPoint, sidesCross)

    #Check that the hand wasn't too small
    if (centroid == 0) & (handImage.all() == 0):
        return -2, -2

    # Identify the gesture being made by the hand

    #Convert gesture found to bet placed

    #Store in hand class to keep variables straight
    handDetail = hand()
    handDetail.handArea = maskedHand
    handDetail.centroid = centroid

    return 1, handDetail

#Adjust the colour of the image to highlight a hand
def handColAdjust(boardImage):
    #Get a mask of the image space
    imageMask  = ((boardImage[:, :, 0] + boardImage[:, :, 1] + boardImage[:, :, 2]) != 0)

    #Convert image to LAB
    boardSpace = cv2.cvtColor(boardImage, cv2.COLOR_BGR2LAB)

    #Create the pseudo-mask
    boardHandM = -(-boardSpace[:, :, 2] - boardImage[:, :, 1] + boardImage[:, :, 2])

    #Adjust for minimums and maximums
    imageMinim = np.amin(np.amin(boardHandM, 1), 0)
    boardHandM = boardHandM - imageMinim
    imageMaxim = np.amax(np.amax(boardHandM, 1), 0)
    boardHandM = boardHandM/imageMaxim
    boardHandM = abs((boardHandM - 1))

    #Return the color-adjusted image
    return np.multiply(boardHandM, imageMask)
    

#Mask a hand in the image
def handMasking(greyImage):
    #Get the average intensity of the image pixels
    maskedArea = (greyImage != 0)
    avergInten = np.sum(np.sum(greyImage, 1), 0) / np.sum(np.sum(maskedArea, 1), 0)
    threshVal  = avergInten + 0.3

    #Threshhold the grey image at its average value
    imThreshed = cv2.threshold(greyImage, threshVal, 1, cv2.THRESH_BINARY)
    imThreshed = imThreshed[1]

    #Open the image to remove insignificant features
    morphInten = 2
    closKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*morphInten, 2*morphInten))
    openKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3*morphInten, 3*morphInten))
    imThreshed = cv2.morphologyEx(imThreshed, cv2.MORPH_CLOSE, closKernel)
    imThreshed = cv2.morphologyEx(imThreshed, cv2.MORPH_OPEN, openKernel)

    #Get the connected components of the binary image
    imageFeats = cv2.connectedComponents((imThreshed.astype(np.uint8)))

    #Search through the connected components to find which is the largest feature and keep that one
    foundFeat  = 0
    foundCount = 0
    for i in range(1, (imageFeats[0] + 1)):
        currFeat = (imageFeats[1] == i)
        pointTot = np.sum(np.sum(currFeat, 1), 0)

        if pointTot > foundCount:
            foundCount = pointTot
            foundFeat  = i

    if foundFeat != 0:
        imThreshed = (imageFeats[1] == foundFeat)
    else:
        imThreshed = 0 * imThreshed
        return imThreshed

    #Dilate the binary image that has been found
    imThreshed = cv2.morphologyEx((imThreshed.astype(np.uint8)), cv2.MORPH_DILATE, openKernel)

    return imThreshed

def identifyCentroid(areaMask, enterPoint, enterSides):
    #Find the limits of the hand in the image
    handFound  = np.nonzero(areaMask)
    handXLimit = np.array((np.amin(handFound[0]), np.amax(handFound[0])))
    handYLimit = np.array((np.amin(handFound[1]), np.amax(handFound[1])))

    #Create a mask of just the hand
    handXSpace = handXLimit[1] - handXLimit[0]
    handYSpace = handYLimit[1] - handYLimit[0]
    handImage  = np.zeros(((handXSpace + 2 * ED_SP), (handYSpace + 2 * ED_SP)))
    handImage[ED_SP:(handXSpace + ED_SP), ED_SP:(handYSpace + ED_SP)] = \
        areaMask[handXLimit[0]:handXLimit[1], handYLimit[0]:handYLimit[1]]

    #Check the hand section found is of sufficient size
    if handYSpace < HT_CUT:
        return 0, 0

    #Find points of hand furthest from side of entry
    edgePoints = []
    maskDimens = np.shape(areaMask)

    if enterSides[2]:
        for i in range(0, maskDimens[1], 5):
            pointIndex = np.argmax(areaMask[:, i])

            if pointIndex:
                edgePoints.append((pointIndex, i))

    if enterSides[0] | enterSides[1]:
        for i in range(0, maskDimens[0], 5):
            pointIndex = np.nonzero(areaMask[i, :])

            if pointIndex[0].any():
                leftIndex  = np.amax(pointIndex[0])
                rightIndex = np.amin(pointIndex[0])

                if enterSides[0]:
                    edgePoints.append((i, leftIndex))

                if enterSides[1]:
                    print('HELLO!!!!')
                    edgePoints.append((i, rightIndex))

    #Find the single furthest point from the point of entry
    fingerDist = 0
    fingerLoca = []
    for i in range(0, len(edgePoints)):
        ll = np.sqrt((edgePoints[i][0] - enterPoint[0]) ** 2 + (edgePoints[i][1] - enterPoint[1]) ** 2)

        if ll > fingerDist:
            fingerDist = ll
            fingerLoca = edgePoints[i]

    handFocus = (fingerLoca[1], fingerLoca[0])

    return handFocus, handImage


###### PROJECT CLASSES ######
class hand:
    def __init__(self):
        self.handArea = []
        self.centroid = []
        self.gesture  = 10

    def handMovement(self, prevMask = 0):
        #If there was no previous mask, all pixels found are movement
        if len(prevMask) == 0:
            return 1

        #Get the sum of hand areas and difference in hand areas
        diffArea = np.logical_xor(self.handArea, prevMask)
        movement = np.sum(np.sum(diffArea, 1), 0)
        totCover = np.sum(np.sum(self.handArea, 1), 0) + np.sum(np.sum(prevMask, 1), 0)

        #Get the proportional movement of the hand
        totShift = movement / totCover

        return totShift





