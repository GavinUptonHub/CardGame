#!/usr/bin/env python

"""
########################################################################################################################
BOARD ISOLATION FUNCTIONS

WRITTEN BY:     Angus Berg, 440212490
DATE STARTED:   12/10/2018

LAST EDITED:   12/10/2018
########################################################################################################################
"""

###### IMPORTS ######
import cv2
import numpy as np

###### PARAMETERS ######
CENTRE_LENGTH = 785

###### PROJECT FUNCTIONS ######

#Find the edges of the board and eliminate features outside of the board area
def defineIsolation(areaImage):
    #Get an image which is the difference of the green and red channels. Normalise
    areaIsolate = areaImage[:, :, 2]
    areaIsolate = (areaIsolate<100)
    areaDimens  = np.shape(areaIsolate)

    #Close gaps in the board area to account for the central line
    closeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    areaIsolate = cv2.morphologyEx((areaIsolate.astype(np.uint8)), cv2.MORPH_CLOSE, closeKernel)

    #Identify the largest feature; keep that and eliminate others.
    areaFeature = cv2.connectedComponents((areaIsolate.astype(np.uint8)))

    foundFeat  = 0
    foundCount = 0
    for i in range(1, (areaFeature[0] + 1)):
        currFeat = (areaFeature[1] == i)
        pointTot = np.sum(np.sum(currFeat, 1), 0)

        if pointTot > foundCount:
            foundCount = pointTot
            foundFeat  = i

    if foundFeat != 0:
        areaIsolate = (areaFeature[1] == foundFeat)
    else:
        areaIsolate = 0 * areaIsolate
        return areaIsolate

    #Run along the found feature and fill in the gaps
    for i in range(0, areaDimens[0]):
        featureLine = areaIsolate[i, :]

        #Find first and last feature positions in this line
        featIndexes = np.nonzero(featureLine)

        if np.any(featIndexes[0]):
            featLimits  = [np.amin(featIndexes), np.amax(featIndexes)]

            #Make all elements between these limits into trues and place back in isolation image
            featureLine[featLimits[0]:featLimits[1]] = 1
            areaIsolate[i, :] = featureLine

    #Erode the feature slightly to account for errors
    erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    areaIsolate = cv2.morphologyEx((areaIsolate.astype(np.uint8)), cv2.MORPH_ERODE, erodeKernel)

    #Blur feature found
    areaBlurred = 255*cv2.blur(areaIsolate.astype(np.float), (1, 1))
    areaBlurred = areaBlurred.astype(np.uint8)

    # Generate the theoretical corner points
    totalPoints = np.nonzero(areaIsolate)
    xEdgePoints = np.array((np.amin(totalPoints[0]), np.amax(totalPoints[0])))
    yEdgePoints = np.array((np.amin(totalPoints[1]), np.amax(totalPoints[1])))
    trueCorners = np.zeros((4, 2))
    trueCorners[(0, 1), 0] = yEdgePoints[0]
    trueCorners[(2, 3), 0] = yEdgePoints[1]
    trueCorners[(1, 2), 1] = xEdgePoints[1]
    trueCorners[(0, 3), 1] = xEdgePoints[0]

    #Find corners of the feature
    areaCorners = cv2.cornerHarris(areaBlurred, 2, 1, 0.001)
    cornerKern  = np.ones((7, 7), np.float)
    areaCorners = cv2.filter2D(areaCorners, -1, cornerKern)
    areaCorners = (areaCorners > 0.05)
    indxCorners = np.nonzero(areaCorners)

    #Average the corners which are close together
    currentIndx = 0
    numberCorns = 0
    corAccuracy = 5
    cornerPoint = np.zeros((0, 2))
    currentCoor = np.zeros((1, 2))

    while currentIndx < len(indxCorners[0]):
        #Figure out which of the corners are placed together
        varianceX = np.abs(indxCorners[0] - indxCorners[0][currentIndx])
        varianceX = (varianceX < corAccuracy)
        varianceY = np.abs(indxCorners[1] - indxCorners[1][currentIndx])
        varianceY = (varianceY < corAccuracy)
        cornerSet = (varianceX & varianceY)

        #Get the average position of the corner
        trueY = np.sum(np.multiply(indxCorners[0], cornerSet)) / np.sum(cornerSet)
        trueY = np.floor(trueY)
        trueX = np.sum(np.multiply(indxCorners[1], cornerSet)) / np.sum(cornerSet)
        trueX = np.floor(trueX)

        #Store the current corner point, increase the index
        currentCoor[0, 0] = trueX.astype(np.uint16)
        currentCoor[0, 1] = trueY.astype(np.uint16)
        cornerPoint = np.concatenate((cornerPoint, currentCoor), 0)
        currentIndx = currentIndx + np.sum(cornerSet)
        numberCorns = numberCorns + 1

    #Find which of the found corners are closest to the theoretical points
    matchCorner = np.zeros((4, 2))
    matchDistan = np.inf * np.ones((4, 1))
    distanTests = np.zeros((4, 1))

    for i in range(0, numberCorns):
        #Get the distances between theoretical corners and current test corner
        distanTests[0, 0] = np.sqrt((cornerPoint[i,0]-trueCorners[0,0])**2 + (cornerPoint[i,1]-trueCorners[0,1])**2)
        distanTests[1, 0] = np.sqrt((cornerPoint[i,0]-trueCorners[1,0])**2 + (cornerPoint[i,1]-trueCorners[1,1])**2)
        distanTests[2, 0] = np.sqrt((cornerPoint[i,0]-trueCorners[2,0])**2 + (cornerPoint[i,1]-trueCorners[2,1])**2)
        distanTests[3, 0] = np.sqrt((cornerPoint[i,0]-trueCorners[3,0])**2 + (cornerPoint[i,1]-trueCorners[3,1])**2)

        #Get the index of the minimum displacement from a corner
        minimumIndx = np.argmin(distanTests, 0)

        #Compare with matched distance to see if point is closer
        if distanTests[minimumIndx, 0] < matchDistan[minimumIndx, 0]:
            matchDistan[minimumIndx, 0] = distanTests[minimumIndx, 0]
            matchCorner[minimumIndx, 0] = cornerPoint[i, 0]
            matchCorner[minimumIndx, 1] = cornerPoint[i, 1]

    cornerPoint = matchCorner

    #Get the perspective transform of the image
    cornerPoint = cornerPoint.astype(np.float32)
    trueCorners = trueCorners.astype(np.float32)
    transMatrix = cv2.getPerspectiveTransform(cornerPoint, trueCorners)

    #Get the output image
    erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    areaIsolate = cv2.morphologyEx((areaIsolate.astype(np.uint8)), cv2.MORPH_ERODE, erodeKernel)
    imageShape  = np.shape(areaIsolate)
    areaIsolate = np.stack((areaIsolate, areaIsolate, areaIsolate), 2)
    outputImage = np.multiply(areaImage, areaIsolate)
    outputImage = cv2.warpPerspective(outputImage, transMatrix, ((imageShape[1] + 30), (imageShape[0] + 30)))

    #Get the centre line of the board
    spaceMasked = cv2.cvtColor(outputImage, cv2.COLOR_BGR2HSV)
    spaceMasked = (spaceMasked[:, :, 0] < 35)
    centreLine  = cv2.HoughLinesP(spaceMasked.astype(np.uint8), 1, (np.pi/180), 10, minLineLength = (imageShape[1]/3))
    centreIndex = 0
    currentLeng = 0

    for i in range(len(centreLine)):
        pp = centreLine[i]
        pp = pp[0]
        ll = np.sqrt(((pp[0] - pp[2])**2 + (pp[1] - pp[3])**2))

        #Find which line has the maximum length
        if (ll > currentLeng) & (pp[1] > 100) & (pp[1] < 600):
            currentLeng = ll
            centreIndex = i

    linePoints  = centreLine[centreIndex][0]

    #Store the parameters for board isolation
    isolDefined = boardIsolator()
    isolDefined.imMask = areaIsolate
    isolDefined.height = imageShape[1] + 30
    isolDefined.width  = imageShape[0] + 30
    isolDefined.matrix = transMatrix
    isolDefined.centre = linePoints
    isolDefined.pixRat = CENTRE_LENGTH / currentLeng

    return isolDefined


###### CLASSES ######
#Class to store the methods of board isolation
class boardIsolator:
    def __init__(self):
        self.imMask = []
        self.height = []
        self.width  = []
        self.matrix = []
        self.centre = []
        self.pixRat = []

    def isolateBoard(self, frameImage):
        #Mask the Provided Image
        sectionImg = np.multiply(frameImage, self.imMask)

        #Transform the image with the stored matrix
        adjustedIm = cv2.warpPerspective(sectionImg, self.matrix, (self.height, self.width))

        return adjustedIm

    def isolatePlayer(self, frameImage):
        #Mask the Provided Image
        sectionImg = np.multiply(frameImage, self.imMask)

        #Transform the image with the stored matrix
        adjustedIm = cv2.warpPerspective(sectionImg, self.matrix, (self.height, self.width))

        #Remove all pixels which fall above the central line found
        lineHeight = np.floor(((self.centre[1] + self.centre[3]) / 2) - 1)
        lineHeight = lineHeight.astype(np.uint16)
        adjustedIm[range(0, lineHeight), :, :] = 0

        return adjustedIm

    def isolateDealer(self, frameImage):
        #Mask the Provided Image
        sectionImg = np.multiply(frameImage, self.imMask)

        #Transform the image with the stored matrix
        adjustedIm = cv2.warpPerspective(sectionImg, self.matrix, (self.height, self.width))

        #Remove all pixels which fall below the central line found
        lineHeight = np.floor(((self.centre[1] + self.centre[3]) / 2) - 1)
        lineHeight = lineHeight.astype(np.uint16)
        adjustedIm[lineHeight:, :, :] = 0

        return adjustedIm

    def isolateSections(self, frameImage):
        # Mask the Provided Image
        sectionImg = np.multiply(frameImage, self.imMask)

        # Transform the image with the stored matrix
        adjustedIm = cv2.warpPerspective(sectionImg, self.matrix, (self.height, self.width))

        #Get the height of the line
        lineHeight = np.floor(((self.centre[1] + self.centre[3]) / 2) - 1)
        lineHeight = lineHeight.astype(np.uint16)

        #Generate the dealer section
        dealerSect = np.zeros(np.shape(adjustedIm))
        dealerSect[:lineHeight, :, :] = adjustedIm[:lineHeight, :, :]

        #Generate the player section
        playerSect = np.zeros(np.shape(adjustedIm))
        playerSect[lineHeight:, :, :] = adjustedIm[lineHeight:, :, :]

        #Return the full adjusted board, player section and dealer section
        return adjustedIm.astype(np.uint8), playerSect.astype(np.uint8), dealerSect.astype(np.uint8)



