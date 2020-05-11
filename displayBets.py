# this is meant to be a comment
import numpy as np
import cv2

def displayBets(centroidsX, centroidsY, heights, bets, cardNumbers, playerBalance,image):
    # Final 2 elements of centroids arrays are the centroids of player and
    # dealer
    #print(cardNumbers)
    #cardNumbers = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
    length = len(centroidsX)-3
    centroidsY_top = int(round(np.amin(centroidsY[0:5])))
    #print(centroidsY_bottom)
    centroidsY_bottom = int(round(np.amin(centroidsY[7:12])))
    #print(centroidsY_top)

    # Bet amounts
    #bets[cardNumber-1]
    for j in range(0,length+3):
        if cardNumbers[j] == 20:
            cardNumbers[j]=0




    for i in range(0,13):# and cY in centroidsY:
        #print('Card12')
        #print(cardNumbers[i])
        if i > 6:
            cv2.circle(image,(centroidsX[i],centroidsY_bottom+((heights[i]))-60),20,(0,0,255),-1)
            #cv2.putText(image,str(cardNumbers[i]),(centroidsX[i]-70,centroidsY_bottom-40),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
            #print count
            if bets[cardNumbers[i]-1] > 9:
                cv2.putText(image,"$"+str(bets[cardNumbers[i]-1]),(centroidsX[i]-15,centroidsY_bottom+(heights[i])-55),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),1)
            else:
                cv2.putText(image,"$"+str(bets[cardNumbers[i]-1]),(centroidsX[i]-10,centroidsY_bottom+(heights[i])-55),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),1)
        elif i < 6:
            cv2.circle(image,(centroidsX[i],centroidsY_top+((heights[i]))-60),20,(0,0,255),-1)
            #print(i)
            #cv2.putText(image,str(cardNumbers[i]),(centroidsX[i]-80,centroidsY_top-40),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
            #print count
            if bets[cardNumbers[i]-1] > 9:
                cv2.putText(image,"$"+str(bets[cardNumbers[i]-1]),(centroidsX[i]-15,centroidsY_top+(heights[i])-55),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),1)
            else:
                cv2.putText(image,"$"+str(bets[cardNumbers[i]-1]),(centroidsX[i]-10,centroidsY_top+(heights[i])-55),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),1)
        else:
            cv2.circle(image,(centroidsX[i],centroidsY[i]+((heights[i]))-60),20,(0,0,255),-1)
            #cv2.putText(image,str(cardNumbers[i]),(centroidsX[i]-80,centroidsY[i]-40),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
            #print count
            if bets[cardNumbers[i]-1] > 9:
                cv2.putText(image,"$"+str(bets[cardNumbers[i]-1]),(centroidsX[i]-15,centroidsY[i]+(heights[i])-55),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),1)
            else:
                cv2.putText(image,"$"+str(bets[cardNumbers[i]-1]),(centroidsX[i]-10,centroidsY[i]+(heights[i])-55),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),1)

        #count = count+1
    # Display the card number next to the dealt Player and Dealer cards in the
    # top section
    #cv2.putText(image,"Player: "+str(cardNumbers[length+2]),(centroidsX[length+2]-100,centroidsY[length+2]-50),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),1)
    #cv2.putText(image,"Deck", (centroidsX[length+1]-100,centroidsY[length+1]-50),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),1)
    #cv2.putText(image,"Dealer: "+str(cardNumbers[length]), (centroidsX[length]-100,centroidsY[length]-50),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,0),1)
    #print(centroidsX[length+1])
    # Variable containing the total amount of money the player has
    #player = 100

    # Display the player's balance in the top left
    im_width = int(image.shape[1])
    #im_width = 1280
    im_height = 720
    if playerBalance == 0:
        cv2.putText(image,"Insufficient Funds",(im_width-350,230),cv2.FONT_HERSHEY_COMPLEX,0.75,(255,0,0),2)
        cv2.putText(image,"Player: $ 0",(im_width-350,200),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,0),2)
    else:
        cv2.putText(image,"Player: $"+str(playerBalance),(im_width-350,200),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,0),2)


    cv2.imshow("Image", image)


"""
def display_bets(centroidsX, centroidsY, height, bets, betValue, cardNumber):
    length = len(centroidsX)
    for cX in centroidsX and cY in centroidsY:
        print cX
"""
