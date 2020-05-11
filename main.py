#main.py
#This function initialises all required aspects of the vision system and then continually processes the live video
#stream to identify hands, cards and bets. A GUI is then shown on top of the live stream.


#Imports
import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt
from card_read import *
from operator import itemgetter, attrgetter
from displayBets import *
from betprocessing import *
from boardIsolation import *
import handDetection as hdt
import time

#Start the player on $30
PlayerBalance = 30

#Configure the parameters required for the capture of screenshots from the live video feed
cap = cv2.VideoCapture(1)
cap.open(0,cv2.CAP_DSHOW);
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#Initialise all of the initial variables
yellow_threshold = 300
a = 0
player_current = 0
dealer_current = 0
player_new = 0
dealer_new = 0
cX = []
cY = []
PlayerBalance_old = 100
this_hand = []
prev_hand = []
bet_place = 0
current_bets = [0,0,0,0,0,0,0,0,0,0,0,0,0]
new_bets = [0,0,0,0,0,0,0,0,0,0,0,0,0]

#Extract an initial screen grab so that we are able to isolate out the playing board
ret,img_read = cap.read()
cv2.imshow('Test',img_read)
cv2.waitKey(0)
image_distort_info = defineIsolation(img_read)

#This is an infinite loop which allows us to continuously capture new images from the live stream
while True:
    a = a + 1
    #Acquire Latest Image
    # Capture from the specified camera

    #Initialise these variables after 20 cycles to make sure that if cards are present they are processed
    if a == 20:
        HandPresent = 1
        BetPresent = 1
    else:
        HandPresent = 0
        BetPresent = 0
    # Loop

    # Read the frame-by-frame feed of the camera
    ret,img_read = cap.read()

    #Apply the isolation section to the image which has been exracted
    img_read, play_area, deal_area = image_distort_info.isolateSections(img_read)

    # Display livestream
    cv2.imshow('Frame', img_read)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    #Call the hand function and identify the location of the hand
    found_hand, this_hand = hdt.handIdentifier(play_area)
    deal_hand, dud = hdt.handIdentifier(deal_area)

    #Determine level of movement that the hand has undergone, and if a hand is located then we need to update the
    #lastest hand detection, and determine the centroid location of the tip of the hand. Then display this on the board
    #as a purple dot
    if found_hand == 1:
        hand_move = this_hand.handMovement(prev_hand)
        prev_hand = this_hand.handArea
        cv2.circle(img_read,(this_hand.centroid[0], this_hand.centroid[1]),5,(200, 0, 100),-1)
    else:
        prev_hand = []
        hand_move = 1


    #If we have located a hand on the board on the board, and playing cards have been detected then we can go throough
    #and update the bet processing to ensure the latest bets have been processed correctly
    if (found_hand == 1) and (bet_place == 0) and (len(cX) > 0) and (len(cY) > 0) and (hand_move < 0.1):
        bet_process_return = betprocessing(this_hand.centroid[0], this_hand.centroid[1], this_hand.gesture, cX, cY, current_bets, PlayerBalance, cardValue)
        if PlayerBalance < 0:

            #If the Player Balance is less than zero dollars then no bets would have been placed, we can update the GUI
            #and return to the start of the while loop
            if len(cX) > 12:
                displayBets(cX,cY,card_height,current_bets,cardValue,PlayerBalance,img_read)
            continue
        #If the player does have money left to bet with then we will go through and update the bets on each of the cards
        #update the player balance and then display these to the screen
        else:
            current_bets = bet_process_return[len(bet_process_return) - 2]
            PlayerBalance = bet_process_return[len(bet_process_return) - 1]
            PlayerBalance_old = PlayerBalance
            bet_place = 1
            if len(cX) > 12:
                displayBets(cX,cY,card_height,current_bets,cardValue,PlayerBalance,img_read)
            continue


    #If we have not found a hand, but a bet is present then we can go through and update the GUI but not try to identify
    #the cards present
    elif ((found_hand == 0) | (hand_move > 0.1)) and bet_place == 1:
        bet_place = 0
        if len(cX) > 12:
            displayBets(cX,cY,card_height,current_bets,cardValue,PlayerBalance,img_read)
        continue

    #If none of the previous if statements have been hit, but a hand or bet is present then we want to update the GUI
    #but we do not want to go through and undergo card identification again
    elif ((found_hand != 0) | (deal_hand != 0)) & (len(cX) > 5):
        if len(cX) > 12:
            displayBets(cX,cY,card_height,current_bets,cardValue,PlayerBalance,img_read)
        continue


    #Card Identification Function(
        #Inputs: Current Image
        #Function: Identifies the location of each of the cards in the image
        #Outputs: [cardValue,cX,cY,card_height]
    #)

    #Extract out the cards which were identified as part of the identification algorithm
    card_values_return = card_identification(img_read)
    test_val = card_values_return[0][0]

    #Check to make sure the first card returned did not have a value of 15, if it did then there has been an error
    #in identifying the cards
    if card_values_return[0][0] != 15:

        #Extract out the card centroid location, card rank and card height from the array that was returned
        cX = card_values_return[:,1]
        cY = card_values_return[:,2]
        cardValue = card_values_return[:,0]
        card_height = card_values_return[:,3]

        #Extract out the rank of the player card
        if card_values_return[len(card_values_return) - 1][2] < 300:
            player_new = card_values_return[len(card_values_return) - 1][0]

        #Extract out the rank of the dealer card
        if card_values_return[len(card_values_return) - 3][2] < 300:
            dealer_new = card_values_return[len(card_values_return) -3][0]
            print(len(card_values_return) -3)

        #Do an if statement to determine the card values have changed for either the player or the dealer
        #If the card values have changed then we will need to process the bets
        if player_new == player_current:
            Player_Change = 0
        else:
            Player_Change = 1

        if dealer_new == dealer_current:
            Dealer_Change = 0
        else:
            Dealer_Change = 1

        #If a change has been made to both cards then do nothing, however if both have changed then we need to go
        #through and process the bets
        if (Player_Change + Dealer_Change) < 2:
            cv2.waitKey(1)
        elif player_new == 20 or dealer_current == 20:
            cv2.waitKey(1)
        elif Player_Change == 1 and Dealer_Change == 1:

            #Wait briefly to make sure that the cards have stabilised and we don't mis-classify the cards
            cv2.waitKey(150)

            #Go through the card identification process again to make sure we have the most update to version of the
            #cards
            card_values_return = card_identification(img_read)
            cX = card_values_return[:,1]
            cY = card_values_return[:,2]
            cardValue = card_values_return[:,0]
            card_height = card_values_return[:,3]
            player_new = card_values_return[len(card_values_return) - 1][0]
            dealer_new = card_values_return[len(card_values_return) -3][0]

            #Send through the new values of the player and dealer card so we can perform the relevant bet arithmetic
            bet_arth_return = betarithmetic(player_new,dealer_new,current_bets,PlayerBalance)
            if len(bet_arth_return) == 0:
                continue

            #Update the current value of the player and dealer cards to make sure that they are stored correctly for
            #next time
            else:
                player_current = player_new
                dealer_current = dealer_new
                current_bets = bet_arth_return[len(bet_arth_return) - 2]
                PlayerBalance = bet_arth_return[len(bet_arth_return) - 1]

                #The Player's Balance only would have changed if the player won, so if so display this fact onto
                #the GUI interface
                if PlayerBalance != PlayerBalance_old:
                    cv2.putText(img_read,"Player Win",(800,240),cv2.FONT_HERSHEY_DUPLEX,0.75,(0,255,0),2)
                    PlayerBalance_old = PlayerBalance
                    cv2.waitKey(100)

        #If we have found at least 12 cards then progress to display the bets
        if len(cX) > 12:

            displayBets(cX,cY,card_height,current_bets,cardValue,PlayerBalance,img_read)

# When done, release the capture and close windows
cap.release()
#cv2.waitKey(0)
cv2.destroyAllWindows()




