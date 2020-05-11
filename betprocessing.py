import numpy as np

#This function takes in the player card, dealer card, current bets and the Player Balance in order to determine
#if the player won any money in the new cards
def betarithmetic(player_current,dealer_current,current_bets,PlayerBalance):

        final_output = []

        #If either the dealer card or the player card have been identifed as a 'No Match' then we don't want to process
        #any of the bets
        if dealer_current != 20 and player_current != 20:

                #If the dealer and player card match then we should just half the player's current bet on the card
                #which was shown, with no change to the player balance
                if player_current == dealer_current:
                        current_bets[player_current - 1] = round(current_bets[player_current -1 ]/2)
                else:

                        #Otherwise, if the new player card already had bets on it then we must update the Player Balance
                        #to represent this as well as re-setting the balance on that card to zero
                        bet_on_plcard = current_bets[player_current-1]
                        PlayerBalance = PlayerBalance + 2*bet_on_plcard
                        current_bets[player_current-1] = 0

                        #No matter if there were bets or not then we set the bets on the card the dealer was dealt
                        #back to zero
                        current_bets[dealer_current-1] = 0

                #Append the two matrices together
                final_output.append(current_bets)
                final_output.append(PlayerBalance)

        #Return the matrices together
        return final_output

#This function takes in the hand position, bet value, card position, current bets, player balance and the playing card
#values to update the current bets on each card so the augmented reality playing chips are correct
def betprocessing(hX, hY, BetValue, cX, cY, current_bets, PlayerBalance, cardValue):
        final_output = []

        #For each of the cards which have been identified on the playing board we must go through and see if the
        #location of the hand is within 50 pixels of this location
        for i in range(0,len(cX) - 1):

                #If the hand is located within 50 pixels of the centroid of the playing card in both an X and Y location
                #then we assume the player wants to bet on this card. The player balance must also be above zero in
                #order for the player to make a bet on a card
                if abs(hX - cX[i]) < 50 and abs(hY - cY[i]) < 50 and PlayerBalance > 0:

                        #Update the current bets array to incorporate the new bet value we are making ont he card
                        current_bets[cardValue[i]-1] = current_bets[cardValue[i]-1] + BetValue
                        PlayerBalance = PlayerBalance - BetValue


        #Append the two matrices together
        final_output.append(current_bets)
        final_output.append(PlayerBalance)

        #Return the matrices together
        return final_output











