


#Import in the relevant packages
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import imutils
from PIL import ImageDraw
from PIL import ImageFont
import argparse
import os
from operator import itemgetter, attrgetter


#This function receives an image and from this identifies all of the relevant playing cards, classifies each of the
#playing cards into a specific rank and displays this rank on top of the live stream. It then returns an n x 4 array
#which contains the cards rank, centroid X position, centroid Y position and height
def card_identification(img):

    #Read in the template image which we utilise to feature match
    img_template = cv2.imread('template_3.png',0)
    img_2 = img_template.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    yellow_line = 250

    #Convert our image to grayscale and extract out the Canny edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,10,200)

    #Convert the image to binary, based on the binary values set in the 'threshold' function
    ret,thresh1 = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)

    #Create a kernel for manipulation, size 2x2
    kernel_1 = np.ones((2,2),np.uint8)

    #Open the image by utilising the kernel we created above
    opened_im = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel_1)

    #Create a second kernel which we will utilise in order to close the card rectangles
    kernel_2 = np.ones((13,13),np.uint8)

    #Perform a series of manipulations which lead to the closure of the card rectangles
    closed_im = cv2.morphologyEx(opened_im,cv2.MORPH_CLOSE,kernel_2)
    closed_im_2 = cv2.morphologyEx(closed_im,cv2.MORPH_CLOSE,kernel_2)
    closed_im_3 = cv2.morphologyEx(closed_im_2,cv2.MORPH_CLOSE,kernel_2)

       #Create another threshold value which we will utilise in order to cctually
    # generate the contours we utilise for manipulation
    ret,thresh2 = cv2.threshold(closed_im_3,127,255,0)
    im2,contours,hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Determine the number of contours which were found, as we will subsequently loop through these contours
    length_C = len(contours)

    #Initialise a range of variables which we will be utilising to store and extract information
    height_mulp = 0
    cX = []
    cY = []
    cXabove = []
    cYabove = []
    xmin = []
    ymin = []
    height_arrabove = []
    height_arr = []
    card_pxl = np.full((50,50,3,length_C),0)
    test_img = np.full((30,20,3,length_C),0)
    card_val = []
    card_value = []
    card_valueabove = []
    lst_intensities = []
    cnt_val = 0;
    img_data = []
    final_array = []
    top_cards = []
    above_line = 0
    broken = 0
    side_ratio = 0

    #Loop through based on the number of contours which we have extracted from our image
    for i in range(0,length_C):
       #For this iteration of the loop store the relevant contour information in cnt
       cnt = contours[i]
       area_2 = cv2.contourArea(cnt)
       #If the area of the current contours is between 5000 and 20,000 then we can proceed
       if cv2.contourArea(cnt) > 9000 and cv2.contourArea(cnt) < 20000:
           #Extract out they key information about the contour in terms of location, height and width
          (x,y,w,h) = cv2.boundingRect(cnt)
          side_ratio = int(100*w/h)

          #Determine the rectangle which defines the contour we have identified, this will also
          # return the angle of rotation of the rectangle
          rect = cv2.minAreaRect(cnt)

          #Generate the moment for the contour which is required to determine the X/Y positions
          M = cv2.moments(cnt)

          broken = 0

          #Extract out the card portion of the image by taking the top left corner and extending out by the height and width of the card
          test_img = img[y:y+h,x:x+w,:]



          #If the rotation of the card is around -90 degrees then we need to correct this, otherwise the card will be rotated
          #in the wrong direction. Otherwise we just perform the rotation based on the value which we extracted before.
          if rect[2] < -75:
             test_img = imutils.rotate(test_img,rect[2]+90)
          else:
             test_img = imutils.rotate(test_img,rect[2])

          height_mulp = 135/h

          test_img_col = cv2.resize(test_img, (0,0), fx = height_mulp, fy = height_mulp)

          test_img_col = test_img_col[0:25,0:20,:]

          #Create a height factor which will zoom when we extract the card we are interested based on how small the card is
          #compared to the expected value of 135 pixels


          #Resize the image based on the above principles, whereby if the card is smaller than we anticipate then we
          #should be increasing the size so when we do template matching it is correct
          #test_img_col = cv2.resize(test_img, (0,0), fx = height_mulp, fy = height_mulp)

          #Convert the image to grayscale and then binary
          test_img = cv2.cvtColor(test_img_col, cv2.COLOR_BGR2GRAY)
          ret,test_img = cv2.threshold(test_img,170,255,cv2.THRESH_BINARY)
          a = 0

          #Check if the top left pixel in the extracted portion is black, if it is then go through and crop the image
          while test_img[0,0] < 100:


              test_img = test_img[1:len(test_img[:,0]),1:len(test_img[0,:])]

              #If we cropped down the image and never reached a white pixel then we have not found an appropriate white
              #portion has been identifed.
              if len(test_img[0,:]) < 1 or len(test_img[:,0]) < 1:
                  broken = 1

                  break

          #If we have indeed found a white section then continue with processing the contour
          if broken == 0:

              #Extract out the top left 25x25 pixels
              test_img = test_img[0:25,0:15]


              #Set the template which we need to match to the test_img we just generated
              template = test_img

              #Extract out the width and height of the template shape and the actual template data as we need to make sure
              #the template can fit in the data image
              w_tmp, h_tmp = template.shape[::-1]
              w_img, h_img = img_template.shape[::-1]


              #Set the method which we plan to use for template matching
              method = eval('cv2.TM_CCOEFF')

              # Ensure that the sizes of the image we are trying to
              if h_img <= h_tmp or h > 300 or w > 250:
                  cv2.waitKey(1)
              else:


                #Plot a circle at the centre of the card
                cv2.circle(img,(int(x+w/2),int(y+h/2)),5,(200,0,0),2)

                #Perform the template matching process
                res = cv2.matchTemplate(img_template,template,method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                #Write the card image to file (just for de-bugging purposes currently)
                cv2.imwrite('image_extract' + str(i) + '.jpg',test_img)

                #Extract out the location of the top left of the matching component in the template data image
                top_left = max_loc
                bottom_right = (top_left[0] + w_tmp, top_left[1] + h_tmp)

                #Determine the sum of the red pixels in the image, this is used to identify the
                sum_red = int(np.mean(test_img_col[15:20,10:15,0])*100/(np.mean(test_img_col[0:14,0:9,1]+1)))

                #Determine the y centroid location of the line is above or below the yellow line dividing the board
                #in tow
                if (int(M["m01"] / M["m00"])) < yellow_line:

                    #Set the above_line variable to 1 so that we know this is either the player's card, dealer's card
                    #or the deck
                    above_line = 1
                    cXabove.append(int(M["m10"] / M["m00"]))
                    cYabove.append(int(M["m01"] / M["m00"]))
                    height_arrabove.append(int(h))
                else:
                    #Set the above_line variable to 0 so we know that it is one of the permanent cards down the bottom
                    above_line = 0

                    #Store the X, Y centroid position of this card and the height of the card
                    cX.append(int(M["m10"] / M["m00"]))
                    cY.append(int(M["m01"] / M["m00"]))
                    height_arr.append(int(h))

                #Apply a series of logic statements to determine what value the card should have. We will
                #classify cards based on the pixel location which is returned from the template matching processing.
                #These pixels values are known and have been extracted from the template image. When a card is identified
                #we store this value in 'card_val' for displaying it to the board and 'card_value' so we can return it
                #from this function

                #If the card is wider than it is high then it must be the deck
                if side_ratio >100 and above_line == 1:
                    card_val = 'Deck'
                    if above_line == 1:
                        card_valueabove.append(0)
                    else:
                        card_value.append(0)
                elif max_val < 50000:
                    card_val = "No Match"
                    if above_line == 1:
                        card_valueabove.append(20)
                    else:
                        card_value.append(20)
                elif top_left[0] < 35:
                    card_val = '2'
                    if above_line == 1:
                        card_valueabove.append(2)
                    else:
                        card_value.append(2)
                elif top_left[0] < 88:
                    card_val = '3'
                    if above_line == 1:
                        card_valueabove.append(3)
                    else:
                        card_value.append(3)
                elif top_left[0] < 141:
                    card_val = '4'
                    if above_line == 1:
                        card_valueabove.append(4)
                    else:
                        card_value.append(4)
                elif top_left[0] < 194:
                    card_val = '5'
                    if above_line == 1:
                        card_valueabove.append(5)
                    else:
                        card_value.append(5)
                elif top_left[0] < 247:
                    card_val = '6'
                    if above_line == 1:
                        card_valueabove.append(6)
                    else:
                        card_value.append(6)
                elif top_left[0] < 300:
                    card_val = '7'
                    if above_line == 1:
                        card_valueabove.append(7)
                    else:
                        card_value.append(7)
                elif top_left[0] < 353:
                    card_val = '8'
                    if above_line == 1:
                        card_valueabove.append(8)
                    else:
                        card_value.append(8)
                elif top_left[0] < 406:
                    card_val = '9'
                    if above_line == 1:
                        card_valueabove.append(9)
                    else:
                        card_value.append(9)
                elif top_left[0] < 459:
                    card_val = '10'
                    if above_line == 1:
                        card_valueabove.append(10)
                    else:
                        card_value.append(10)
                elif top_left[0] < 512:
                    card_val = 'A'
                    if above_line == 1:
                        card_valueabove.append(1)
                    else:
                        card_value.append(1)
                elif top_left[0] < 565:
                    card_val = 'J'
                    if above_line == 1:
                        card_valueabove.append(11)
                    else:
                        card_value.append(11)
                elif top_left[0] < 618:
                    card_val = 'K'
                    if above_line == 1:
                        card_valueabove.append(13)
                    else:
                        card_value.append(13)
                elif top_left[0] > 617:
                    card_val = 'Q'
                    if above_line == 1:
                        card_valueabove.append(12)
                    else:
                        card_value.append(12)
                else:
                    card_val = '0'
                    card_value.append(0)


                #Print the text to the screen to display the card value
                cv2.putText(img,card_val,(x+25,y-5), font, 1,(255,181,0),2,cv2.LINE_AA)

                cnt_val = cnt_val + 1


    #Append the relevant arrays together so that we can export them from this function
    final_array.append(card_value)
    final_array.append(cX)
    final_array.append(cY)
    final_array.append(height_arr)
    top_cards.append(card_valueabove)
    top_cards.append(cXabove)
    top_cards.append(cYabove)
    top_cards.append(height_arrabove)


    #Transpose the array so that when it is returned from this function it is ordered based on the those cards found
    #in the bottom row below the horizontal line, this will assist with the location of the betting chips
    transposed = np.transpose(final_array)
    if len(transposed) <1:
        return [[15,15,15,15],[15,15,15,15],[15,15,15,15],[15,15,15,15]]
    #print('transposed')
    #print(transposed)
    f = transposed[transposed[:,2].argsort()]


    #Transpose the cards found above the horizontal line on the playing board so that we can order them in horizontal
    #value, this way we know which is the player card, which is the deck and which is the dealer's card
    transposed = np.transpose(top_cards)
    if len(card_valueabove) == 0:
        return f
    else:
        top_f = transposed[transposed[:,1].argsort()]
        final_return = np.concatenate((f,top_f))

        return final_return




