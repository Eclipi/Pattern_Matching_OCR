import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

templateDirectoryLeft = 'D:/Python/Pattern_Matching/templates1/*.png'
templateDirectoryRight = 'D:/Python/Pattern_Matching/templates2/*.png'
testImageDirectory = 'D:/Python/Pattern_Matching/test_images/2_5.png'

def recongizeNumber():
    max_loc_data = []
    max_val_data = []
    max_loc_data2 = []
    max_val_data2 = []

    #empty list to store template images
    template_data1=[]
    template_data2=[]
    #make a list of all template images from a directory
    files1= glob.glob(templateDirectoryLeft)
    files2= glob.glob(templateDirectoryRight)

    for myfile in files1:
        image = cv2.imread(myfile, 0)
        template_data1.append(image)

    for myfile in files2:
        image = cv2.imread(myfile, 0)
        template_data2.append(image)

    test_image=cv2.imread(testImageDirectory)
    test_image= cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    leftImage = test_image[0:int(len(test_image[1])), 0:int(len(test_image[0])/2)]
    rightImage = test_image[0:int(len(test_image[1])), int(len(test_image[0])/2):]

    #loop for matching leftImage
    for tmp in template_data1:
        (tH, tW) = tmp.shape[:2]
        # cv2.imshow("Template", tmp)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        result = cv2.matchTemplate(leftImage, tmp, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        max_loc_data.append(max_loc)
        max_val_data.append(max_val)

    first_num = str(max_val_data.index(max(max_val_data)))
    top_left = max_loc_data[max_val_data.index(max(max_val_data))]
    bottom_right = (top_left[0] + tW, top_left[1] + tH)
    cv2.rectangle(test_image,top_left, bottom_right,255, 2)


    for tmp in template_data2:
        (tH, tW) = tmp.shape[:2]
        # cv2.imshow("Template", tmp)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        result = cv2.matchTemplate(rightImage, tmp, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        max_loc_data2.append(max_loc)
        max_val_data2.append(max_val)

    if (max_val_data2.index(max(max_val_data2))) == 0:
        jum = '.0'
    else: jum = '.5'

    top_left = max_loc_data2[max_val_data2.index(max(max_val_data2))]
    bottom_right = (top_left[0] + tW, top_left[1] + tH)
    cv2.rectangle(rightImage,top_left, bottom_right,255, 2)


    print('The number is : ', first_num + jum)


    # cv2.imshow('Result',test_image)
    # cv2.waitKey(0)

    plt.subplot(122),plt.imshow(test_image,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(test_image)

    plt.show()

if __name__ == "__main__":
    recongizeNumber()