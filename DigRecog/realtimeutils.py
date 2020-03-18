def shadow_remover(cv2 , np , im):   
    grayscale_plane = cv2.split(im)[0]
    dilated_img = cv2.dilate(grayscale_plane, np.ones((3, 3), np.uint16))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(grayscale_plane, bg_img)
    normalized_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return normalized_img
    
def process(cv2 , np , im , model):   

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    #remove shadows
    im_gray = shadow_remover(cv2 , np , im_gray)
    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("thresh",im_th)
    # Find contours in the image
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    try:
        if rects is not None:
            img_rows = img_cols = 28
            for rect in rects:
                # Draw the rectangles
                if rect[2]>=5 and rect[2]<=30 and rect[3] >=30 and rect[3] <=50:
                    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
                    # Make the rectangular region around the digit
                    roi = im_th[rect[1]-10:rect[1]+rect[3]+10,rect[0]-10:rect[0]+rect[2]+10]
                    # Resize the image
                    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                    roi = cv2.dilate(roi, (3, 3))
                    
                    # reshape the roi
                    roi = roi.reshape(1, img_rows, img_cols, 1)
                    #use nn to predict numbers
                    prediction = model.predict(roi)
                    cv2.putText(im,str(prediction.argmax()) , (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    except Exception as e:
        print(e)       
    return im
