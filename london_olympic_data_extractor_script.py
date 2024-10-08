
import csv
import cv2
import pytesseract 
import pandas as pd
import re
import os
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def extract_text(roi):
   
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to the grayscale image
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]    
   
    # Get the text from the thresholded image using Tesseract OCR
    text = pytesseract.image_to_string(threshold)
    return text
    
# def extract_text_grayscale(roi):
   
    
#     threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]    
#     text = pytesseract.image_to_string(threshold)
    
#     return text

def extract_numbers(roi):
   
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(threshold, config='outputbase digits')    
    # text = pytesseract.image_to_string(threshold, lang='digits') 
    return text
    
def extract_number_grayscale(roi):
   
    
    threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]    
   
    # Get the text from the thresholded image using Tesseract OCR
    text = pytesseract.image_to_string(threshold, config='outputbase digits')
    # text = pytesseract.image_to_string(threshold, lang='digits')
    return text

def extract_frame(img_rgb,template):
    
    desired_width = 1625
    desired_height = 911
    img_rgb = cv2.resize(img_rgb, (desired_width,desired_height))
    
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #plt.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
    
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where( res >= threshold)
    #for pt in zip(*loc[::-1]):
     #cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2) 
    cropped_image=img_rgb.copy()
    
    for pt in zip(*loc[::-1]):
        x, y = pt
        w, h = template.shape[::-1]
        # Crop the matched region
        cropped_image = img_rgb[y:y+h, x:x+w]

        # Save the cropped image
        #cv.imwrite('cropped_image.jpg', cropped_image)

        # Draw a rectangle around the matched region
        #cv.rectangle(img_rgb, pt, (x + w, y + h), (0, 0, 255), 2)
    
    cropped_image=cv2.resize(cropped_image,(template.shape[1],template.shape[0]) )
    return cropped_image

def get_country(frame):
       
    roi = frame[0:60,40:105]
        
    #plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))   
    return extract_text(roi)   
    
def get_name(frame):
       
    roi = frame[0:60,210:600]
    #plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  
        
    return extract_text(roi)

def extract_somersaults(text):
    match1 = re.search(r'\d+% SOMERSAULTS', text,re.IGNORECASE)
    if match1:
        somersaults = float(match1.group(0).split('%')[0])+0.5
        return somersaults
    match2 = re.search(r'(\d+(\.\d+)?)\D*\b(somersaults?)\b', text, re.IGNORECASE)
    if match2:
        somersaults = match2.group(1)
        return somersaults

def extract_twists(text):
    match1 = re.search(r'\d+% TWISTS', text,re.IGNORECASE)
    if match1:
        twists = float(match1.group(0).split('%')[0])+0.5
        return twists
    match2 = re.search(r'(\d+(\.\d+)?)\D*\b(twists?)\b', text, re.IGNORECASE)
    if match2:
        twists = match2.group(1)
        return twists

def extract_allothers(text): 
    difficulty=""
    round=""
    divePosition=""
    diveGroup=""
    
    match = re.search(r'DIFFICULTY (\d+(\.\d+)?)', text, re.IGNORECASE)
    if match:
        difficulty = match.group(1)
    match = re.search(r'ROUND (\d+)', text, re.IGNORECASE)
    if match:
        round = match.group(1)        
    match = re.search(r'(\w+) POSITION', text,re.IGNORECASE)
    if match:
        divePosition = match.group(1) 
    match = re.search(r'(ARMSTAND|BACK|INWARD|FORWARD|REVERSE|TWISTER|ARMSTAND BACK|ARMSTAND FORWARD|BACK ARMSTAND|BACK FORWARD|BACK INWARD|BACK REVERSE|BACK TWISTER|FORWARD ARMSTAND|FORWARD BACK|FORWARD INWARD|FORWARD REVERSE|FORWARD TWISTER|INWARD BACK|INWARD FORWARD|INWARD REVERSE|INWARD TWISTER|REVERSE BACK|REVERSE FORWARD|REVERSE INWARD|REVERSE TWISTER|TWISTER BACK|TWISTER FORWARD|TWISTER INWARD|TWISTER REVERSE)', text, re.IGNORECASE)
    if match:
        diveGroup = match.group(1)
    
    return difficulty,round,divePosition,diveGroup

def extract_diveinfo(frame):
    country=""
    firstName=""
    secondName=""
    difficulty = ""
    round = ""
    diveGroup=""
    divePosition=""
    somersaults=""
    twists=""
    
    country =""
    name=""
    
    text = extract_text(frame)
    country=get_country(frame)
    name=get_name(frame)
    difficulty,round,divePosition,diveGroup=extract_allothers(text)
    somersaults = extract_somersaults(text)
    twists = extract_twists(text)  

    return round,country,name,difficulty,divePosition,somersaults,diveGroup,twists

def skel(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel

def get_final_score(frame,text):
    score=""
    match = re.search(r'SCORE (\d+)', text, re.IGNORECASE)
    if match:
        score = match.group(1)  
        return score
    
    roi = frame[50:100,910:1100]
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  
    return extract_numbers(roi)

def extract_penalty(text):       
    penalty = ""
    judgesScore=""
    match = re.search(r'PENALTY (\d+)', text, re.IGNORECASE)
    if match:
        penalty = match.group(1)    
    return penalty

def get_score(frame):
    #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    penalty="0"
    text=extract_text(frame)
    match = re.search(r'PENALTY (\d+)', text, re.IGNORECASE)
    if match:
        penalty = match.group(1)
    roi = frame[110:155,5:1150]
    # cv2.imshow("a",roi)
    # cv2.waitKey(0)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    roi_new=remove_strikes(roi) 
    text=extract_number_grayscale(roi_new)
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to floating-point values
    numbers = [float(number) for number in numbers]
    return numbers,penalty 
    
def remove_strikes(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    eroded_image = cv2.erode(image, kernel, iterations=5)
    result = image-eroded_image 
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    result =cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    result=cv2.erode(result,kernel,iterations=2)
    
    result = skel(result)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,3))
    result=cv2.dilate(result,kernel,iterations=2)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    eroded_image = cv2.erode(result, kernel, iterations=5)    
    # cv2.imshow("a",eroded_image)
    # cv2.waitKey(0)
    return result

def extract_clip(input_video_path, output_video_path, start_time, end_time):
    print(input_video_path)
    print(output_video_path)
    print(start_time)
    print(end_time)
    # Create a VideoFileClip object for the input video
    video_clip = VideoFileClip(input_video_path)

    # Extract the subclip between start_time and end_time
    clip = video_clip.subclip(start_time, end_time)

    # Write the clip to the output video file
    clip.write_videofile(output_video_path)

    # Close the VideoFileClip object
    video_clip.close()
# %%
def save_data(row,fileName):
    if not os.path.isfile(fileName+".csv"):
            # Create a new CSV file with headers if it doesn't exist
            df = pd.DataFrame(columns=['Sno', 'Match Name', 'Time', 'Name', 'Country', 'Position', 'Difficulty', 'Group', 'Somersaults', 'Twists', 'Penalty', 'Final_Score',])
    else:
            # Read the existing CSV file into a DataFrame
            df = pd.read_csv(fileName+".csv")
    row_df=pd.DataFrame([row],columns=df.columns)
    df=pd.concat([df,row_df], ignore_index=True)
    # df=df.append(row_df,ignore_index=True)
        # Save the updated DataFrame to the CSV file
    df.to_csv(fileName+'.csv', index=False)

def correct_score(score):
    pattern = r'(\d+) (\d+(\.\d+)?)?'
    match = re.search(pattern,score)
    if match:
        num2 = match.group(2)
        return num2
    else:
        return score

def extract_score_frame(img,template):
    cropped_frame=img[775:973,350:1606]
    cropped_frame = cv2.cvtColor(cropped_frame,cv2.COLOR_RGB2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(cropped_frame,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.5  # You can adjust this threshold according to your needs
    loc = np.where(res >= threshold) 
    # applying template matching anf if it matches then length is > 0 
    if len(loc[0]) > 0:
        # cv2.imshow("f",cropped_frame)
        # cv2.waitKey(0)
        return True
    else:
        return False

def get_diver_name(frame):
    cropped_name=frame[4:73,280:1100]
    # cv2.imshow("f",cropped_name)
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(cropped_name,lang="eng")
    text = text.replace("\n", "")
    print(text) 
    return text

def get_diver_country(frame):
    cropped_country=frame[7:75,30:262]
    text = pytesseract.image_to_string(cropped_country,lang="eng")
    text = text.replace("\n", "")
    text=text[0:3]
    print(text)
    return text

def get_diver_position(frame):
    cropped_position=frame[85:131,960:1210]
    # cv2.imshow("f",cropped_position)
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(cropped_position,lang="eng")
    text = text.replace("\n","")
    text=text.split()
    print(text)
    return text[0]

def get_diver_difficulty(frame):
    cropped_difficulty=frame[83:131,463:775]
    # cv2.imshow("f",cropped_difficulty)
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(cropped_difficulty,lang="eng")
    x=re.findall(r"\d+\.\d+",text)
    difficulty = float(x[0])
    return difficulty

def get_somersaults_and_twists(text):
    ans=0
    for i in range(0,len(text)):
        text[i] = text[i].strip().upper()
        if(('%' in text[i].strip())):
                somersaults = float(text[i].split('%')[0])+0.5
                ans=i
                break
        elif ((text[i].strip().isdigit())):
                somersaults = float(text[i])
                ans=i
                break
    twists=0
    for j in range(ans+1,len(text)):
        text[j] = text[j].strip().upper()
        if ((text[j].strip().isdigit())):
                twists = float(text[j])
                return somersaults,twists
    return somersaults,twists

def get_diveposition(text):
    for i in range(0,len(text)):
        text[i] = text[i].strip().upper()
        if('FORWARD'==text[i] or 'BACK'==text[i] or 'INWARD'==text[i] or 'ARMSTAND'==text[i] or 'REVERSE'==text[i]):
            return text[i]

def get_diver_group(frame):
    cropped_divegroup=frame[140:190,13:180]
    # cv2.imshow("f",cropped_divegroup)
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(cropped_divegroup,lang="eng")
    text=text.replace("\n","")
    print(text)
    return text

def get_other_details(frame):
    # cv2.imshow("f",frame)
    # cv2.waitKey(0)
    text=pytesseract.image_to_string(frame,lang="eng")
    text=text.split()
    diveposition=get_diveposition(text)
    somersaults,twists = get_somersaults_and_twists(text)
    return [diveposition,somersaults,twists]

def extract_info(frame):
    cropped_frame=frame[775:973,350:1606]
    # cv2.imshow("d",cropped_frame)
    # cv2.waitKey(0)
    cropped_frame=cv2.cvtColor(cropped_frame,cv2.COLOR_RGB2GRAY)
    threshold_img = cv2.threshold(cropped_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imshow("d",threshold_img)
    # cv2.waitKey(0)
    # cv2.imshow("F",threshold_img)
    # cv2.waitKey(0)
    name = get_diver_name(cropped_frame)
    country = get_diver_country(cropped_frame)
    position = get_diver_position(cropped_frame)
    difficulty = get_diver_difficulty(cropped_frame)
    # extract bottom stuff
    # plt.imshow(cropped_frame)
    bottom_frame = threshold_img[140:190,0:1220]
    # plt.imshow(bottom_frame)
    data = get_other_details(bottom_frame)
    info={"Name":name,"Country":country,"Position":position,"Difficulty":difficulty,"Group":data[0],"Somersaults":data[1],"Twists":data[2]}
    return info 

def extract_penalty_val(frame):
    cropped_penalty=frame[0:50,575:890]
    # cv2.imshow("a",cropped_penalty)
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(cropped_penalty,lang="eng")
    text=text.replace("\n","")
    match = re.search(r'PENALTY (\d+)', text, re.IGNORECASE)
    if match:
        penalty = float(match.group(1))    
    return penalty
def extract_final_score(frame):
    final_score=-1
    cropped_final_score=frame[0:44,980:1219]
    # cv2.imshow("f",cropped_final_score)
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(cropped_final_score,lang="eng")
    text=text.replace("\n","")
    match = re.search(r'(\d+\.\d+)',text,re.IGNORECASE)
    if match:
        final_score=float(match.group(1))
        return final_score
    return final_score
def extract_results(frame):
    cropped_frame=frame[857:965,350:1570]
    # cv2.imshow("d",cropped_frame)
    # cv2.waitKey(0)
    cropped_frame_gray=cv2.cvtColor(cropped_frame,cv2.COLOR_RGB2GRAY)
    threshold_img = cv2.threshold(cropped_frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imshow("d",threshold_img)
    # cv2.waitKey(0)
    # cv2.imshow("F",threshold_img)
    # cv2.waitKey(0)
    # plt.imshow(frame)
    # extract penalty
    penalty = extract_penalty_val(threshold_img)
    final_score=extract_final_score(threshold_img)
    return (penalty,final_score)
    # extract strikes with numbers
    # plt.imshow(frame)
    scores_frame = frame[913:961,355:1565]
    cv2.imwrite("scores.jpg",scores_frame)
    gray = cv2.cvtColor(scores_frame, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Remove horizontal lines (strikethrough)
    kernel = np.ones((3,3), np.uint8)
    horizontal_lines_removed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)
    plt.imshow(horizontal_lines_removed)
    # Perform OCR on the processed image
    text = pytesseract.image_to_string(horizontal_lines_removed)

    print(text)
    # plt.imshow(scores)
    # roi = cropped_frame[57:107,4:1217]
    # roi_new=remove_strikes(roi)
    # text=extract_number_grayscale(roi_new) 
    # def get_score(frame):
    # #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # penalty="0"
    # text=extract_text(frame)
    # match = re.search(r'PENALTY (\d+)', text, re.IGNORECASE)
    # if match:
    #     penalty = match.group(1)
    # roi = frame[110:155,5:1150]
    # cv2.imshow("a",roi)
    # cv2.waitKey(0)
    # plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    # roi_new=remove_strikes(roi) 
    # text=extract_number_grayscale(roi_new)
    # numbers = re.findall(r'\d+\.\d+|\d+', text)
    # # Convert the extracted numbers to floating-point values
    # numbers = [float(number) for number in numbers]
    # return numbers,penalty 
    # # name = get_diver_name(cropped_frame)
    # # country = get_diver_country(cropped_frame)
    # # position = get_diver_position(cropped_frame)
    # # difficulty = get_diver_difficulty(cropped_frame)
    # # # extract bottom stuff
    # # # plt.imshow(cropped_frame)
    # # bottom_frame = threshold_img[140:190,0:1220]
    # # # plt.imshow(bottom_frame)
    # # data = get_other_details(bottom_frame)
    # # info={"name":name,"country":country,"position":position,"difficulty":difficulty,"group":data[0],"somersaults":data[1],"twists":data[2]}
    # # return info 

    

# %%
def extract_from_video(filePath,videoName,template,templateRes):
    # desired_width = 1631
    # desired_height = 907
    cap = cv2.VideoCapture(filePath)
    fps=cap.get(cv2.CAP_PROP_FPS)
    minToSkip=4
    framesToSkip=fps*60*minToSkip
    framesToSkipEverySec=fps-1
    currentFrame=0
    currentFrameInSec=0
    diveStarted=False
    sno=1
    row=[]

    # Initialize a list to store the scores

    print("start")
    # Process each frame of the video
    # cap.set(cv2.CAP_PROP_POS_FRAMES,13*60*25)
    diveStarted=False
    ddd=[]
    row={}
    while True:
        # Read the next frame 
        ret, frame = cap.read()
        currentFrame+=1
        currentFrameInSec+=1
        if currentFrame<framesToSkip :
            if currentFrame%(fps*60)==0:
                print(currentFrame//(fps*60) ," min(s) Skipped")
            continue
        if currentFrame//(fps*60)<minToSkip:
            continue
        if currentFrameInSec<framesToSkipEverySec:
            continue
        # Break the loop if the video has ended
        if not ret:
            print("completed")
            break

        current_pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        total_seconds = int(current_pos // 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        timeInVideo =f"{minutes} minutes and {seconds} seconds"
        if ((not diveStarted) and (extract_score_frame(frame,template))):
            diveStarted=True
            #if template matched then perform OCR
            row['Sno']=sno
            row['Match Name']=videoName
            row['Time']=timeInVideo
            info = extract_info(frame)
            row.update(info)
            print(timeInVideo," dive started")
        elif(diveStarted and (extract_score_frame(frame,templateRes))):
            diveStarted=False
            results=extract_results(frame)
            res={'Penalty':results[0],'Final_Score':results[1]}
            row.update(res)
            print(timeInVideo," dive scores announced")
            # extract_clip(filePath,'ExtractedVideos/'+str(sno)+'_'+videoName+'.mp4',int(row['Time']),int(total_seconds))
            sno+=1
            #print(text)
            print(row)
            save_data(row,"scores_new")
            row={}
        '''
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use OCR to extract the text from the frame
        text = pytesseract.image_to_string(gray,lang="eng")   

        words = text.lower().split()  # split the text into a list of words
        filtered_words = [word for word in words if (word != 'olympic' and word != 'olympic' and word!='2012' and word!='201' and word!='london' and word!='london2012' and word!="lon" and word!="londo") ]  # use list comprehension to remove the word "hello"
        text = ' '.join(filtered_words)  # join the remaining words back into a string
        #if(text!=None):
        #    print(text)
        name=""
        round=""
        country=""
        difficulty=""
        divePosition=""
        somersaults=""
        diveGroup=""
        twists=""
    
        
        if not diveStarted and "difficulty" in text.lower() and "penalty" not in text.lower() and "position" in text.lower():
            diveStarted=True
            # the following fucntion matches the template and return the image
            cropped=extract_frame(frame,template)
            # plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            round,country,name,difficulty,divePosition,somersaults,diveGroup,twists = extract_diveinfo(cropped)
            row = {'Sno':sno,'Match name':videoName,"Time":total_seconds,'Name':name.strip(),'Country':country.strip(),'Difficulty':difficulty,'Dive Position':divePosition,'Somersaults':somersaults,
        'Dive Group':diveGroup,
        'Twists':twists }
            print(timeInVideo," dive started")
      
        if diveStarted and( ("difficulty" in text.lower() and "penalty" in text.lower()) | ("penalty" in text.lower()) ):
            print(timeInVideo," dive scores announced")
            cv2.imshow("a",frame)
            cv2.waitKey(0)
            cropped=extract_frame(frame,templateRes)
            cv2.imshow("A",cropped)
            cv2.waitKey(0)
            plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            croppedText=extract_text(cropped)
            score=get_score(cropped)
            finalScore=get_final_score(cropped,text).strip()
            finalScore=correct_score(finalScore)
            print(finalScore,score)
            row['Score'] = score
            row['Final Score'] = finalScore
            # extract_clip(filePath,'ExtractedVideos/'+str(sno)+'_'+videoName+'.mp4',int(row['Time']),int(total_seconds))
            diveStarted=False
            sno+=1
            #print(text)
            print(row)
            save_data(row,"scores")
            '''
        currentFrameInSec=0    
    cap.release()

# %%
templateLondon = cv2.imread('HelperImages/template_london_2.png', cv2.IMREAD_GRAYSCALE)
templateResLondon = cv2.imread('HelperImages/template_res_london_2.png', cv2.IMREAD_GRAYSCALE)
# extract_from_video("DownloadedVideos/Diving - Mens 3m Springboard - Final  London 2012 Olympic Games.mp4","Mens_3m_Springboard_-_Final_London_2012",templateLondon,templateResLondon)
extract_from_video("E:/BTP/DownloadedVideos/Diving - Mens 3m Springboard - Final  London 2012 Olympic Games.mp4","Mens_3m_Springboard_-_Final_London_2012",templateLondon,templateResLondon)


