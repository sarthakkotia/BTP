# %%
import csv
import cv2
import pytesseract
from pytesseract import image_to_string
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

    
def extract_numbers(roi):
   
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]    
    text = pytesseract.image_to_string(threshold, lang='digits')
    
    return text

def extract_number_grayscale(roi):
   
    
    threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]    
   
    # Get the text from the thresholded image using Tesseract OCR
    text = pytesseract.image_to_string(threshold, lang='digits')
    return text

def get_country(frame):
       
    roi = frame[0:70,0:82]
        
    #plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))   
    return extract_text(roi)   
    
def get_name(frame):
       
    roi = frame[10:70,155:800]
        
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
    result=cv2.dilate(result,kernel,iterations=3)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    eroded_image = cv2.erode(result, kernel, iterations=12)    
    
    return result

def extract_final_score(text):
    finalScore=""
    pattern1 = r"round \d+ (\d+\.\d+)"        
    match1 = re.search(pattern1, text,re.IGNORECASE)
    pattern2=r'\d{2}\.\d{2}'
    match2 = re.search(pattern2, text,re.IGNORECASE)
    if match1:
        finalScore = match1.group(1)       
    elif match2:
        finalScore = match2.group(0)               
    return finalScore

def extract_penalty(text):       
    penalty = ""
    judgesScore=""
    match = re.search(r'PENALTY (\d+)', text, re.IGNORECASE)
    if match:
        penalty = match.group(1)    
    return penalty

def get_score(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    penalty=""
    text=extract_text(frame)
    match = re.search(r'PENALTY (\d+)', text, re.IGNORECASE)
    if match:
        penalty = match.group(1)
        
    roi = frame[70:120,0:650]
    #plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    roi_new=remove_strikes(roi) 
    text=extract_number_grayscale(roi_new)
    numbers = re.findall(r'\d+\.\d+|\d+', text)

    # Convert the extracted numbers to floating-point values
    numbers = [float(number) for number in numbers]
    return numbers,penalty 
    
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

def extract_frame(img_rgb,template):
  
    desired_width = 1631
    desired_height = 907
    img_rgb = cv2.resize(img_rgb, (desired_width,desired_height))
    
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #plt.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))
    
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.3
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
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
    
# %%
def save_data(row,fileName):
    #     # Check if the CSV file exists
    # if not os.path.isfile(fileName+".csv"):
    #     # Create a new CSV file with headers if it doesn't exist
    #     df = pd.DataFrame(columns=['Sno','Match name','Name','Country','Difficulty','Dive Position','Somersaults', 'Dive Group', 'Twists','Score','Final Score'])
    # else:
    #     # Read the existing CSV file into a DataFrame
    #     df = pd.read_csv(fileName+".csv")
    # df = df.append(row, ignore_index=True)
    # # Save the updated DataFrame to the CSV file
    # df.to_csv(fileName+'.csv', index=False)
    if not os.path.isfile(fileName+".csv"):
            # Create a new CSV file with headers if it doesn't exist
            df = pd.DataFrame(columns=['Sno', 'Match Name', 'Time','Time_in_sec', 'Name', 'Country', 'Position', 'Difficulty', 'Group', 'Somersaults', 'Twists', 'Penalty', 'Final_Score',])
    else:
            # Read the existing CSV file into a DataFrame
            df = pd.read_csv(fileName+".csv")
    row_df=pd.DataFrame([row],columns=df.columns)
    df=pd.concat([df,row_df], ignore_index=True)
    # df=df.append(row_df,ignore_index=True)
        # Save the updated DataFrame to the CSV file
    df.to_csv(fileName+'.csv', index=False)

def extract_score_frame(img,template):
    cropped_frame=img[835:1024,330:1700]
    cropped_frame = cv2.cvtColor(cropped_frame,cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(cropped_frame,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.2  # You can adjust this threshold according to your needs
    loc = np.where(res >= threshold) 
    # applying template matching anf if it matches then length is > 0 
    if len(loc[0]) > 0:
        cv2.imshow("f",cropped_frame)
        cv2.waitKey(0)
        return True
    else:
        return False

def get_diver_name(frame):
    cropped_name=frame[4:73,280:1100]
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
    text = pytesseract.image_to_string(cropped_position,lang="eng")
    text = text.replace("\n","")
    text=text.split()
    print(text)
    return text[0]

def get_diver_difficulty(frame):
    cropped_difficulty=frame[83:131,463:775]
    text = pytesseract.image_to_string(cropped_difficulty,lang="eng")
    x=re.findall(r"\d+\.\d+",text)
    difficulty = float(x[0])
    return difficulty

def get_diveposition(text):
    for i in range(0,len(text)):
        text[i] = text[i].strip().upper()
        if('FORWARD'==text[i] or 'BACK'==text[i] or 'INWARD'==text[i] or 'ARMSTAND'==text[i] or 'REVERSE'==text[i]):
            return text[i]
        
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

def get_other_details(frame):
    text=pytesseract.image_to_string(frame,lang="eng")
    text=text.split()
    diveposition=get_diveposition(text)
    somersaults,twists = get_somersaults_and_twists(text)
    return [diveposition,somersaults,twists]

def extract_info(frame):
    cropped_frame=frame[775:973,350:1606]
    cropped_frame=cv2.cvtColor(cropped_frame,cv2.COLOR_RGB2GRAY)
    threshold_img = cv2.threshold(cropped_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
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

def extract_results(frame):
    cropped_frame=frame[857:965,350:1570]
    cropped_frame_gray=cv2.cvtColor(cropped_frame,cv2.COLOR_RGB2GRAY)
    threshold_img = cv2.threshold(cropped_frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imshow("F",threshold_img)
    # cv2.waitKey(0)
    # plt.imshow(frame)
    # extract penalty
    penalty = extract_penalty_val(threshold_img)
    final_score=extract_final_score(threshold_img)
    return (penalty,final_score)
# %%
def extract_from_video(filePath,videoName,template,templateRes):
    cap = cv2.VideoCapture(filePath)
    fps=cap.get(cv2.CAP_PROP_FPS)
    minToSkip=9
    framesToSkip=fps*60*minToSkip
    framesToSkipEverySec=fps-1
    currentFrame=0
    currentFrameInSec=0
    diveStarted=False
    sno=1
    row={}

    # Initialize a list to store the scores

    print("start")
    diveStarted=False
    row={}
    # Process each frame of the video
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
            cv2.imshow("f",frame)
            cv2.waitKey(0)
            diveStarted=True
            #if template matched then perform OCR
            row['Sno']=sno
            row['Match Name']=videoName
            row['Time']=timeInVideo
            row['Time_in_sec']=total_seconds
            info = extract_info(frame)
            row.update(info)
            print(timeInVideo," dive started")
        elif(diveStarted and (extract_score_frame(frame,templateRes))):
            diveStarted=False
            results=extract_results(frame)
            res={'Penalty':results[0],'Final_Score':results[1]}
            row.update(res)
            print(timeInVideo," dive scores announced")
            extract_clip(filePath,'ExtractedNewVideos/'+str(sno)+'_'+videoName+'.mp4',int(row['Time_in_sec']),int(total_seconds))
            sno+=1
            print(row)
            save_data(row,"scores_new")
            row={}
        currentFrameInSec=0    
    cap.release()

    #     # Convert the frame to grayscale
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     # Use OCR to extract the text from the frame
    #     text = pytesseract.image_to_string(gray,lang="eng")

    #     words = text.lower().split()  # split the text into a list of words
    #     filtered_words = [word for word in words if (word != 'tokyo' and word!='2020' and word!='202' and word!='toky' and word!='tokyo2020' and word!="0ky0" and word!="toky0") ]  # use list comprehension to remove the word "hello"
    #     text = ' '.join(filtered_words)  # join the remaining words back into a string

    #     name=""
    #     round=""
    #     country=""
    #     difficulty=""
    #     divePosition=""
    #     somersaults=""
    #     diveGroup=""
    #     twists=""

    #     if not diveStarted and "round" in text.lower() and "difficulty" in text.lower() and "penalty" not in text.lower() and "position" in text.lower():
    #         cv2.imshow("f",frame)
    #         cv2.waitKey(0)
    #         plt.imshow(frame)
    #         diveStarted=True
    #         b = extract_score_frame(frame,template)
    #         cropped=extract_frame(frame,template)
    #         plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    #         round,country,name,difficulty,divePosition,somersaults,diveGroup,twists = extract_diveinfo(cropped)
    #         row = {'Sno':sno,'Match name':videoName,"Time":total_seconds,'Name':name.strip(),'Country':country.strip(),'Difficulty':difficulty,'Dive Position':divePosition,'Somersaults':somersaults,
    #     'Dive Group':diveGroup,
    #     'Twists':twists }
    #         print(timeInVideo," dive started")

    #     if diveStarted and( ("difficulty" in text.lower() and "penalty" in text.lower()) | ("penalty" in text.lower()) ):
    #         print(timeInVideo," dive scores announced")
    #         cropped=extract_frame(frame,templateRes)
    #         plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    #         croppedText=extract_text(cropped)
    #         score=get_score(cropped)
    #         finalScore=extract_final_score(text)
    #         #print(finalScore,score)
    #         row['Score'] = score
    #         row['Final Score'] = finalScore
    #         extract_clip(filePath,'ExtractedVideos/'+str(sno)+'_'+videoName+'.mp4',int(row['Time']),int(total_seconds))
    #         diveStarted=False
    #         sno+=1
    #         print(row)
    #         save_data(row,"scores")



    #     currentFrameInSec=0
    # cap.release()

# %%

templateTokyo = cv2.imread(r'D:\Development\Thesis\HelperImages\template_tokyo_2.png', cv2.IMREAD_GRAYSCALE)
templateResTokyo = cv2.imread(r'D:\Development\Thesis\HelperImages\tokyo_res_template_2.png', cv2.IMREAD_GRAYSCALE)
filePath=r"E:\BTP\DownloadedVideos\Full Womens 10M Platform FINAL Tokyo2020.mp4"
videoName="Women_ 10m_Platform_Semi-Final_Diving_Tokyo2020"
extract_from_video(filePath,videoName,templateTokyo,templateResTokyo)

# %%
filePath="DownloadedVideos/Men10m_Platform_Final_Diving_Tokyo2020.mp4"
videoName="Men10m_Platform_Final_Diving_Tokyo2020"
extract_from_video(filePath,videoName,templateTokyo,templateResTokyo)

# %%
filePath="DownloadedVideos/Men10m_Platform_Preliminary_Diving_Tokyo2020.mp4"
videoName="Men10m_Platform_Preliminary_Diving_Tokyo2020"
extract_from_video(filePath,videoName,templateTokyo,templateResTokyo)

# %%
filePath="DownloadedVideos/Men3m_Springboard_Preliminary_Diving_Tokyo2020.mp4"
videoName="Men3m_Springboard_Preliminary_Diving_Tokyo2020"
extract_from_video(filePath,videoName,templateTokyo,templateResTokyo)

# %%



