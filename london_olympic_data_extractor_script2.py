
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
    cv2.imshow("f",cropped_divegroup)
    cv2.waitKey(0)
    text = pytesseract.image_to_string(cropped_divegroup,lang="eng")
    text=text.replace("\n","")
    print(text)
    return text

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

# %%
templateLondon = cv2.imread('HelperImages/template_london_2.png', cv2.IMREAD_GRAYSCALE)
templateResLondon = cv2.imread('HelperImages/template_res_london_2.png', cv2.IMREAD_GRAYSCALE)
# extract_from_video("DownloadedVideos/Diving - Mens 3m Springboard - Final  London 2012 Olympic Games.mp4","Mens_3m_Springboard_-_Final_London_2012",templateLondon,templateResLondon)
extract_from_video("E:/BTP/DownloadedVideos/Diving - Mens 3m Springboard - Final  London 2012 Olympic Games.mp4","Mens_3m_Springboard_-_Final_London_2012",templateLondon,templateResLondon)


