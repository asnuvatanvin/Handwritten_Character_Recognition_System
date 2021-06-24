import tkinter as tk
from tkinter import *
#from tkinter.filedialog import askopenfilename

import cv2

import numpy as np 
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from PIL import Image ,ImageTk 
from tkinter.filedialog import askopenfilename,asksaveasfilename
import sqlite3
from tkinter import ttk
from time import sleep
from tkinter import filedialog



from keras import backend as K
K.image_data_format()

def showimages(src_img,bin_img,final_thr):
    
   # img = cv2.imread('face_person1.jpg')
# convert the images to PIL format...
    '''
    edged = Image.fromarray(src_img)

    
    hsl = filedialog.asksaveasfile(mode='w',initialdir = "./",title = "save file as",filetypes = (("jpg files","*.jpg"),("jpeg files","*.jpeg")))
    if hsl is None:
        return
    edged.save(hsl)
    
    
    window_name="RESULT"
    '''
    cv2.imwrite("output.jpg",src_img)
   # cv2.nameWindow(window_name,cv2.WINDOW_NORMAL())
    '''
    cv2.imshow(window_name, src_img)
    
    #cv2.imshow("Binary Image", bin_img)
    #cv2.imshow("Segment Image", final_thr)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    '''
    


def line_array(array):
    list_x_upper = []
    list_x_lower = []
    for y in range(5, len(array)-5):
        s_a, s_p = strtline(y, array)
        e_a, e_p = endline(y, array)
        print(str(s_a) + ',' + str(s_p) + ',' + str(e_a) + ',' + str(e_p) + ',' + str(y))
        if s_a>=7 and s_p>=5:
            list_x_upper.append(y)
        # bin_img[y][:] = 255
        if e_a>=5 and e_p>=7:
            list_x_lower.append(y)
            # bin_img[y][:] = 255
    return list_x_upper, list_x_lower

def strtline(y, array):
    count_ahead = 0
    count_prev = 0
    for i in array[y:y+10]:
        if i > 3:
            count_ahead+= 1  
    for i in array[y-10:y]:
        if i == 0:
            count_prev += 1  
    return count_ahead, count_prev

def endline(y, array):
    count_ahead = 0
    count_prev = 0
    for i in array[y:y+10]:
        if i==0:
            count_ahead+= 1  
    for i in array[y-10:y]:
        if i >3:
            count_prev += 1  
    return count_ahead, count_prev

def endline_word(y, array, a):
    count_ahead = 0
    count_prev = 0
    for i in array[y:y+2*a]:
        if i < 2:
            count_ahead+= 1  
    for i in array[y-a:y]:
        if i > 2:
            count_prev += 1  
    return count_prev ,count_ahead

def end_line_array(array, a):
    list_endlines = []
    for y in range(len(array)):
        e_p, e_a = endline_word(y, array, a)
        #print(e_p, e_a)
        if e_a >= int(0.8*a) and e_p >= int(0.7*a):
            list_endlines.append(y)
    return list_endlines

def refine_endword(array):
    refine_list = []
    for y in range(len(array)-1):
        if array[y]+1 < array[y+1]:
            refine_list.append(array[y])
    refine_list.append(array[-1])
    return refine_list


def refine_array(array_upper, array_lower):
    upperlines = []
    lowerlines = []
    for y in range(len(array_upper)-1):
        if array_upper[y] + 5 < array_upper[y+1]:
            upperlines.append(array_upper[y]-10)
    for y in range(len(array_lower)-1):
        if array_lower[y] + 5 < array_lower[y+1]:
            lowerlines.append(array_lower[y]+10)

    upperlines.append(array_upper[-1]-10)
    lowerlines.append(array_lower[-1]+10)
    
    return upperlines, lowerlines

def letter_width(contours):
    letter_width_sum = 0
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x,y,w,h = cv2.boundingRect(cnt)
            letter_width_sum += w
            count += 1

    return letter_width_sum/count


def end_wrd_dtct(final_local, i, bin_img, mean_lttr_width , width, lines_img ,final_thr):
    count_y = np.zeros(shape = width)
    for x in range(width):
        for y in range(final_local[i],final_local[i+1]):
            if bin_img[y][x] == 255:
                count_y[x] += 1
    #end_lines = end_line_array(count_y, int(mean_lttr_width))
    #endlines = refine_endword(end_lines)
    #print(i)
    '''for x in range(len(count_y)):
        if max(count_y[0:x+1]) >= 3 and max(count_y[x:]) >= 3 and (20-np.count_nonzero(count_y[x-10:x+10])) > 6:
            print(x)'''

    contours, hierarchy = cv2.findContours(lines_img[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_width_sum = 0
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            letter_width_sum += w
            count += 1
    if count != 0 :
        mean_width = letter_width_sum / count
    else:
        mean_width = 0
    #print(mean_width)
    spaces = []
    line_end = []
    for x in range(len(count_y)):
        number = int(0.5*int(mean_width)) - np.count_nonzero(count_y[x-int(0.25*int(mean_width)):x+int(0.25*int(mean_width))])
        if max(count_y[0:x + 1]) >= 3 and number >= 0.4*int(mean_width):
            spaces.append(x)
        if max(count_y[x:]) <= 2:
            line_end.append(x)
    true_line_end = min(line_end) + 10
    #spaces = refine_endword(spaces)
    #print(spaces)
    #print(true_line_end)
    reti = []
    final_spaces = []
    for j in range(len(spaces)):
        if spaces[j] < true_line_end:
            if spaces[j] == spaces[j-1] + 1:
                reti.append(spaces[j-1])
            elif spaces[j] != spaces[j-1] + 1 and spaces[j-1] == spaces[j-2] +1:
                reti.append(spaces[j-1])
                retiavg = int(sum(reti)/len(reti))
                final_spaces.append(retiavg)
                reti = []
            elif spaces[j] != spaces[j-1] + 1 and spaces[j-1] != spaces[j-2] +1 and spaces[j] != spaces[j+1] -1:
                final_spaces.append(spaces[j])
        elif spaces[j] == true_line_end:
            final_spaces.append(true_line_end)
    #print(final_spaces)
    for x in final_spaces:
        final_thr[final_local[i]:final_local[i+1], x] = 255
    return final_spaces


def letter_seg(lines_img, x_lines, i):
    copy_img = lines_img[i].copy()
    x_linescopy = x_lines[i].copy()
    
    letter_img = []
    letter_k = []
    
    contours, hierarchy = cv2.findContours(copy_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   
    for cnt in contours:
        if cv2.contourArea(cnt) > 5:
            x,y,w,h = cv2.boundingRect(cnt)
            # letter_img.append(lines_img[i][y:y+h, x:x+w])
            letter_k.append((x,y,w,h))

    letter_width_sum = 0
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            letter_width_sum += h
            count += 1

    #mean_height = letter_width_sum/count

    letter = sorted(letter_k, key=lambda student: student[0])

    for e in range(len(letter)):
        if e < len(letter)-1:
            if abs(letter[e][0] - letter[e+1][0]) <= 2:
                x,y,w,h = letter[e]
                x2,y2,w2,h2 = letter[e+1]
                if h >= h2:
                    letter[e] = (x,y2,w,h+h2)
                    letter.pop(e+1)
                elif h < h2:
                    letter[e+1] = (x2,y,w2,h+h2)
                    letter.pop(e)

    for e in range(len(letter)):
        letter_img_tmp = lines_img[i][letter[e][1]-0:letter[e][1]+letter[e][3]+0,letter[e][0]-0:letter[e][0]+letter[e][2]+0]
        letter_img_tmp = cv2.resize(letter_img_tmp, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        width = letter_img_tmp.shape[1]
        height = letter_img_tmp.shape[0]
        count_y = np.zeros(shape=(width))
        for x in range(width):
            for y in range(height):
                if letter_img_tmp[y][x] == 255:
                    count_y[x] = count_y[x] +1
        #print(count_y)
        max_list = []
        for z in range(len(count_y)):
            if z>=5 and z<= len(count_y)-6:
                if max(count_y[z-5:z+6]) == count_y[z] and count_y[z] >= 2:
                    max_list.append(z)
            elif z<5:
                if max(count_y[0:z+6]) == count_y[z] and count_y[z] >= 2:
                    max_list.append(z)
            elif z > len(count_y)-6:
                if max(count_y[z-5:len(count_y)-1]) == count_y[z] and count_y[z] >= 2:
                    max_list.append(z)
       # print(max_list)
        rem_list = []
        final_max_list = []
        for z in range(len(max_list)):
            if z > 0:
                if max_list[z]-max_list[z-1] <= 3:
                    rem_list.append(z-1)
        for z in range(len(max_list)):
            if z not in rem_list:
                final_max_list.append(max_list[z])
       # print(final_max_list)
        if len(final_max_list) <= 1:
           #print(False)
           s=1
        else:
            max_len = len(final_max_list) - 1
            for j in range(max_len):
                list = count_y[final_max_list[j]:final_max_list[j+1]]
                min_list = sorted(list)[:3]
                avg = sum(min_list)/len(min_list)
               # print(avg)



    #x_linescopy.pop(0)
    word = 1
    letter_index = 0
    #print("here I am")
    #print(x_linescopy)
    for e in range(len(letter)):
        #print(str(letter[e][0]) + ',' + str(letter[e][1]) + ',' + str(letter[e][2]) + ',' + str(letter[e][3]) + ',' + str(e))
        if(letter[e][0]<x_linescopy[0]):
            letter_index += 1
            letter_img_tmp = lines_img[i][letter[e][1]-0:letter[e][1]+letter[e][3]+5,letter[e][0]-2:letter[e][0]+letter[e][2]+2]
            letter_img = cv2.resize(letter_img_tmp, dsize =(28, 28), interpolation = cv2.INTER_AREA)
            cv2.imwrite('./result/img1/'+str(i+1)+'_'+str(word)+'_'+str(letter_index)+'.jpg', 255-letter_img)
        else:
            x_linescopy.pop(0)
            word += 1
            letter_index = 1
            letter_img_tmp = lines_img[i][letter[e][1]-0:letter[e][1]+letter[e][3]+5,letter[e][0]-2:letter[e][0]+letter[e][2]+2]
            letter_img = cv2.resize(letter_img_tmp, dsize =(28, 28), interpolation = cv2.INTER_AREA)
            cv2.imwrite('./result/img1/'+str(i+1)+'_'+str(word)+'_'+str(letter_index)+'.jpg', 255-letter_img)
            # print(letter[e][0],x_linescopy[0], word)


def add_padding(img, pad_l, pad_t, pad_r, pad_b):
    
    height, width = img.shape
    #Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype = np.int)
    img = np.concatenate((pad_left, img), axis = 1)

    #Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis = 0)

    #Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis = 1)

    #Adding padding to the bottom
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis = 0)

    return img


def generate_output():
    st=''

    st=root.file
   
   
    print("\n........Program Initiated.......\n")
  
    src_img= cv2.imread(st,1)
    copy = src_img.copy()
    height = src_img.shape[0]
    width = src_img.shape[1]

    print("\n Resizing Image........")
    src_img = cv2.resize(copy, dsize =(1320, int(1320*height/width)), interpolation = cv2.INTER_AREA)

    height = src_img.shape[0]
    width = src_img.shape[1]

    print("#---------Image Info:--------#")
    print("\tHeight =",height,"\n\tWidth =",width)
    print("#----------------------------#")

    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    print("Applying Adaptive Threshold with kernel :- 21 X 21")
    bin_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,20)
    coords = np.column_stack(np.where(bin_img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    center = (w//2,h//2)
    angle = 0
    M = cv2.getRotationMatrix2D(center,angle,1.0)
    bin_img = cv2.warpAffine(bin_img,M,(w,h),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    bin_img1 = bin_img.copy()
    bin_img2 = bin_img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)
# final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
# final_thr = cv2.dilate(bin_img,kernel1,iterations = 1)
    print("Noise Removal From Image.........")
    final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    contr_retrival = final_thr.copy()


    print("Beginning Character Segmentation..............")
    count_x = np.zeros(shape= (height))
    for y in range(height):
        for x in range(width):
            if bin_img[y][x] == 255 :
                count_x[y] = count_x[y]+1

    local_minima = []
    for y in range(len(count_x)):
        if y >= 10 and y <= len(count_x)-11:
            arr1 = count_x[y-10:y+10]
        
        elif y < 10:
            arr1 = count_x[0:y+10]
            
        else:
            arr1 = count_x[y-10:len(count_x)-1]
        if min(arr1) == count_x[y]:
            local_minima.append(y)

    final_local = []
    init = []
    end = []
    for z in range(len(local_minima)):
        if z != 0 and z!= len(local_minima)-1:
            if local_minima[z] != (local_minima[z-1] +1) and local_minima[z] != (local_minima[z+1] -1):
                final_local.append(local_minima[z])
            elif local_minima[z] != (local_minima[z-1] + 1) and local_minima[z] == (local_minima[z+1] -1):
                init.append(local_minima[z])
            elif local_minima[z] == (local_minima[z-1] + 1) and local_minima[z] != (local_minima[z+1] -1):
                end.append(local_minima[z])
            
        elif z == 0:
            if local_minima[z] != (local_minima[z+1]-1):
                final_local.append(local_minima[z])
            elif local_minima[z] == (local_minima[z+1]-1):
                init.append(local_minima[z])
            
        elif z == len(local_minima)-1:
            if local_minima[z] != (local_minima[z-1]+1):
                final_local.append(local_minima[z])
            elif local_minima[z] == (local_minima[z-1]+1):
                   end.append(local_minima[z])
            
    for j in range(len(init)):
       mid = (init[j] + end[j])/2
       if (mid % 1) != 0:
           mid = mid+0.5
       final_local.append(int(mid))

    final_local = sorted(final_local)

    no_of_lines = len(final_local) - 1

    print("\nGiven Text has   # ",no_of_lines, " #   no. of lines")

    lines_img = []

    for i in range(no_of_lines):
        lines_img.append(bin_img2[final_local[i]:final_local[i+1], :])

    contours, hierarchy = cv2.findContours(contr_retrival,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    final_contr = np.zeros((final_thr.shape[0],final_thr.shape[1],3), dtype = np.uint8)
    cv2.drawContours(src_img, contours, -1, (0,255,0), 1)

    mean_lttr_width = letter_width(contours)
    print("\nAverage Width of Each Letter:- ", mean_lttr_width)


    x_lines = []

    for i in range(len(lines_img)):
        x_lines.append(end_wrd_dtct(final_local, i, bin_img, mean_lttr_width,width ,lines_img,final_thr))

    for i in range(len(x_lines)):
        x_lines[i].append(width)

    #print(x_lines)

#-------------Letter Segmentation-------------#

    cv2.waitKey(0)
    for i in range(no_of_lines):
        letter_seg(lines_img, x_lines, i)   


    chr_img = bin_img1.copy()

    contours, hierarchy = cv2.findContours(chr_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# final_contr = np.zeros((final_thr.shape[0],final_thr.shape[1],3), dtype = np.uint8)
# cv2.drawContours(src_img, contours, -1, (0,255,0), 1)

    i=0;
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x,y,w,h = cv2.boundingRect(cnt)
            #cv2.rectangle(src_img,(x,y),(x+w,y+h),(0,255,0),2)
            crop_image = bin_img[y:y+h,x:x+w]
            cv2.imwrite("./result/" + str(i) +".png",crop_image)
            i=i+1


    images = []
    folder='./result'
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        th, a = cv2.threshold(img, 127, 255,cv2.THRESH_OTSU)
        if a is not None:
            a=cv2.resize(a,(100,80))
            # create blank image - y, x
            col_sum = np.where(np.sum(a, axis = 0)>0)
            row_sum = np.where(np.sum(a, axis = 1)>0)
            y1, y2 = row_sum[0][0], row_sum[0][-1]
            x1, x2 = col_sum[0][0], col_sum[0][-1]
        
            cropped_image = a[y1:y2, x1:x2]        
            cropped_image=cv2.resize(a,(20,20))
            padded_image = add_padding(cropped_image, 4, 4, 4, 4)
            cv2.imwrite('./resized/'+filename,padded_image)

    print("Images resized and saved into designated folder")
       
    #Load model  
    model=load_model('./Merged Dataset Models/JANABHATT1.h5')
    print(model.summary())

    import string
    letter_count = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',
                    11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',
                    20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',
                    30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z'}
    print('Letter_count: ',letter_count.items())


    x=[]
    res=[]
    fname=[]
    folder='./resized'
    dirFiles=os.listdir(folder)
    dirFiles = sorted(dirFiles,key=lambda x: int(os.path.splitext(x)[0]))
    for filename in dirFiles:
        imt = cv2.imread(os.path.join(folder,filename))
        imt = cv2.blur(imt,(6,6))
        gray = cv2.cvtColor(imt,cv2.COLOR_BGR2GRAY)
        ret, imt = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
        #kernel = np.ones((5,5),np.uint8)
        #imt = cv2.dilate(imt,kernel,iterations=1)
        if imt is not None:
            #imt=cv2.imread(im)
            imt = imt.reshape((28, 28, 1))
            #plt.imshow(imt)        
            #plt.show()
            imt=imt/255
            x.append(imt)
            fname.append(filename)
   # print(fname)
    #print(x)
    x=np.array(x,dtype='float32');    
    predictions = model.predict(x)
    classes = np.argmax(predictions,axis=1)    
        #print(predictions)

    g=0
    namelist=[]
    for i in range(len(classes)):
        for k,v in letter_count.items():
            if k==classes[i]:
                namelist.append(v);
                #print(filename,classes)
            
    i=0;
    align = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(src_img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(src_img, namelist[i], (x - 10, y - 10),
	        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            align.append((x,y,x+w,namelist[i]))
            i=i+1
    align.reverse()


    sum=0
    count=0
    avg = None
    checkpoints = []

    for i in range(len(align)-1):
        y1 = align[i][1]
        y2 = align[i+1][1]
        yy = abs(y1-y2)
        if avg!=None and yy>avg*3:
            checkpoints.append(i)
        sum = sum + yy
        count = count + 1
        avg = sum/count + 1



    initial = 0
    final = []
    for i in range (len(checkpoints)):
        temp = align[initial:checkpoints[i]+1]
        temp.sort(key=lambda tup:tup[0])
        final = final + temp
        initial = checkpoints[i] + 1
    
    temp = align[initial:len(align)]
    temp.sort(key=lambda tup:tup[0])
    final = final + temp
    

    sum=0
    count=0
    avg = None

    outF = open("HCDROUTPUT.txt", "w")

    for i in range(len(final)):
        outF.write(final[i][3])
        if i+1<len(final):
            d1 = final[i][2]
            d2 = final[i+1][0]
            d = abs(d1-d2)
            if avg!=None and d>avg*3:
                outF.write(" ")
            else:
                sum = sum + d
                count = count + 1
                avg = sum/count + 1
        if(i+1<len(final) and final[i][0]>final[i+1][0]):
            outF.write("\n")
    
    outF.close()

    return(src_img,bin_img,final_thr)
class Login:
    def __init__(self,root):
        
        self.root=root
        self.root.title("Hand written character and digit recognition system")
        self.root.geometry('1199x600+300+200')
        self.root.configure(bg='#17BA96')
        self.root.resizable(False,False)
        
        t=''
        #frame
        Frame_frame=Frame(self.root,bg="white")
        Frame_frame.place(x=350,y=100,height=340,width=500)
        
        #label
        label_0 = Label(Frame_frame, text="SELECT AN IMAGE",font=("Raleway",20,"bold"),fg='#E96A0A',bg="white").place(x=110,y=10)
        browse_text=tk.StringVar()
        browse_btn=tk.Button(Frame_frame,textvariable=browse_text,command=lambda:open_file(), font="Raleway",bg="#20bebe",fg="white")
        browse_text.set("Browse")
        browse_btn.pack()
        browse_btn.place(x=180,y=100,height=40,width=150)
        
        submit_text=tk.StringVar()
        submit =tk.Button(Frame_frame, textvariable=submit_text,command=lambda:show(),font="Raleway",bg="#20bebe",fg="white")
        submit_text.set("Generate")
        submit.pack()
        submit.place(x=180,y=150,height=40,width=150)
        
        image_buttn =tk.Button(Frame_frame, text ="Show Image",command=lambda:output_image(),font="Raleway",bg='#20bebe',fg='white')
        image_buttn.pack()
        image_buttn.place(x=60,y=230,height=40,width=150)
        
        doc_buttn =tk.Button(Frame_frame, text ="Show Text",command=lambda:output_doc(),font="Raleway",bg='#20bebe',fg='white')
        doc_buttn.pack()
        doc_buttn.place(x=290,y=230,height=40,width=150)
        
        exit_buttn =tk.Button(self.root, text ="Exit",command=lambda:exit(),font="Raleway",bg='white',fg='#E96A0A')
        exit_buttn.pack()
        exit_buttn.place(x=550,y=530,height=40,width=150)
       
            
        
        def output_doc():
            def read():
                text_file = open("./HCDROUTPUT.txt",'r')
                stuff=text_file.read()
                my_text.insert(END,stuff)
                text_file.close()
            def save_text(new_win):
             text_file = open("./HCDROUTPUT.txt",'w+')
             text_file.write(my_text.get(1.0,END))
             
             text_file.close()
            def close():
                new_win.destroy()
            k = "Generated"
            if submit_text.get() == k:
                new_win= Toplevel(root)
                new_win.geometry('1199x600+300+200')
                new_win.title("output Text File")
            
                my_text=Text(new_win,width=40, height=10,font=("Raleway",20))
                my_text.pack(pady=20)
            
                read()
                image_buttn =tk.Button(new_win, text ="Save edited text",command=lambda:save_text(new_win),font="Raleway",bg='#20bebe',fg='white')
                image_buttn.pack()
            
            
                close_buttn =tk.Button(new_win, text ="Close",command=lambda:close(),font="Raleway",bg='#20bebe',fg='white')
                close_buttn.pack()
                close_buttn.place(x=520,y=500,height=40,width=150)
            else:
                 messagebox.showerror("Error","No output generated",parent=self.root)   
            
           
            
        def output_image():
         k = "Generated"
         if submit_text.get() == k:
            window_name="RESULT"
            im=cv2.imread("./output.jpg")
   # cv2.nameWindow(window_name,cv2.WINDOW_NORMAL())
            cv2.imshow(window_name, im)
    
    #cv2.imshow("Binary Image", bin_img)
    #cv2.imshow("Segment Image", final_thr)
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
         else :
             messagebox.showerror("Error","No output generated",parent=self.root)
            
        def exit():
            self.root.destroy()
        
        def show():
            
         d="loaded"
         if browse_text.get() == d:
            teams = range(100)   
            popup = tk.Toplevel()
            tk.Label(popup, text="Processing..").place(x=300,y=100)
            progress = 0
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(popup, variable=progress_var, maximum=100)
            progress_bar.grid(row=1, column=0)#.pack(fill=tk.X, expand=1, side=tk.BOTTOM)
            popup.pack_slaves()
            i=0
            progress_step = float(100.0/len(teams))
            for team in teams:
                i=i+1
                #print(i)
                popup.update()
                sleep(.05) # lauch task
                progress += progress_step
                progress_var.set(progress)
                if i==49:
                    src_img,bin_img,final_thr=generate_output()
            popup.destroy() 
            submit_text.set("Generated")
            showimages(src_img,bin_img,final_thr)
            browse_text.set("Browse")
        
         else :
             messagebox.showerror("Error","No image selected",parent=self.root)
            
        def open_file():
            browse_text.set("loading...")
            root.file =filedialog.askopenfilename(initialdir = "C:/Users/Asus/Desktop/FINAL/images",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("jpeg files","*.jpeg")))
            if root.file:
                browse_text.set("loaded") 
                submit_text.set("Generate")
                
            else:
                browse_text.set("Browse")
                submit_text.set("Generate")
                
        
if __name__== "__main__" :   
    
 root = Tk()
 obj = Login(root)

root.mainloop()




