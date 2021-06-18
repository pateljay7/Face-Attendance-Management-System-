from users.models import tblUserdetails
import calendar
from django.shortcuts import render, redirect
from .forms import usernameForm, DateForm, UsernameAndDateForm, DateForm_2, Holiday_form
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from users.models import tblPresent, tblTime, tblHoliday
import seaborn as sns
import pandas as pd
#from django.db.models import Count
import mpld3
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
import time
from datetime import timedelta, date
#from wsgiref.util import FileWrapper
#from django.http import StreamingHttpResponse
#from PIL import Image
mpl.use('Agg')



# function to check user data in present in user database 
def username_present(username):
    if User.objects.filter(username=username).exists():
        return True
    return False

# function to check user data in present in Present database
def has_Present(username):
    if tblPresent.objects.filter(user=username).exists():
        return True
    return False

# function to check user data in present in Time database
def has_Times(username):
    if tblTime.objects.filter(user=username).exists():
        return True
    return False

# function to check user data in present in user_details database
def has_Details(username):
    if tblUserdetails.objects.filter(user=username).exists():
        return True
    return False


# To create a dataset for user 
def create_dataset(username):
    id = username
    # if folder of dataset for particular is not exist then it will automatically create 
    if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id)) == False):
        os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
    directory = 'face_recognition_data/training_dataset/{}/'.format(id)

    # Detect face
    # Loading the HOG face detector and the shape predictpr for allignment
    print("[INFO] Loading the facial detector")
    detector = dlib.get_frontal_face_detector()
    # Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
    predictor = dlib.shape_predictor(
        'face_recognition_data/shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    # capture images from the webcam and process and detect the face
    # Initialize the video stream
    print("[INFO] Initializing Video stream")
    vs = VideoStream(src=0).start()
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is
    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while(True):
        # Capturing the image
        # vs.read each frame
        frame = vs.read()
        # Resize each image
        frame = imutils.resize(frame, width=800)
        # the returned img is a colored image but for the classifier to work we need a greyscale image
        # to convert
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # To store the faces
        # This will detect all the images in the current frame, and it will return the coordinates of the faces
        # Takes in image and some other parameter for accurate result
        faces = detector(gray_frame, 0)
        # In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.

        for face in faces:
            print("inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame, gray_frame, face)
            sampleNum = sampleNum+1

            if face is None:
                print("face is none")
                continue

            cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
            face_aligned = imutils.resize(face_aligned, width=400)
            #cv2.imshow("Image Captured",face_aligned)
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(50)

        # Showing the image in another window
        # Creates a window with window name "Face" and with the image img
        cv2.imshow("Add Images", frame)
        # Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        key = cv2.waitKey(50) & 0xFF

        # To get out of the loop
        if(sampleNum > 10) or key==ord('q'):
            break

    # Stoping the videostream
    vs.stop()
    # destroying all the windows
    cv2.destroyAllWindows()

# this function is used for predict the detected user 
# it will return predicted person name and percentage probability of person
def predict(face_aligned, svc, threshold=0.7):
    face_encodings = np.zeros((1, 128))
    # exception handling 
    try:
        x_face_locations = face_recognition.face_locations(face_aligned)
        faces_encodings = face_recognition.face_encodings(
            face_aligned, known_face_locations=x_face_locations)
        if(len(faces_encodings) == 0):
            return ([-1], [0])
    except:
        return ([-1], [0])
    # probability of person detected
    prob = svc.predict_proba(faces_encodings)
    # detected person name
    result = np.where(prob[0] == np.amax(prob[0]))
    if(prob[0][result[0]] <= threshold):
        return ([-1], prob[0][result[0]])

    return (result[0], prob[0][result[0]])

# this function is used for visualization of user's facial data
# it will generate scatter graph of all user's face data
def vizualize_Data(embedded, targets,):
    X_embedded = TSNE(n_components=2).fit_transform(embedded)
    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1))
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    # save scatter graph as .png format and save it in folder
    plt.savefig(
        './recognition/static/recognition/img/training_visualisation.png')
    plt.close()

# For update attendance in database for mark in 
# when user click on mark in button and if face will be detected then this function will update user attendance in database
def update_attendance_in_db_in(present):
    # save attendance data with time and today's date
    today = datetime.date.today()
    time = datetime.datetime.now()
    for person in present:
        # get username from present list 
        user = User.objects.get(username=person)
        # exception handling
        try:
            qs = tblPresent.objects.get(user=user, date=today)
        except:
            qs = None

        if qs is None:
            if present[person] == True:
                a = tblPresent(user=user, date=today, present=True)
                a.save()
        else:
            if present[person] == True:
                qs.present = True
                qs.save(update_fields=['present'])
        if present[person] == True:
            a = tblTime(user=user, date=today, time=time, out=False)
            a.save()

# For update attendance in database for mark out
# when user click on mark out button and if face will be detected then this function will update user attendance in database
def update_attendance_in_db_out(present):
    # save attendance data with time and today's date
    today = datetime.date.today()
    time = datetime.datetime.now()
    for person in present:
        # get the user name 
        user = User.objects.get(username=person)
        if present[person] == True:
            # update attendance in db
            a = tblTime(user=user, date=today, time=time, out=True)
            a.save()

# check the validity of times
def check_validity_times(times_all):
    if(len(times_all) > 0):
        sign = times_all.first().out
    else:
        sign = True
    times_in = times_all.filter(out=False) # time in
    times_out = times_all.filter(out=True) # time out
    if(len(times_in) != len(times_out)):
        sign = True
    break_hourss = 0
    if(sign == True):
        check = False
        break_hourss = 0
        return (check, break_hourss)
    prev = True
    prev_time = times_all.first().time
    for obj in times_all:
        curr = obj.out
        if(curr == prev):
            check = False
            break_hourss = 0
            return (check, break_hourss)
        if(curr == False):
            curr_time = obj.time
            to = curr_time
            ti = prev_time
            break_time = ((to-ti).total_seconds())/3600
            break_hourss += break_time
        else:
            prev_time = obj.time
        prev = curr
    return (True, break_hourss)


# for convert the Hours into minutes
def convert_hours_to_hours_mins(hours):
    h = int(hours)
    hours -= h
    m = hours*60
    m = math.ceil(m)
    return str(str(h) + " hrs " + str(m) + "  mins")

# this function is used for make database object to store user data by given time range and all enmployee 
def hours_vs_date_given_employee(present_qs, time_qs, admin=True):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    qs = present_qs

    for obj in qs:
        date = obj.date
        # get time in form Time database
        times_in = time_qs.filter(date=date).filter(out=False).order_by('time')
        # get time out form Time database
        times_out = time_qs.filter(date=date).filter(out=True).order_by('time')
        times_all = time_qs.filter(date=date).order_by('time')
        obj.time_in = None # initialize time in
        obj.time_out = None # initialize time out
        obj.hours = 0
        obj.break_hours = 0
        # save time in
        if (len(times_in) > 0):
            obj.time_in = times_in.first().time
        # save time out
        if (len(times_out) > 0):
            obj.time_out = times_out.last().time

        if(obj.time_in is not None and obj.time_out is not None):
            ti = obj.time_in
            to = obj.time_out
            hours = ((to-ti).total_seconds())/3600
            obj.hours = hours

        else:
            obj.hours = 0
        # if break hours is avilable and valid then it save in var
        (check, break_hourss) = check_validity_times(times_all)
        if check:
            obj.break_hours = break_hourss
        else:
            obj.break_hours = 0
        # append data in dataframe
        df_hours.append(obj.hours)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)
    # read dataframe 
    df = read_frame(qs)
    df["hours"] = df_hours
    df["break_hours"] = df_break_hours

    # it will create bar plot and saved it as .png file in folder
    sns.barplot(data=df, x='date', y='hours')
    plt.xticks(rotation='vertical')
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    if(admin):
        plt.savefig(
            './recognition/static/recognition/img/attendance_graphs/hours_vs_date/graph.png')
        plt.close()
    else:
        plt.savefig(
            './recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
        plt.close()
    return qs


# this function is used for make database object to store user data by given time range and for particular enmployee 
def hours_vs_employee_given_date(present_qs, time_qs):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    df_username = []
    qs = present_qs

    for obj in qs:
        # get username
        user = obj.user
        times_in = time_qs.filter(user=user).filter(out=False) # query for time in from Time database
        times_out = time_qs.filter(user=user).filter(out=True) # query for time in from Present database
        times_all = time_qs.filter(user=user)
        obj.time_in = None # initialize time in
        obj.time_out = None # initialize time out
        obj.hours = 0
        obj.hours = 0
        # save time in
        if (len(times_in) > 0):
            obj.time_in = times_in.first().time
        # save time out
        if (len(times_out) > 0):
            obj.time_out = times_out.last().time
        if(obj.time_in is not None and obj.time_out is not None):
            ti = obj.time_in
            to = obj.time_out
            hours = ((to-ti).total_seconds())/3600 # convert it into hours
            obj.hours = hours
        else:
            obj.hours = 0
        (check, break_hourss) = check_validity_times(times_all)
        if check:
            obj.break_hours = break_hourss
        else:
            obj.break_hours = 0
        # append data in dataframe
        df_hours.append(obj.hours)
        df_username.append(user.username)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)
    # read dataframe
    df = read_frame(qs)
    df['hours'] = df_hours
    df['username'] = df_username
    df["break_hours"] = df_break_hours

    # it will create bar plot and saved it as .png file in folder
    sns.barplot(data=df, x='username', y='hours')
    plt.xticks(rotation='vertical')
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    plt.savefig(
        './recognition/static/recognition/img/attendance_graphs/hours_vs_employee/graph.png')
    plt.close()
    return qs

# return total number of employee
def total_number_employees():
    qs = User.objects.all()
    return (len(qs) - 1)
    # -1 to account for admin

# return number of employee present today
def employees_present_today():
    today = datetime.date.today()
    qs = tblPresent.objects.filter(date=today).filter(present=True)
    return len(qs)


# used for prepare the graph(number of empployees VS date) for current week
def this_week_emp_count_vs_date():
    print("this week vs dte")
    # today's date
    today = datetime.date.today()
    # calculate some day of last week
    some_day_last_week = today-datetime.timedelta(days=7)
    # calculate monday of last week
    monday_of_last_week = some_day_last_week - \
        datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
    # monday of this week
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    # query to retrive recordes of current week moday to today 
    qs = tblPresent.objects.filter(
        date__gte=monday_of_this_week).filter(date__lte=today)
    str_dates = []
    emp_count = []
    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    # calculate number of employees per days for one week
    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs = tblPresent.objects.filter(date=date).filter(present=True)
        emp_count.append(len(qs))

    while(cnt < 5):
        date = str(monday_of_this_week+datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if(str_dates.count(date)) > 0:
            idx = str_dates.index(date)
            emp_cnt_all.append(emp_count[idx])
        else:
            emp_cnt_all.append(0)

    # create dataframe
    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["Number of employees"] = emp_cnt_all
    # plot line graph for date vs num of employees
    sns.lineplot(data=df, x='date', y='Number of employees')
    # save graph in folder 
    plt.savefig(
        './recognition/static/recognition/img/attendance_graphs/this_week/1.png')
    plt.close()


# used for prepare the graph(number of empployees VS date) for previous week
def last_week_emp_count_vs_date():
    # today's date
    today = datetime.date.today()
    # calculate some day of last week
    some_day_last_week = today-datetime.timedelta(days=7)
    # calculate monday of last week
    monday_of_last_week = some_day_last_week - \
        datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
    # calculate monday of current week
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    # query to retrive recordes of current week moday to today 
    qs = tblPresent.objects.filter(date__gte=monday_of_last_week).filter(
        date__lt=monday_of_this_week)
    str_dates = []
    emp_count = []

    str_dates_all = []
    emp_cnt_all = []
    cnt = 0
# calculate number of employees per days for one week
    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs = tblPresent.objects.filter(date=date).filter(present=True)
        emp_count.append(len(qs))

    while(cnt < 5):
        date = str(monday_of_last_week+datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if(str_dates.count(date)) > 0:
            idx = str_dates.index(date)
            emp_cnt_all.append(emp_count[idx])
        else:
            emp_cnt_all.append(0)
    # create dataframe
    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["emp_count"] = emp_cnt_all
    # plot line graph for date vs num of employees
    sns.lineplot(data=df, x='date', y='emp_count')
    # save graph in .png format in folder
    plt.savefig(
        './recognition/static/recognition/img/attendance_graphs/last_week/1.png')
    plt.close()


# Home screen
def home(request):
    return render(request, 'recognition/home.html')

# it will redirect user to Admin dashboard or Employee dashboard
@login_required
def dashboard(request):
	# retrive the username for given request
    user = request.user.username
	# Redirect to Admin dashboard
    if user == "admin":
        print("admin")
        total_num_of_emp = total_number_employees() # total number of employees
        emp_present_today = employees_present_today() # total number of employees present today
        this_week_emp_count_vs_date() # for current week graph
        last_week_emp_count_vs_date() # for last week graph
        return render(request, "recognition/admin_dashboard.html", {'total_num_of_emp': total_num_of_emp, 'emp_present_today': emp_present_today})
	# Redirect to Employee dashboard
    else:
        print("not admin")
        user = request.user
		## logic for employee monthly summary report 
        today = datetime.date.today()  # Today's date
        y = datetime.date.today().strftime('%Y')  # current year
        m = datetime.date.today().strftime('%m')  # current month
        num_day = calendar.monthrange(int(y), int(m))  # number of days in current month
        start_dt = date(int(y), int(m), 1)  # first day of current month
        end_dt = datetime.date.today()  # present day
		# retrive the holiday list between  first day of month  to present day
        h = tblHoliday.objects.filter(date__gte=start_dt).filter(date__lt=end_dt) # holiday list object  
        hd = [] # to store holiday date in list object
        for i in h:
            hd.append(str(i.date))
		# retrive data for requested user from present database
        start = tblPresent.objects.filter(user=user)
		# filter database by start date and present date
        start = start.filter(date__gte=start_dt).filter(
            date__lte=end_dt).order_by('-date')
		# find the total mark present for given user
        totalPresent = tblPresent.objects.filter(user=user).filter(
            date__gte=start_dt).filter(present=True).count()
        num_days = int(datetime.date.today().strftime('%d'))
		# find the mark absent for given user
        totalAbsent = int(num_days)-int(totalPresent) # absent variable
        weekoff = 0 # weekoff var
        holiday = 0 # holiday var
		# find the total number of weekoff and holiday between first day of month to present date
        for dt in daterange(start_dt, end_dt):
            y = dt.year
            m = dt.month
            d = dt.day
            day = date(y, m, d)
			# for weekoff
            if calendar.day_name[day.weekday()] == "Sunday":
                weekoff += 1
			# for holiday
            if str(day) in hd:
                holiday += 1
        totalAbsent = totalAbsent-weekoff
        totalSalaryday = totalPresent+weekoff
		
		# for today's report for given user

        status = "" # today's attendance status
        if start.filter(date=today).exists():
            status = "1"
		# today's time in
        ti = tblTime.objects.filter(user=user).filter(
            date=today).filter(out=False).order_by('time')
		# today's time out
        to = tblTime.objects.filter(user=user).filter(
            date=today).filter(out=True).order_by('-time')
		# format database object into date
        if ti:
            ti = ti.first().time.time()
        if to:
            to = to.first().time.time()

        return render(request, 'recognition/employee_dashboard.html', {'user': user, 'ds': start_dt, 'df': end_dt, 'presentDays': totalPresent, 'absentDays': totalAbsent, 'weekoffDays': weekoff, 'salaryDays': totalSalaryday, 'totalDays': num_days, 'holidays': holiday, 'today': today, 'status': status, 'ti': ti, 'to': to})


def daterange(date1, date2):
    for n in range(int((date2 - date1).days)+1):
        yield date1 + timedelta(n)

# for add photo sample to detect the face of employee
@login_required
def add_photos(request):
	# if you are not admin then you can't add sample photos
    if request.user.username != 'admin':
        return redirect('not-authorised')
    if request.method == 'POST':
        # retrive form
        form = usernameForm(request.POST)
        data = request.POST.copy()
        # retrive username from form
        username = data.get('username')
		# for add sample photos , user must be registered in User database
        if username_present(username):
            # create user dataset or update dataset
            create_dataset(username)
            messages.success(request, f'Dataset Created')
            return redirect('add-photos')
        else:
            messages.warning(
                request, f'No such username found. Please register employee first.')
            return redirect('dashboard')
    else:
        form = usernameForm()
        return render(request, 'recognition/add_photos.html', {'form': form})

# for delete the particular user data 
@login_required
def delete_user(request):
    # if not admin then you can't delete user
    if request.user.username != 'admin':
        return redirect('not-authorised')
    if request.method == 'POST':
        # get the form
        form = usernameForm(request.POST)
        data = request.POST.copy()
        # retrive username from form
        username = data.get('username')
        # if a username is present in user database then you can delete it
        if username_present(username):
            User.objects.filter(username=username).delete() # delete recored from user database
            try:
                if has_Details(username): # check if recored is exist
                    tblUserdetails.objects.filter(user=username).delete() # delete recored from user_details database
                if has_Present(username): # check if recored is exist
                    tblPresent.objects.filter(user=username).delete() # delete recored from Present database
                if has_Times(username): # check if recored is exist
                    Times.objects.filter(user=username).delete() # delete recored from Times database
            except:
                print("none")

            messages.success(request, f'User has been deleted..')
            return redirect('delete-user')
        else:
            messages.warning(
                request, f'No such username found. Please register employee first.')
            return redirect('delete-user')
    else:
        form = usernameForm()
        return render(request, 'recognition/delete_user.html', {'form': form})

# this function is use to detect the face of employee for mark in  
def mark_your_attendance(request):
    # set path for important library
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"

    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False
    vs = VideoStream(src=0).start() # start video frame that capture by camera
    sampleNum = 0
    person_name = "" 
    i = 0
    flag = False # when person will detected flag will be set True and loop will be terminate
    frame = []

    while(True):

        # read frame from the videostream
        frame = vs.read()
        # reset the frame size
        frame = imutils.resize(frame, width=800)
        # cv2.rectangle(frame,(100,100),(400,400),(0,255,0))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0) # find all face that are detected in one frame and make a list of all face
        # get the one by one face from the list
        for face in faces:
            print("INFO : inside for loop")
            i += 1
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame, gray_frame, face)
            # make rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # to get the predicted person with percentage
            (pred, prob) = predict(face_aligned, svc)

            if(pred != [-1]):
                # fetch the person name 
                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                pred = person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred, 0) + 1
                if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
                    count[pred] = 0
                # if person count > 0 then
                else:
                    # person is detected with name
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred, 0) + 1
                    print(pred, present[pred], count[pred])
                    flag = True # detection flag is set true
                    snap = frame
                # put detected person name on video screen
                cv2.putText(frame, str(person_name) + str(prob), (x+6,
                            y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            else:
                person_name = "unknown"
                cv2.putText(frame, str(person_name), (x+6, y+h-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        # Showing the image in another window
        # Creates a window with window name "Face" and with the image img
        cv2.imshow("Mark Attendance - In - Press q to exit", frame)
        # Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        # cv2.waitKey(1)
        # To get out of the loop
        key = cv2.waitKey(50) & 0xFF
        # press q to close  the video capturing frame
        if(key == ord("q")):
            break
        if flag:
            break
    # stop the video capturing
    vs.stop()
    # filename = 'C:/Users/patel/Desktop/Face-Recognition-system  V1.3/snap/' + person_name + \
    #	str(datetime.datetime.timestamp(datetime.datetime.now()))+'.jpg'
    #cv2.imwrite(filename, img=frame)
    cv2.destroyAllWindows()

    if person_name != 'unknown':
        # call this function to update attendance of detected person
        update_attendance_in_db_in(present)
        # write a message for feedback  to person
        confirmationMessage = "Welcome !!!"+person_name
        return render(request, "recognition/confirmation.html", {'confirmationMessage': confirmationMessage})

    return redirect('home')

# this function is use to detect the face of employee for mark out
def mark_your_attendance_out(request):
    # set path for important library
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path = "face_recognition_data/svc.sav"

    with open(svc_save_path, 'rb') as f:
        svc = pickle.load(f)
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1, 128))
    # get number of faces from frame
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False
    vs = VideoStream(src=0).start()  # start video frame that capture by camera

    sampleNum = 0
    person_name = ""
    while(True):
        # read frame from the videostream
        frame = vs.read()
        # reset the frame size
        frame = imutils.resize(frame, width=800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)
        flag = False  # when person will detected flag will be set True and loop will be terminate

        for face in faces:
            print("INFO : inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame, gray_frame, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            (pred, prob) = predict(face_aligned, svc)
            # if pred is not (-1) then
            if(pred != [-1]):
                # get the person name
                person_name = encoder.inverse_transform(np.ravel([pred]))[0]
                pred = person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred, 0) + 1

                if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
                    count[pred] = 0
                # if person count in > 0 then
                else:
                    # person is detected with name
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred, 0) + 1
                    print(pred, present[pred], count[pred])
                    flag = True
                # show person name with probability percentage
                cv2.putText(frame, str(person_name) + str(prob), (x+6,
                            y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            else:
                person_name = "unknown"
                cv2.putText(frame, str(person_name), (x+6, y+h-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # cv2.putText()
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            # cv2.waitKey(50)

        # Showing the image in another window
        # Creates a window with window name "Face" and with the image img
        cv2.imshow("Mark Attendance- Out - Press q to exit", frame)
        # Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        # cv2.waitKey(1)
        # To get out of the loop
        key = cv2.waitKey(50) & 0xFF
        if(key == ord("q")):
            break
        if flag:
            break

    # Stoping the videostream
    vs.stop()

    # destroying all the windows
    cv2.destroyAllWindows()
    if person_name != "unknown":
        update_attendance_in_db_out(present)
        confirmationMessage = "Bye !!!"+person_name
        return render(request, "recognition/confirmation.html", {'confirmationMessage': confirmationMessage})
    return render(request, "recognition/home.html")


# for train your model , it's required when you add any new photo samples
@login_required
def train(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    # import important directory
    training_dir = "face_recognition_data/training_dataset"
    count = 0 # counter variable
    for person_name in os.listdir(training_dir):
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            count += 1
    X = []
    personNameList = []
    i = 0
    # fetch person name one by one from directory
    for person_name in os.listdir(training_dir):
        print(str(person_name))
        # set current directory path with person name directory
        curr_directory = os.path.join(training_dir, person_name)
        if not os.path.isdir(curr_directory):
            continue
        for imagefile in image_files_in_folder(curr_directory):
            print(str(imagefile))
            image = cv2.imread(imagefile)
            try:
                X.append((face_recognition.face_encodings(image)[0]).tolist())
                personNameList.append(person_name) # append the person name 
                i += 1
            except:
                print("removed")
                os.remove(imagefile)
    # convert personNameList into numpy array
    targets = np.array(personNameList)
    encoder = LabelEncoder()
    # fit data in model
    encoder.fit(personNameList)
    personNameList = encoder.transform(personNameList)
    X1 = np.array(X)
    print("shape: " + str(X1.shape))
    # save encoder classes file in folder
    np.save('C:/Users/patel/Desktop/Face-Recognition-system  V1.3/face_recognition_data/classes.npy', encoder.classes_)
    svc = SVC(kernel='linear', probability=True)
    # fit data in model
    svc.fit(X1, personNameList)
    svc_save_path = "face_recognition_data/svc.sav"
    with open(svc_save_path, 'wb') as f:
        pickle.dump(svc, f)
    # for visualize the data on scatter graph
    vizualize_Data(X1, targets)
    messages.success(request, f'Training Complete.')
    return render(request, "recognition/train.html")


@login_required
def not_authorised(request):
    return render(request,'recognition/not_authorised.html')


# View attendance by date for admin
@login_required
def view_attendance_date(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    qs = None
    time_qs = None
    present_qs = None

    if request.method == 'POST':
        form = DateForm(request.POST)
        # check the validity of form
        if form.is_valid():
            date = form.cleaned_data.get('date') # given date
            date1 = form.cleaned_data.get('date')
            time_qs = tblTime.objects.filter(date=date)  # Query on Time database by date
            present_qs = tblPresent.objects.filter(date=date)  # Query on Present database by date

            if(len(time_qs) > 0 or len(present_qs) > 0):
                qs = hours_vs_employee_given_date(present_qs, time_qs)
                y = date1.strftime('%Y')  # year
                m = date1.strftime('%m')  # month
                d = date1.strftime('%d')  # day

                userList = [] # empty list 
                # append username in list u
                for i in qs:
                    userList.append(i.user.username)

                # empty list
                # used to store user attributes by date
                userAttendanceList = [] 

                # to store data of user which are present at given date
                for i in qs:
                    if i.user.username in userList:
                        a =attendance_list(str(i.date), i.user.username, "P",
                               i.time_in, i.time_out, i.hours, i.break_hours)
                        userAttendanceList.append(a)

                # all employees username
                total_emp = User.objects.order_by('-username')
                # to store data of user which are absent at given date
                for i in total_emp:
                    if i.username not in userList:
                        a = attendance_list(date, i.username, "A", "-", "-", "-", "-")
                        userAttendanceList.append(a)
                # list sort by name
                userAttendanceList.sort(key=lambda x: str(x.date))
                return render(request, 'recognition/view_attendance_date.html', {'form': form, 'userAttendanceList': userAttendanceList})
            else:
                messages.warning(request, f'No records for selected date.')
                return redirect('view-attendance-date')
    else:

        form = DateForm()
        return render(request, 'recognition/view_attendance_date.html', {'form': form, 'qs': qs})

# View attendance of employee by given range of date
@login_required
def view_attendance_employee(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    time_qs = None
    present_qs = None
    qs = None

    if request.method == 'POST':
        form = UsernameAndDateForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username') # get the username from form
            if username_present(username):
                u = User.objects.get(username=username) # username 
                time_qs = tblTime.objects.filter(user=u)  # query on Time database by given username
                present_qs = tblPresent.objects.filter(user=u)  # query on Present database by given username
                date_from = form.cleaned_data.get('date_from')  # starting date
                date_to = form.cleaned_data.get('date_to')  # ending date

                # check date validity
                if date_to < date_from:
                    messages.warning(request, f'Invalid date selection.')
                    return redirect('view-attendance-employee')

                else:
                    # query to find Time data for given date range
                    time_qs = time_qs.filter(date__gte=date_from).filter(
                        date__lte=date_to).order_by('-date')
                    # query to find Present data for given date range                        
                    present_qs = present_qs.filter(date__gte=date_from).filter(present=True).filter(
                        date__lte=date_to).order_by('-date')

                    if (len(time_qs) > 0 or len(present_qs) > 0):
                        attendanceList = hours_vs_date_given_employee(
                            present_qs, time_qs, admin=True)
                        userAttendanceList = []
                        days = []
                        x = 0
                        for i in attendanceList:
                            if i.present:
                                a = attendance_list(str(i.date), i.user.username, "P",i.time_in, i.time_out, i.hours, i.break_hours)
                            #else:
                             #   a = attendance_list(str(i.date), i.user.username, "A",i.time_in, i.time_out, i.hours, i.break_hours)
                            userAttendanceList.append(a)  # append the list of employee data
                            days.append(str(a.date))
                        #num_days = int(datetime.date.today().strftime('%d')) # number of days in
                        h = tblHoliday.objects.filter(date__gte=date_from).filter(date__lt=date_to) # query on holoday database by give date range
                        hd = [] # emply list to store holiday date
                        for i in h:
                            hd.append(str(i.date))

                        # for starting date
                        startDate_year = date_from.strftime('%Y') # year
                        startDate_month = date_from.strftime('%m') # month
                        startDate_day = date_from.strftime('%d') # day
                        # for ending date
                        endDate_year = date_to.strftime('%Y') # year
                        endDate_month = date_to.strftime('%m') # month
                        endDate_day= date_to.strftime('%d') # day

                        start_dt = date(int(startDate_year), int(startDate_month), int(startDate_day))
                        end_dt = date(int(endDate_year), int(endDate_month), int(endDate_day))
                        i = 0
                        ho = 0 # counter for holiday
                        wo = 0 # counter for weekoff
                        for dt in daterange(start_dt, end_dt):
                            y = dt.year
                            m = dt.month
                            d = dt.day
                            day = str(date(y, m, d)) # string object for day
                            d = date(y, m, d) # date object of day
                            
                            if day not in days:
                                # for count weekoff 
                                if calendar.day_name[d.weekday()] == "Sunday":
                                    a = attendance_list(str(day), u, "WO",
                                           "-", "-", "-", "-")
                                    wo=wo+1
                                # for count holiday
                                elif day in hd:
                                    a = attendance_list(str(day), u, "HO",
                                           "-", "-", "-", "-")
                                    ho += 1
                                # for count absent
                                else:
                                    a = attendance_list(str(day), u, "A",
                                           "-", "-", "-", "-")
                                userAttendanceList.append(a)
                            i = i+1
                    
                        userAttendanceList.sort(key=lambda x: str(x.date))
                        num_days = end_dt-start_dt # num of days between start date and end date
                        num_days = num_days.days+1

                        totalPresent=len(attendanceList) # total present 
                        totalAbsent = int(num_days)-int(totalPresent)
                    
                        totalAbsent = totalAbsent-wo # count number of absent = total absent - holiday
                        salaryDay = totalPresent+wo # salary day = present day + weekoff

                        return render(request, 'recognition/view_attendance_employee.html', {'form': form, 'qs': userAttendanceList, 'totalPresent': totalPresent, 'totalAbsent': totalAbsent, 'totalWeekOff': wo, 'totalSalaryDay': salaryDay, 'totalHoliday': ho})
                    else:
                        messages.warning(
                            request, f'No records for selected duration.')
                        notfound = 1
                        return render(request, 'recognition/view_attendance_employee.html', {'form': form, 'nf': notfound})

            else:
                print("invalid username")
                messages.warning(request, f'No such username found.')
                return redirect('view-attendance-employee')

    else:
        
        form = UsernameAndDateForm()
        return render(request, 'recognition/view_attendance_employee.html', {'form': form})


# class for the data of employee
class attendance_list:
    def __init__(self, date, employeeName, attendanceStatus, timeIn, timeOut, totalHours, breakHours):
        self.date = date
        self.employeeName = employeeName
        self.attendanceStatus = attendanceStatus
        self.timeIn = timeIn
        self.timeOut = timeOut
        self.totalHours = totalHours
        self.breakHours = breakHours

# used for view the attendance of user (for user only)
# employee can view their attendance with date range
# also can view attendance summary with pie-chart 
@login_required
def view_my_attendance_employee_login(request):
    if request.user.username == 'admin':
        return redirect('not-authorised')
    qs = None
    time_qs = None
    present_qs = None
    u = request.user # retrive the username

    if request.method == 'POST':
        form = DateForm_2(request.POST)
        # check the validity of form
        if form.is_valid():
            # query to retrive recordes of given user from the Times database
            time_qs = tblTime.objects.filter(user=u)
            # query to retrive recordes of given user from the Present database
            present_qs = tblPresent.objects.filter(user=u)
            # starting date for date range
            date_from = form.cleaned_data.get('date_from')
            # ending date for date range
            date_to = form.cleaned_data.get('date_to')
            if date_to < date_from:
                messages.warning(request, f'Invalid date selection.')
                return redirect('view-my-attendance-employee-login')
            else:
                # set recordes order by date
                time_qs = time_qs.filter(date__gte=date_from).filter(
                    date__lte=date_to).order_by('-date')
                # set recordes order by date
                present_qs = present_qs.filter(date__gte=date_from).filter(
                    date__lte=date_to).order_by('-date')
                # if recordes found then and only then
                if (len(time_qs) > 0 or len(present_qs) > 0):
                    qs = hours_vs_date_given_employee(
                        present_qs, time_qs, admin=False)
                    userAttendancelist = []
                    days = [] # days list
                    for i in qs:
                        if i.present:
                            a = attendance_list(str(i.date), i.user.username, "P",
                                   i.time_in, i.time_out, i.hours, i.break_hours)
                        #else:
                            #a = attendance_list(str(i.date), i.user.username, "A",
                             #      i.time_in, i.time_out, i.hours, i.break_hours)
                        userAttendancelist.append(a)
                        days.append(str(a.date))
                        # print("\n",a.d)
                    num_days = int(datetime.date.today().strftime('%d')) # number of days from month 1st date
                    holiday = tblHoliday.objects.filter(
                        date__gte=date_from).filter(date__lte=date_to)
                    holidayList = []
                    for i in holiday:
                        holidayList.append(str(i.date))

                    print("holiday list :", holidayList)
                    startDate_year = date_from.strftime('%Y') # year
                    startDate_month = date_from.strftime('%m') # month
                    startDate_day = date_from.strftime('%d') # day

                    endDate_year = date_to.strftime('%Y') # year
                    endDate_month= date_to.strftime('%m') # month
                    endDate_day= date_to.strftime('%d') # day

                    start_dt = date(int(startDate_year), int(startDate_month), int(startDate_day)) # starting date
                    end_dt = date(int(endDate_year), int(endDate_month), int(endDate_day)) # ending date
                   # i = 0 # counter
                    ho = 0 # counter for holiday
                    wo = 0 # weekoff counter
                    
                    print("number of days ",len(qs))
                    for dt in daterange(start_dt, end_dt):
                        y = dt.year
                        m = dt.month
                        d = dt.day
                        day = str(date(y, m, d)) # string var of date form given range
                        d = date(y, m, d) # date object
                        # check date in above days list
                        if day not in days:
                            # check for weekoff
                            if calendar.day_name[d.weekday()] == "Sunday":
                                a = attendance_list(str(day), u, "WO", "-", "-", "-", "-")
                                wo+=1
                                #c
                            # check for holiday
                            elif day in holidayList:
                                a = attendance_list(str(day), u, "HO", "-", "-", "-", "-")
                                ho+=1
                            # check for Absent
                            else:
                                a = attendance_list(str(day), u, "A", "-", "-", "-", "-")
                            userAttendancelist.append(a)
                        #i = i+1
            
                    userAttendancelist.sort(key=lambda x: str(x.date))
                    totalPresent = present_qs.filter(date__gte=start_dt).filter(date__lte=end_dt).count() # count present days for user
                    n = end_dt-start_dt # number of days for given date range
                    n = n.days+1
                    totalAbsent = int(n)-int(totalPresent)-wo-ho # number of absent days
                    salaryDay = int(totalPresent)+ho+wo # number of salary days

                    return render(request, 'recognition/view_my_attendance_employee_login.html', {'form': form, 'qs': qs, 'userAttendancelist': userAttendancelist, 'totalPresent': totalPresent, 'totalAbsent': totalAbsent, 'totalWeekOff': wo, 'totalHolidays': ho, 'totalSalaryDays': salaryDay})
                else:
                    messages.warning(
                        request, f'No records for selected duration.')
                    return redirect('view-my-attendance-employee-login')
    else:
        form = DateForm_2()
        return render(request, 'recognition/view_my_attendance_employee_login.html', {'form': form, 'qs': qs})

# view employee profile
# employee can view his/her profile after login 
@login_required
def view_my_profile(request):

    users = request.user
    q = 0 # flag
    p = 0 # flag
    u = tblUserdetails.objects.filter(user=users).exists()
    if u:
        userDetails = tblUserdetails.objects.get(user=users)
        p = 1 # set if user data present in user_details database
    else:
        print("not present")
        # print(users)
        userDetails = users
        q = 1 # set if user data not present in user_details database

    return render(request, 'recognition/user_profile.html', {'userDetails': userDetails, 'q': q, 'p': p})

# class for save the user details
class user:
    def __init__(self, user, email, contact, department):
        self.user = user
        self.email = email
        self.contact = contact
        self.department = department

# view employee details in admin side
# admin can view all employee details 
@login_required
def view_employee(request):
    # merge the user details from both databases
    userDb = tblUserdetails.objects.order_by('user') # query in user database
    userDetailsdb = User.objects.order_by('username')  # query in user_details database

    userRecordList = []
    userNameList = []
    userNameSet = {""}

    # store data in list from the user database
    for i in userDb:
        # fetch user details and save it
        userRecord = user(i.user, i.email, i.contact, i.department)
        # append user attendance list
        userRecordList.append(userRecord)
        userName = userRecord.user
        print("in 1st:", userName)
        userNameList.append(userName)
    for i in userNameList:
        userNameSet.add(i.username)
    # store data in list from the user_details database
    for i in userDetailsdb:
        if i.username in userNameSet:
            print('')
        else:
            # if user recorde not in user_details
            print("not in list", i.username, userNameSet)
            userRecord = user(i.username, '', '', '')
            userRecordList.append(userRecord)

    return render(request, 'recognition/view_employees.html', {'qs': userDb, 'userRecordList': userRecordList})

# used for edit the profile of user
# user can edit his/her information and can save it
def editprofile(request):
    if request.method == "POST":
        # get the user name
        # retrive updated email / contact / department
        user = request.user
        email = request.POST.get('email')
        contact = request.POST.get('contact')
        dept = request.POST.get('department')

    # if user is present in User_details database 
    if tblUserdetails.objects.filter(user=user).exists():
        # create object of User_details database with updated information
        tblUserdetails.objects.filter(user=user).update(
            email=email, contact=contact, department=dept)
        # update the user information in database
        userDetails = tblUserdetails.objects.get(user=user)
        print("exist")
    # if user is not present in User_details database , then create new database object and then save it
    else:
        tblUserdetails.objects.create(
            user=user, email=email, contact=contact, department=dept)
        userDetails = tblUserdetails.objects.get(user=user)
        print("not exist")
        print("profile edited")

    return render(request, 'recognition/user_profile.html', {'p': 1, 'q': 0, 'userDetails': userDetails})

# add holiday from admin side
# admin can add holidy name and holiday date in database
@login_required
def holiday(request):
    today = datetime.date.today()
    holidayList = tblHoliday.objects.filter(date__gte=today).order_by('date')
    
    if request.user.username == 'admin':
        if request.method == "POST":    
            form = Holiday_form(request.POST)
            # check validity of form
            if form.is_valid():
                # get the name and date of holiday
                name = form.cleaned_data.get('name')
                date = form.cleaned_data.get('date')
                # add data in holiday database
                tblHoliday.objects.create(name=name, date=date)
                return render(request, 'recognition\holiday.html', {'form': form, 'holidayList': holidayList})
        else:
            form = Holiday_form()
            return render(request, 'recognition\holiday.html', {'form': form, 'holidayList': holidayList})
    else:
            return redirect('not-authorised')

# display holiday in user side
@login_required
def employee_holiday(request):
    # get the present date 
    today = datetime.date.today()
    # retrive the holiday name that will come after the  current date only
    holidayList = tblHoliday.objects.filter(date__gte=today).order_by('date')
    if(request.user.username != 'admin'):
        return render(request, 'recognition\employee_holiday.html', {'holidayList': holidayList})
    else:
        return render('not_authorised')
