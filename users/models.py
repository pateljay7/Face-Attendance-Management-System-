from django.db import models
from django.contrib.auth.models import User

import datetime
# Create your models here.
# Present database
class tblPresent(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	date=models.DateField(default=datetime.date.today)
	present=models.BooleanField(default=False)
# Time database to save time in and time out details 
class tblTime(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	date=models.DateField(default=datetime.date.today)
	time=models.DateTimeField(null=True,blank=True)
	out=models.BooleanField(default=False)
	image=models.BinaryField(blank=True)
# user details database to save user details
class tblUserdetails(models.Model):
	user=models.ForeignKey(User,on_delete=models.CASCADE)
	email=models.CharField(max_length=120)
	contact=models.CharField(max_length=10)
	department=models.CharField(max_length=100)
# holiday database to save holiday details with name and date
class tblHoliday(models.Model):
	name=models.CharField(max_length=30)
	date=models.DateField()

	

