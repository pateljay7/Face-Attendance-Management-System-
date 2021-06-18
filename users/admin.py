from django.contrib import admin
from .models import tblTime,tblPresent,tblUserdetails,tblHoliday

# Register your models here.
admin.site.register(tblTime)
admin.site.register(tblPresent)
admin.site.register(tblUserdetails)
admin.site.register(tblHoliday)