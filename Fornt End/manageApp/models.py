from django.db import models


class UserModels(models.Model):
    userid=models.IntegerField(auto_created=True,default=0)
    name=models.CharField(default="",max_length=100)
    mobile=models.CharField(default="",max_length=100)
    email=models.CharField(default="",max_length=100)
    password=models.CharField(default="",max_length=100)

class LogStorage(models.Model):
    userid=models.IntegerField(default=0)
