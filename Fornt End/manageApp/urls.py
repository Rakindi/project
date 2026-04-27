from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', view=views.index,name="Index"),
    path('signup/',view=views.signup,name="signup"),
    path('register/',view=views.register,name="register"),
    path('userlogins/',view=views.userlogin,name="userlogin"),
    path('upload/',view=views.upload,name="upload"),
    path('uploadscreen/',view=views.uploadscreen,name="uploadscreen"),
    path('logout/',view=views.logout,name="logout"),
    path('profile/',view=views.profile,name="profile")
       
]
