# core/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # Main site pages
    path('', views.home, name='home'),
    path('about/', views.about_view, name='about'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Course and dashboard pages
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('courses/', views.courses_list_view, name='courses_list'),
    path('enroll/<int:course_id>/', views.enroll_view, name='enroll'),
    path('course/<int:course_id>/', views.video_player_view, name='video_player'),
    
    # URLS FOR NOTE MANAGEMENT (This is the crucial part)
    path('note/add/<int:video_id>/', views.add_note_view, name='add_note'),
    path('note/edit/<int:note_id>/', views.edit_note_view, name='edit_note'),
    path('note/delete/<int:note_id>/', views.delete_note_view, name='delete_note'),

    # URL for AI Assistant
    path('assistant/', views.gemini_assistant_view, name='gemini_assistant'),

    # ADD THIS NEW URL FOR THE COURSE DETAILS
    path('roadmap/<int:course_id>/', views.roadmap_view, name='roadmap'),]
handler404 = 'core.views.custom_404_view'
