from django.db import models
from django.contrib.auth.models import User

class Course(models.Model):
    course_id = models.CharField(max_length=50, unique=True)
    title = models.CharField(max_length=200)
    description = models.TextField()
    image_url = models.URLField(max_length=200)

    def __str__(self):
        return self.title

class Video(models.Model):
    video_id = models.CharField(max_length=50, unique=True)
    title = models.CharField(max_length=200)
    video_url = models.URLField(max_length=200)
    course = models.ForeignKey(Course, related_name='videos', on_delete=models.CASCADE)

    def __str__(self):
        return self.title

class Enrollment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    enrolled_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'course')

    def __str__(self):
        return f"{self.user.username} enrolled in {self.course.title}"
    
class Note(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    content = models.TextField()
    video_timestamp = models.PositiveIntegerField(help_text="Timestamp in seconds")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Note by {self.user.username} for {self.video.title} at {self.video_timestamp}s"
    

