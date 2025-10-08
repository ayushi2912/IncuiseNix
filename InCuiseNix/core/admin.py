from django.contrib import admin
from .models import Course, Video, Enrollment, Note

admin.site.register(Course)
admin.site.register(Video)
admin.site.register(Enrollment)
admin.site.register(Note)

