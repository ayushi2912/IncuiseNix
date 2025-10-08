import json
from django.core.management.base import BaseCommand
from django.db import connection
from core.models import Course, Video, Note, Enrollment

def get_youtube_id(url):
    """Extracts the YouTube video ID from a URL."""
    if "embed/" in url:
        return url.split("embed/")[1].split("?")[0]
    elif "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    return None

class Command(BaseCommand):
    help = 'Loads courses and videos from a JSON file and resets table IDs.'

    def handle(self, *args, **kwargs):
        self.stdout.write('Clearing all existing course-related data...')
        # Delete in an order that respects foreign key constraints
        Note.objects.all().delete()
        Enrollment.objects.all().delete()
        Video.objects.all().delete()
        Course.objects.all().delete()
        self.stdout.write(self.style.SUCCESS('Successfully cleared old data.'))

        # Reset the auto-increment counters for the tables
        self.stdout.write('Resetting database auto-increment counters...')
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE core_note AUTO_INCREMENT = 1;")
            cursor.execute("ALTER TABLE core_enrollment AUTO_INCREMENT = 1;")
            cursor.execute("ALTER TABLE core_video AUTO_INCREMENT = 1;")
            cursor.execute("ALTER TABLE core_course AUTO_INCREMENT = 1;")
        self.stdout.write(self.style.SUCCESS('Successfully reset counters.'))

        # Proceed with loading the new data
        with open('courses.json', 'r') as f:
            courses_data = json.load(f)

            for course_data in courses_data:
                course, created = Course.objects.get_or_create(
                    course_id=course_data['course_id'],
                    defaults={
                        'title': course_data['title'],
                        'description': course_data['description'],
                        'image_url': course_data.get('image_url', '')
                    }
                )
                
                if created:
                    self.stdout.write(f'Created course: {course.title}')

                for video_data in course_data['videos']:
                    youtube_id = get_youtube_id(video_data['video_url'])
                    
                    if youtube_id:
                        Video.objects.get_or_create(
                            video_id=youtube_id,
                            defaults={
                                'course': course,
                                'title': video_data['title'],
                                'video_url': video_data['video_url']
                            }
                        )
                self.stdout.write(self.style.SUCCESS(f'Successfully loaded videos for {course.title}'))

        self.stdout.write(self.style.SUCCESS('Finished loading all courses and videos.'))