# core/management/commands/populate_transcripts.py

import os
import csv
from django.core.management.base import BaseCommand
from django.conf import settings
from core.models import Video, Transcript

class Command(BaseCommand):
    help = 'Loads transcripts from CSV files directly within the media directory.'

    def handle(self, *args, **kwargs):
        media_dir = settings.MEDIA_ROOT
        self.stdout.write(self.style.SUCCESS("--- Starting Transcript Population ---"))
        self.stdout.write(f"Searching for transcript files in: {media_dir}")

        if not os.path.exists(media_dir):
            self.stdout.write(self.style.ERROR('Your media directory (MEDIA_ROOT) does not exist.'))
            return

        # Find all CSV files directly in the media directory
        csv_files = [f for f in os.listdir(media_dir) if f.endswith('.csv')]

        if not csv_files:
            self.stdout.write(self.style.WARNING('No .csv transcript files were found in the media directory.'))
            return
            
        self.stdout.write(f"Found {len(csv_files)} CSV file(s) to process.")

        for filename in csv_files:
            try:
                # Extract the video_id from filenames like 'VIDEO_ID_transcript.csv'
                if '_transcript.csv' not in filename:
                    self.stdout.write(self.style.WARNING(f'  Skipping file with incorrect name format: {filename}'))
                    continue

                video_id_from_filename = filename.replace('_transcript.csv', '')
                
                # Find the video in the database that matches the extracted ID
                video = Video.objects.get(video_id=video_id_from_filename)
                
                # Clear old transcripts to avoid duplicates
                Transcript.objects.filter(video=video).delete()

                file_path = os.path.join(media_dir, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip the header row

                    transcripts_to_create = [
                        Transcript(video=video, start=float(row[0]), content=row[1]) for row in reader if row
                    ]
                    
                    if transcripts_to_create:
                        Transcript.objects.bulk_create(transcripts_to_create)
                        self.stdout.write(self.style.SUCCESS(f'  > Successfully loaded {len(transcripts_to_create)} transcript lines for video: {video.title}'))
                    else:
                        self.stdout.write(self.style.WARNING(f'  > No transcript lines found in file: {filename}'))

            except Video.DoesNotExist:
                self.stdout.write(self.style.WARNING(f'  > WARNING: A video with video_id "{video_id_from_filename}" was not found in the database.'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'  > ERROR: An unexpected error occurred with file {filename}: {e}'))
        
        self.stdout.write(self.style.SUCCESS("\n--- Script Finished ---"))