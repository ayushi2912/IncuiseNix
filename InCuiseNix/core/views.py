# File: core/views.py

import os
import json
import re
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from .forms import SignUpForm, LoginForm, NoteForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from .models import Enrollment, Course, Video, Note
from django.template.defaultfilters import date as _date
from django.conf import settings
from transcripts.services import get_transcript

# --- IMPORTS FOR LOCAL RAG ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser


# --- User Authentication and Static Pages ---
def custom_404_view(request, exception):
    return redirect('home')

def home(request):
    return render(request, 'core/home.html')

def about_view(request):
    return render(request, 'core/about.html')

def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = SignUpForm()
    return render(request, 'core/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
    else:
        form = LoginForm()
    return render(request, 'core/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('home')

# --- Core Platform Views ---
@login_required
def dashboard_view(request):
    enrolled_courses = Enrollment.objects.filter(user=request.user).select_related('course')
    context = {
        'enrolled_courses': [enrollment.course for enrollment in enrolled_courses]
    }
    return render(request, 'core/dashboard.html', context)

@login_required
def courses_list_view(request):
    enrolled_course_ids = Enrollment.objects.filter(user=request.user).values_list('course__id', flat=True)
    all_courses = Course.objects.all()
    context = {
        'all_courses': all_courses,
        'enrolled_course_ids': set(enrolled_course_ids),
    }
    return render(request, 'core/courses_list.html', context)

@login_required
@require_POST
def enroll_view(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    Enrollment.objects.get_or_create(user=request.user, course=course)
    return redirect('dashboard')

@login_required
def video_player_view(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    is_enrolled = Enrollment.objects.filter(user=request.user, course=course).exists()
    if not is_enrolled:
        return redirect('dashboard')
    all_videos = course.videos.all().order_by('id')
    video_id_from_url = request.GET.get('vid')
    if video_id_from_url:
        current_video = get_object_or_404(Video, video_id=video_id_from_url, course=course)
    else:
        current_video = all_videos.first()
    notes = Note.objects.filter(user=request.user, video=current_video)
    form = NoteForm()
    transcript = get_transcript(current_video.video_url)
    context = {
        'course': course,
        'all_videos': all_videos,
        'current_video': current_video,
        'notes': notes,
        'form': form,
        'transcript': transcript,
    }
    return render(request, 'core/video_player.html', context)

# --- AJAX Views for Note Management ---
@login_required
@require_POST
def add_note_view(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    form = NoteForm(request.POST)
    if form.is_valid():
        note = form.save(commit=False)
        note.user = request.user
        note.video = video
        note.save()
        return JsonResponse({'status': 'success', 'note': {'id': note.id, 'content': note.content, 'timestamp': note.video_timestamp, 'created_at': _date(note.created_at, 'd M Y, H:i')}})
    return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)

@login_required
@require_POST
def edit_note_view(request, note_id):
    note = get_object_or_404(Note, id=note_id, user=request.user)
    data = json.loads(request.body)
    new_content = data.get('content')
    if new_content:
        note.content = new_content
        note.save()
        return JsonResponse({'status': 'success', 'message': 'Note updated successfully.'})
    return JsonResponse({'status': 'error', 'message': 'Content cannot be empty.'}, status=400)

@login_required
@require_POST
def delete_note_view(request, note_id):
    note = get_object_or_404(Note, id=note_id, user=request.user)
    note.delete()
    return JsonResponse({'status': 'success', 'message': 'Note deleted successfully.'})


# --- NEW HELPER FUNCTION for RAG ---
def parse_time_to_seconds(time_str: str):
    total_seconds = 0
    patterns = [
        r'(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?',
        r'(\d+):(\d+):(\d+)', r'(\d+):(\d+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, time_str)
        if match:
            groups = [int(g) if g else 0 for g in match.groups()]
            if pattern == r'(\d+):(\d+):(\d+)':
                h, m, s = groups; return h * 3600 + m * 60 + s
            elif pattern == r'(\d+):(\d+)':
                m, s = groups; return m * 60 + s
            elif pattern == r'(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?':
                h, m, s = groups; total_seconds = h * 3600 + m * 60 + s
                return total_seconds if total_seconds > 0 else None
    return None


# --- UPDATED AI Assistant View with LOCAL RAG Pipeline ---
@login_required
def gemini_assistant_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method.'}, status=405)

    try:
        google_api_key = getattr(settings, "GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("CRITICAL ERROR: GEMINI_API_KEY/GOOGLE_API_KEY is not set.")
            return JsonResponse({'error': "AI service is not configured correctly (missing API key)."}, status=500)

        data = json.loads(request.body)
        user_question = data.get('prompt')
        if not user_question:
            return JsonResponse({'error': 'A prompt is required.'}, status=400)
        
        PERSIST_DIRECTORY = os.path.join(settings.BASE_DIR.parent, "chroma_db_memory")
        if not os.path.exists(PERSIST_DIRECTORY):
             return JsonResponse({'response': "The AI memory has not been built yet. Please ask an administrator to build it."})

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)

        # --- REVISED RETRIEVER LOGIC ---
        # Using Maximal Marginal Relevance (MMR) to get more diverse and relevant results.
        # It fetches more documents initially (fetch_k=20) and then selects the best 5 (k=5) that are different from each other.
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 5, 'fetch_k': 20}
        )

        template = """
        You are an AI assistant for the InCuiseNix e-learning platform.
        Answer the user's question using ONLY the context provided below from the video transcript.
        If the context does not contain the answer, say "The transcript does not have enough information to answer that."

        CONTEXT: {context}
        QUESTION: {question}
        ANSWER:
        """
        prompt_template = PromptTemplate.from_template(template)
        
        # Use the Gemini Chat Model for the FINAL answer generation
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2, google_api_key=google_api_key)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(user_question)
        return JsonResponse({'response': answer})

    except Exception as e:
        print(f"An error occurred in the Gemini assistant view: {e}")
        return JsonResponse({'error': 'An error occurred while communicating with the AI service.'}, status=500)


# --- ROADMAP VIEW ---
@login_required
def roadmap_view(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    if not Enrollment.objects.filter(user=request.user, course=course).exists():
        return JsonResponse({'error': 'You are not enrolled in this course.'}, status=403)
    course_data = {
        'title': course.title,
        'description': course.description
    }
    return JsonResponse(course_data)
