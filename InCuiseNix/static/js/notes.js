import * as api from './modules/note-api.js';
import * as ui from './modules/note-ui.js';
import { formatTimestamp } from './modules/utils.js';

document.addEventListener('DOMContentLoaded', function() {
    const addNoteModalEl = document.getElementById('addNoteModal');
    const addNoteModal = new bootstrap.Modal(addNoteModalEl);
    const addNoteForm = document.getElementById('addNoteForm');
    const editNoteModalEl = document.getElementById('editNoteModal');
    const editNoteModal = new bootstrap.Modal(editNoteModalEl);
    const editNoteForm = document.getElementById('editNoteForm');
    const editNoteContent = document.getElementById('editNoteContent');
    const notesList = document.getElementById('notes-list-container');
    const notePopupOverlay = document.getElementById('note-popup-overlay');
    const popupCloseBtn = document.getElementById('popup-close-btn');

    // Format all timestamps on page load
    document.querySelectorAll('.note-timestamp').forEach(el => {
        const seconds = parseInt(el.dataset.seconds, 10);
        if (!isNaN(seconds)) {
            el.textContent = `Timestamp: ${formatTimestamp(seconds)}`;
        }
    });

    addNoteModalEl.addEventListener('show.bs.modal', function() {
        if (window.videoPlayer) {
            window.videoPlayer.pause();
            const currentTime = Math.round(window.videoPlayer.currentTime);
            addNoteForm.querySelector('input[name="video_timestamp"]').value = currentTime;
        }
    });

    addNoteForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const videoId = this.dataset.videoId;
        const formData = new FormData(this);

        try {
            const data = await api.addNote(videoId, formData);
            if (data.status === 'success') {
                ui.addNoteToUI(data.note);
                addNoteForm.reset();
                addNoteModal.hide();
                if (window.videoPlayer) window.videoPlayer.play();
            }
        } catch (error) {
            console.error('Error adding note:', error);
        }
    });

    editNoteForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const noteId = this.dataset.noteId;
        const newContent = editNoteContent.value;

        try {
            const data = await api.editNote(noteId, newContent);
            if (data.status === 'success') {
                ui.updateNoteInUI(noteId, newContent);
                editNoteModal.hide();
            }
        } catch (error) {
            console.error('Error updating note:', error);
        }
    });

    notesList.addEventListener('click', async function(event) {
        const target = event.target;
        const editButton = target.closest('.btn-edit-note');
        const deleteButton = target.closest('.btn-delete-note');
        const noteCard = target.closest('.note-card');

        if (editButton) {
            editNoteContent.value = editButton.dataset.noteContent;
            editNoteForm.dataset.noteId = editButton.dataset.noteId;
            editNoteModal.show();
            return;
        }

        if (deleteButton) {
            const noteId = deleteButton.dataset.noteId;
            if (confirm('Are you sure you want to delete this note?')) {
                try {
                    const data = await api.deleteNote(noteId);
                    if (data.status === 'success') {
                        ui.removeNoteFromUI(noteId);
                    }
                } catch (error) {
                    console.error('Error deleting note:', error);
                }
            }
            return;
        }
        
        if (noteCard) {
            ui.showNotePopup(noteCard);
        }
    });

    popupCloseBtn.addEventListener('click', ui.closeNotePopup);
    notePopupOverlay.addEventListener('click', (event) => {
        if (event.target === notePopupOverlay) {
            ui.closeNotePopup();
        }
    });
});