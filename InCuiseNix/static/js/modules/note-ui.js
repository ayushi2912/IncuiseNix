import { formatTimestamp } from './utils.js';

const notesList = document.getElementById('notes-list-container');
const noNotesMessage = document.getElementById('no-notes-message');
const notePopupOverlay = document.getElementById('note-popup-overlay');
const popupNoteContent = document.getElementById('popup-note-content');
const popupNoteMeta = document.getElementById('popup-note-meta');

function createNoteCard(note) {
    const safeContent = note.content.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    return `
        <div class="card mb-3 note-card" 
             data-note-id="${note.id}"
             data-full-content="${safeContent}"
             data-meta-info="Timestamp: ${note.timestamp}s &mdash; Added on: ${note.created_at}">
            <div class="card-body">
                <div class="note-status-indicator"></div>
                <div class="note-text-wrapper">
                    <p class="card-text text-truncate">${safeContent}</p>
                    <p class="note-timestamp" data-seconds="${note.timestamp}">
                        Timestamp: ${formatTimestamp(note.timestamp)}
                    </p>
                </div>
                <div class="note-actions">
                    <button class="note-btn note-btn-edit btn-edit-note"
                            data-note-id="${note.id}"
                            data-note-content="${safeContent}">
                        Edit <i class="fas fa-pen-to-square ms-1"></i>
                    </button>
                    <button class="note-btn note-btn-delete btn-delete-note"
                            data-note-id="${note.id}">
                        <i class="fas fa-trash-can"></i>
                    </button>
                </div>
            </div>
        </div>
    `;
}

export function addNoteToUI(note) {
    if (noNotesMessage) noNotesMessage.style.display = 'none';
    const newNoteCardHTML = createNoteCard(note);
    notesList.insertAdjacentHTML('afterbegin', newNoteCardHTML);
}

export function updateNoteInUI(noteId, newContent) {
    const noteCard = document.querySelector(`.note-card[data-note-id='${noteId}']`);
    if (noteCard) {
        noteCard.querySelector('.card-text').textContent = newContent;
        noteCard.dataset.fullContent = newContent;
    }
    const editButton = document.querySelector(`.btn-edit-note[data-note-id='${noteId}']`);
    if (editButton) editButton.dataset.noteContent = newContent;
}

export function removeNoteFromUI(noteId) {
    const noteCard = document.querySelector(`.note-card[data-note-id='${noteId}']`);
    if (noteCard) noteCard.remove();
}

export function showNotePopup(noteCard) {
    const noteContent = noteCard.dataset.fullContent;
    const noteMeta = noteCard.dataset.metaInfo;

    popupNoteContent.textContent = noteContent;
    popupNoteMeta.innerHTML = noteMeta;
    notePopupOverlay.style.display = 'flex';
}

export function closeNotePopup() {
    notePopupOverlay.style.display = 'none';
}