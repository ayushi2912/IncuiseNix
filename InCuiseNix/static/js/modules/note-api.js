const playerContainer = document.getElementById('player-data-container');
const csrfToken = playerContainer ? playerContainer.dataset.csrfToken : '';

async function fetchAPI(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Something went wrong');
    }
    return response.json();
}

export function addNote(videoId, formData) {
    return fetchAPI(`/note/add/${videoId}/`, {
        method: 'POST',
        headers: { 'X-CSRFToken': csrfToken },
        body: formData,
    });
}

export function editNote(noteId, newContent) {
    return fetchAPI(`/note/edit/${noteId}/`, {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrfToken,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ content: newContent }),
    });
}

export function deleteNote(noteId) {
    return fetchAPI(`/note/delete/${noteId}/`, {
        method: 'POST',
        headers: { 'X-CSRFToken': csrfToken },
    });
}