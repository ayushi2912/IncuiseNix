document.addEventListener('DOMContentLoaded', function() {
    const assistantForm = document.getElementById('assistant-form');
    const assistantInput = document.getElementById('assistant-input');
    const chatBox = document.getElementById('assistant-chat-box');
    const sendButton = document.getElementById('assistant-send-btn');
    const csrfToken = document.querySelector('input[name="csrfmiddlewaretoken"]')?.value || document.getElementById('player-data-container').dataset.csrfToken;

    // 1. Create a new showdown converter to handle Markdown
    const converter = new showdown.Converter();

    assistantForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const prompt = assistantInput.value.trim();
        if (!prompt) return;

        // Display user's message
        appendMessage(prompt, 'user');
        assistantInput.value = '';

        // Show loading indicator
        showLoadingIndicator();

        try {
            const response = await fetch('/assistant/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({ prompt: prompt })
            });

            const data = await response.json();

            // Remove loading indicator before appending the response
            removeLoadingIndicator();

            if (data.response) {
                appendMessage(data.response, 'assistant');
            } else {
                appendMessage('Sorry, an error occurred. Please try again.', 'assistant');
            }

        } catch (error) {
            console.error('Error:', error);
            removeLoadingIndicator();
            appendMessage('Sorry, an error occurred. Please try again.', 'assistant');
        }
    });

    function appendMessage(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender);

        // 2. Check if the message is from the assistant to convert it
        if (sender === 'assistant') {
            // Convert the Markdown string to HTML
            const htmlContent = converter.makeHtml(message);
            messageElement.innerHTML = htmlContent;
        } else {
            // Keep user messages as plain text
            messageElement.textContent = message;
        }

        chatBox.appendChild(messageElement);
        // Scroll to the latest message
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function showLoadingIndicator() {
        sendButton.disabled = true;
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('chat-message', 'loading');
        loadingElement.id = 'loading-indicator';
        loadingElement.innerHTML = `
            <div class="d-flex align-items-center">
                <strong class="me-2">Assistant is thinking...</strong>
                <div class="spinner-border spinner-border-sm" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
        chatBox.appendChild(loadingElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function removeLoadingIndicator() {
        sendButton.disabled = false;
        const loadingElement = document.getElementById('loading-indicator');
        if (loadingElement) {
            loadingElement.remove();
        }
    }
});