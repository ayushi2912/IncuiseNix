document.addEventListener('DOMContentLoaded', function() {
    const playerElement = document.getElementById('player');
    if (!playerElement) return;

    const player = new Plyr('#player', {
        controls: [
            'play-large',
            'play',
            'progress',
            'current-time',
            'mute',
            'volume',
            'captions',
            'settings',
            'fullscreen'
        ],
        settings: ['captions', 'speed', 'loop'],
        youtube: {
            rel: 0,
            cc_load_policy: 1,
            noCookie: true,
        }
    });

    window.videoPlayer = player;

    // --- Transcript Click-to-Seek Functionality ---
    const transcriptContainer = document.getElementById('transcript-container');
    if (transcriptContainer && window.videoPlayer) {
        transcriptContainer.addEventListener('click', function(event) {
            const line = event.target.closest('.transcript-line');
            if (line) {
                const startTime = parseFloat(line.dataset.start);
                if (!isNaN(startTime)) {
                    window.videoPlayer.currentTime = startTime;
                    window.videoPlayer.play();
                }
            }
        });
    }

    // --- START: Corrected Logic for Toggleable Sections ---
    const transcriptBtn = document.getElementById('toggle-transcript-btn');
    const assistantBtn = document.getElementById('toggle-assistant-btn');
    const transcriptSection = document.getElementById('transcript-section');
    const assistantSection = document.getElementById('assistant-chat');
    
    const sections = [transcriptSection, assistantSection].filter(el => el != null);

    function toggleSection(targetSection) {
        const isAlreadyOpen = targetSection.classList.contains('show');
        
        // First, hide all sections to ensure only one is open at a time
        sections.forEach(section => section.classList.remove('show'));

        // If the target section wasn't already open, show it.
        // If it was open, it will now be closed because of the step above.
        if (!isAlreadyOpen) {
            targetSection.classList.add('show');
        }
    }

    if (transcriptBtn) {
        transcriptBtn.addEventListener('click', () => toggleSection(transcriptSection));
    }
    if (assistantBtn) {
        assistantBtn.addEventListener('click', () => toggleSection(assistantSection));
    }

    // Close button for the transcript section
    const closeTranscriptBtn = transcriptSection ? transcriptSection.querySelector('.btn-close') : null;
    if (closeTranscriptBtn) {
        closeTranscriptBtn.addEventListener('click', () => {
            transcriptSection.classList.remove('show');
        });
    }
    
    // Close button for the AI assistant section
    const closeAssistantBtn = assistantSection ? assistantSection.querySelector('#close-assistant') : null;
    if (closeAssistantBtn) {
        closeAssistantBtn.addEventListener('click', () => {
            assistantSection.classList.remove('show');
        });
    }
    // --- END: Corrected Logic for Toggleable Sections ---
});