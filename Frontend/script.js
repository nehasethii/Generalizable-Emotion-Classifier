// Sample texts for quick testing
const sampleTexts = [
    "I'm so excited about this amazing opportunity! Can't wait to get started!",
    "This product is terrible. I'm extremely disappointed and frustrated.",
    "I'm not sure how I feel about this. It's a bit confusing and overwhelming."
];

// All possible emotions in GoEmotions dataset
const ALL_EMOTIONS = [
    'joy', 'admiration', 'approval', 'caring', 'excitement',
    'gratitude', 'love', 'optimism', 'pride', 'amusement',
    'anger', 'annoyance', 'disappointment', 'sadness', 'fear',
    'surprise', 'curiosity', 'confusion', 'disgust', 'embarrassment',
    'nervousness', 'realization', 'relief', 'remorse', 'grief',
    'desire', 'neutral'
];

// Emoji mapping for each emotion with animated variants
const EMOTION_EMOJIS = {
    'joy': '😊',
    'admiration': '🤩',
    'approval': '👍',
    'caring': '🤗',
    'excitement': '🎉',
    'gratitude': '🙏',
    'love': '❤️',
    'optimism': '🌟',
    'pride': '😌',
    'amusement': '😄',
    'anger': '😠',
    'annoyance': '😒',
    'disappointment': '😞',
    'sadness': '😢',
    'fear': '😨',
    'surprise': '😲',
    'curiosity': '🤔',
    'confusion': '😕',
    'disgust': '🤢',
    'embarrassment': '😳',
    'nervousness': '😰',
    'realization': '💡',
    'relief': '😌',
    'remorse': '😔',
    'grief': '😭',
    'desire': '🤤',
    'neutral': '😐'
};

// Animation colors for each emotion type
const EMOTION_COLORS = {
    'anger': ['#ff0000', '#ff4500', '#dc143c'],
    'joy': ['#ffd700', '#ffa500', '#ffff00'],
    'sadness': ['#4169e1', '#1e90ff', '#00bfff'],
    'fear': ['#800080', '#9370db', '#8a2be2'],
    'love': ['#ff1493', '#ff69b4', '#ff6347'],
    'surprise': ['#ffff00', '#ffd700', '#ffa500'],
    'disgust': ['#00ff00', '#32cd32', '#228b22'],
    'neutral': ['#808080', '#a9a9a9', '#c0c0c0']
};

/**
 * Use sample text by index
 * @param {number} index - Index of the sample text
 */
function useSampleText(index) {
    document.getElementById('textInput').value = sampleTexts[index];
}

/**
 * Main function to analyze emotion from input text
 */
function analyzeEmotion() {
    const text = document.getElementById('textInput').value.trim();
    
    if (!text) {
        alert('Please enter some text to analyze!');
        return;
    }

    const selectedModel = document.querySelector('input[name="model"]:checked').value;

    // Show loading state
    showLoading();

    // Simulate API call with mock data (replace with actual API call)
    setTimeout(() => {
        const mockResults = generateMockResults(text);
        displayResults(mockResults);
    }, 1500);

    // TO INTEGRATE WITH YOUR BACKEND, REPLACE THE ABOVE WITH:
    /*
    fetch('/api/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            model: selectedModel
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        displayResults(data.emotions);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while analyzing emotions. Please try again.');
        hideLoading();
    });
    */
}

/**
 * Generate mock results for demonstration
 * @param {string} text - Input text
 * @returns {Array} Array of emotion objects with confidence scores
 */
function generateMockResults(text) {
    // Create random confidence scores for emotions
    const results = ALL_EMOTIONS.map(emotion => ({
        emotion: emotion,
        confidence: Math.random() * 100
    }));

    // Sort by confidence in descending order
    results.sort((a, b) => b.confidence - a.confidence);
    
    // Return top 8 emotions
    return results.slice(0, 8);
}

/**
 * Display analysis results
 * @param {Array} results - Array of emotion objects
 */
function displayResults(results) {
    hideLoading();
    showResultsSections();
    
    displayEmotionList(results);
    displayChart(results);
    displayStatistics(results);
}

/**
 * Display emotion list with confidence bars
 * @param {Array} results - Array of emotion objects
 */
function displayEmotionList(results) {
    const emotionList = document.getElementById('emotionList');
    emotionList.innerHTML = '';
    
    results.forEach((result, index) => {
        const emoji = EMOTION_EMOJIS[result.emotion] || '😐';
        const item = document.createElement('div');
        item.className = 'emotion-item';
        item.style.animationDelay = `${index * 0.1}s`;
        item.innerHTML = `
            <div class="emotion-emoji" data-emotion="${result.emotion}">${emoji}</div>
            <div class="emotion-name">${capitalizeFirst(result.emotion)}</div>
            <div class="emotion-bar-container">
                <div class="emotion-bar" style="width: ${result.confidence}%">
                    <span class="emotion-percentage">${result.confidence.toFixed(1)}%</span>
                </div>
            </div>
        `;
        emotionList.appendChild(item);
    });
    
    // Add click and double-click event listeners to all emojis
    addEmojiClickEvents();
}

/**
 * Display bar chart of top emotions
 * @param {Array} results - Array of emotion objects
 */
function displayChart(results) {
    const chartContainer = document.getElementById('chartContainer');
    chartContainer.innerHTML = '';
    
    // Display top 5 emotions in chart
    results.slice(0, 5).forEach(result => {
        const emoji = EMOTION_EMOJIS[result.emotion] || '😐';
        const bar = document.createElement('div');
        bar.className = 'chart-bar';
        bar.style.height = `${result.confidence * 2.5}px`;
        bar.title = `${capitalizeFirst(result.emotion)}: ${result.confidence.toFixed(1)}%`;
        bar.innerHTML = `
            <div class="chart-value">${result.confidence.toFixed(0)}%</div>
            <div class="chart-emoji">${emoji}</div>
            <div class="chart-label">${capitalizeFirst(result.emotion)}</div>
        `;
        chartContainer.appendChild(bar);
    });
}

/**
 * Display statistics summary
 * @param {Array} results - Array of emotion objects
 */
function displayStatistics(results) {
    const statsGrid = document.getElementById('statsGrid');
    
    // Calculate statistics
    const topEmotion = results[0];
    const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length;
    const emotionCount = results.filter(r => r.confidence > 20).length;
    
    statsGrid.innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${capitalizeFirst(topEmotion.emotion)}</div>
            <div class="stat-label">Primary Emotion</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${topEmotion.confidence.toFixed(1)}%</div>
            <div class="stat-label">Confidence</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${emotionCount}</div>
            <div class="stat-label">Emotions Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${avgConfidence.toFixed(1)}%</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
    `;
}

/**
 * Show loading spinner
 */
function showLoading() {
    document.getElementById('loading').classList.add('active');
    document.querySelectorAll('.results-section').forEach(section => {
        section.classList.remove('active');
    });
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    document.getElementById('loading').classList.remove('active');
}

/**
 * Show all result sections
 */
function showResultsSections() {
    document.querySelectorAll('.results-section').forEach(section => {
        section.classList.add('active');
    });
}

/**
 * Clear all inputs and results
 */
function clearAll() {
    document.getElementById('textInput').value = '';
    document.querySelectorAll('.results-section').forEach(section => {
        section.classList.remove('active');
    });
    hideLoading();
}

/**
 * Capitalize first letter of a string
 * @param {string} str - Input string
 * @returns {string} String with first letter capitalized
 */
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Add click and double-click event listeners to emotion emojis
 */
function addEmojiClickEvents() {
    const emojis = document.querySelectorAll('.emotion-emoji');
    
    emojis.forEach(emojiElement => {
        let clickCount = 0;
        let clickTimer = null;
        
        emojiElement.addEventListener('click', function(e) {
            clickCount++;
            
            if (clickCount === 1) {
                // Single click - small animation
                clickTimer = setTimeout(() => {
                    const emotion = this.getAttribute('data-emotion');
                    triggerSmallAnimation(this, emotion, e);
                    clickCount = 0;
                }, 300);
            } else if (clickCount === 2) {
                // Double click - full screen animation
                clearTimeout(clickTimer);
                const emotion = this.getAttribute('data-emotion');
                triggerFullScreenAnimation(emotion, e);
                clickCount = 0;
            }
        });
    });
}

/**
 * Trigger small particle burst animation (Single Click)
 * @param {Element} emojiElement - The emoji DOM element
 * @param {string} emotion - The emotion type
 * @param {Event} event - Click event
 */
function triggerSmallAnimation(emojiElement, emotion, event) {
    const rect = emojiElement.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    // Get emoji and colors
    const emoji = EMOTION_EMOJIS[emotion] || '😐';
    const colors = getEmotionColors(emotion);
    
    // Create burst particles
    const particleCount = 12;
    for (let i = 0; i < particleCount; i++) {
        createBurstParticle(emoji, centerX, centerY, i, particleCount);
    }
    
    // Animate the emoji itself
    emojiElement.style.animation = 'none';
    setTimeout(() => {
        emojiElement.style.animation = 'emojiFloat 2s ease-in-out infinite';
    }, 10);
    
    // Add scale animation
    emojiElement.style.transform = 'scale(1.5) rotate(360deg)';
    setTimeout(() => {
        emojiElement.style.transform = '';
    }, 500);
}

/**
 * Trigger full-screen explosion animation (Double Click)
 * @param {string} emotion - The emotion type
 * @param {Event} event - Click event
 */
function triggerFullScreenAnimation(emotion, event) {
    // Create fullscreen overlay
    const overlay = document.createElement('div');
    overlay.className = 'fullscreen-emoji-overlay';
    document.body.appendChild(overlay);
    
    const emoji = EMOTION_EMOJIS[emotion] || '😐';
    const colors = getEmotionColors(emotion);
    
    // Create giant center emoji
    const giantEmoji = document.createElement('div');
    giantEmoji.className = 'giant-emoji';
    giantEmoji.textContent = emoji;
    overlay.appendChild(giantEmoji);
    
    // Create burst particles all around
    const burstCount = 30;
    for (let i = 0; i < burstCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'emoji-burst-particle';
        particle.textContent = emoji;
        
        const angle = (360 / burstCount) * i;
        const distance = 300 + Math.random() * 200;
        const tx = Math.cos(angle * Math.PI / 180) * distance;
        const ty = Math.sin(angle * Math.PI / 180) * distance;
        
        particle.style.left = '50%';
        particle.style.top = '50%';
        particle.style.setProperty('--tx', tx + 'px');
        particle.style.setProperty('--ty', ty + 'px');
        particle.style.animationDelay = (i * 0.02) + 's';
        
        overlay.appendChild(particle);
    }
    
    // Create confetti
    createConfetti(overlay, colors, 50);
    
    // Remove overlay after animation
    setTimeout(() => {
        overlay.remove();
    }, 2000);
}

/**
 * Create burst particle for small animation
 * @param {string} emoji - Emoji character
 * @param {number} centerX - X position
 * @param {number} centerY - Y position
 * @param {number} index - Particle index
 * @param {number} total - Total particles
 */
function createBurstParticle(emoji, centerX, centerY, index, total) {
    const particle = document.createElement('div');
    particle.className = 'emoji-burst-particle';
    particle.textContent = emoji;
    particle.style.position = 'fixed';
    particle.style.left = centerX + 'px';
    particle.style.top = centerY + 'px';
    particle.style.fontSize = '2em';
    particle.style.pointerEvents = 'none';
    particle.style.zIndex = '9999';
    
    const angle = (360 / total) * index;
    const distance = 100 + Math.random() * 50;
    const tx = Math.cos(angle * Math.PI / 180) * distance;
    const ty = Math.sin(angle * Math.PI / 180) * distance;
    
    particle.style.setProperty('--tx', tx + 'px');
    particle.style.setProperty('--ty', ty + 'px');
    
    document.body.appendChild(particle);
    
    setTimeout(() => {
        particle.remove();
    }, 2000);
}

/**
 * Create confetti pieces
 * @param {Element} container - Container element
 * @param {Array} colors - Array of colors
 * @param {number} count - Number of confetti pieces
 */
function createConfetti(container, colors, count) {
    for (let i = 0; i < count; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti-piece';
        confetti.style.left = Math.random() * 100 + '%';
        confetti.style.top = -10 + 'px';
        confetti.style.setProperty('--color', colors[Math.floor(Math.random() * colors.length)]);
        confetti.style.animationDelay = (Math.random() * 0.5) + 's';
        confetti.style.animationDuration = (2 + Math.random() * 1) + 's';
        
        container.appendChild(confetti);
    }
}

/**
 * Get colors for emotion type
 * @param {string} emotion - Emotion name
 * @returns {Array} Array of colors
 */
function getEmotionColors(emotion) {
    if (['anger', 'annoyance'].includes(emotion)) return EMOTION_COLORS.anger;
    if (['joy', 'amusement', 'excitement'].includes(emotion)) return EMOTION_COLORS.joy;
    if (['sadness', 'grief', 'disappointment'].includes(emotion)) return EMOTION_COLORS.sadness;
    if (['fear', 'nervousness'].includes(emotion)) return EMOTION_COLORS.fear;
    if (['love', 'caring', 'gratitude'].includes(emotion)) return EMOTION_COLORS.love;
    if (['surprise'].includes(emotion)) return EMOTION_COLORS.surprise;
    if (['disgust'].includes(emotion)) return EMOTION_COLORS.disgust;
    return EMOTION_COLORS.neutral;
}

// Add event listener for Enter key in textarea
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('textInput');
    if (textarea) {
        textarea.addEventListener('keydown', function(e) {
            // Ctrl/Cmd + Enter to analyze
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                analyzeEmotion();
            }
        });
    }
});