/**
 * Challenge Page JavaScript
 * Handles security verification and adaptation challenges
 */

class ChallengeManager {
    constructor() {
        this.socket = null;
        this.challengeType = this.getChallengeType();
        this.challengeData = null;
        this.isActive = false;
        
        // Behavioral data collection for challenge response
        this.behavioralData = {
            keystrokes: [],
            mouse_events: [],
            challenge_start: Date.now()
        };
        
        // Challenge configuration
        this.verificationPhrases = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How razorback-jumping frogs can level six piqued gymnasts.",
            "Jinxed wizards pluck ivy from the big quilt.",
            "The five boxing wizards jump quickly."
        ];
        
        this.adaptationParagraphs = [
            "In the digital age, cybersecurity has become paramount for protecting sensitive information and maintaining user privacy. Behavioral biometrics represents a cutting-edge approach to continuous authentication, analyzing unique patterns in how users interact with their devices. This technology provides an additional layer of security beyond traditional password-based systems.",
            
            "Machine learning algorithms can detect subtle changes in typing patterns, mouse movements, and other behavioral characteristics that are difficult for attackers to replicate. By continuously monitoring user behavior, organizations can identify potential security breaches before they cause significant damage. The integration of artificial intelligence in security systems enables real-time threat detection and adaptive responses to emerging cyber threats.",
            
            "Behavioral authentication systems learn and adapt to users' natural evolution of typing patterns over time. Factors such as fatigue, stress, injury, or simply growing familiarity with a device can cause gradual changes in behavioral metrics. Advanced systems can distinguish between natural behavioral drift and suspicious anomalies that might indicate unauthorized access attempts."
        ];
        
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.startBehavioralCollection();
        this.initializeChallenge();
        this.setupProgressTracking();
    }
    
    getChallengeType() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('type') || 'verification';
    }
    
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/socket.io/`;
        
        this.socket = io(wsUrl, {
            auth: {
                token: localStorage.getItem('auth_token')
            }
        });
        
        this.socket.on('connect', () => {
            console.log('Connected to server for challenge');
        });
        
        this.socket.on('challenge_result', (data) => {
            this.handleChallengeResult(data);
        });
        
        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            this.showMessage('Connection error during challenge', 'error');
        });
    }
    
    setupEventListeners() {
        // Start challenge button
        const startBtn = document.getElementById('startChallengeBtn');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startChallenge());
        }
        
        // Submit challenge button
        const submitBtn = document.getElementById('submitChallengeBtn');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => this.submitChallenge());
        }
        
        // Cancel button
        const cancelBtn = document.getElementById('cancelBtn');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.cancelChallenge());
        }
        
        // Retry button
        const retryBtn = document.getElementById('retryBtn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => this.retryChallenge());
        }
        
        // Window focus/blur handling
        window.addEventListener('focus', () => {
            if (this.isActive) {
                this.resumeChallenge();
            }
        });
        
        window.addEventListener('blur', () => {
            if (this.isActive) {
                this.pauseChallenge();
            }
        });
        
        // Prevent page navigation during active challenge
        window.addEventListener('beforeunload', (e) => {
            if (this.isActive) {
                e.preventDefault();
                e.returnValue = 'Challenge in progress. Are you sure you want to leave?';
                return e.returnValue;
            }
        });
    }
    
    startBehavioralCollection() {
        // Keystroke monitoring
        document.addEventListener('keydown', (e) => {
            if (this.isActive) {
                this.recordKeystroke(e, 'keydown');
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (this.isActive) {
                this.recordKeystroke(e, 'keyup');
            }
        });
        
        // Mouse monitoring
        document.addEventListener('mousemove', (e) => {
            if (this.isActive) {
                this.recordMouseEvent(e, 'mousemove');
            }
        });
        
        document.addEventListener('mousedown', (e) => {
            if (this.isActive) {
                this.recordMouseEvent(e, 'mousedown');
            }
        });
        
        document.addEventListener('mouseup', (e) => {
            if (this.isActive) {
                this.recordMouseEvent(e, 'mouseup');
            }
        });
        
        document.addEventListener('click', (e) => {
            if (this.isActive) {
                this.recordMouseEvent(e, 'click');
            }
        });
        
        document.addEventListener('wheel', (e) => {
            if (this.isActive) {
                this.recordMouseEvent(e, 'wheel');
            }
        });
    }
    
    recordKeystroke(event, type) {
        const keystrokeData = {
            type: type,
            key: event.key,
            code: event.code,
            timestamp: Date.now(),
            target: this.getTargetInfo(event.target),
            shiftKey: event.shiftKey,
            ctrlKey: event.ctrlKey,
            altKey: event.altKey,
            metaKey: event.metaKey,
            location: event.location,
            repeat: event.repeat,
            challenge_phase: this.getCurrentPhase()
        };
        
        this.behavioralData.keystrokes.push(keystrokeData);
        this.updateProgress();
        
        // Limit buffer size
        if (this.behavioralData.keystrokes.length > 1000) {
            this.behavioralData.keystrokes = this.behavioralData.keystrokes.slice(-800);
        }
    }
    
    recordMouseEvent(event, type) {
        const mouseData = {
            type: type,
            x: event.clientX,
            y: event.clientY,
            timestamp: Date.now(),
            target: this.getTargetInfo(event.target),
            button: event.button,
            buttons: event.buttons,
            challenge_phase: this.getCurrentPhase()
        };
        
        if (type === 'wheel') {
            mouseData.deltaX = event.deltaX;
            mouseData.deltaY = event.deltaY;
            mouseData.deltaZ = event.deltaZ;
            mouseData.deltaMode = event.deltaMode;
        }
        
        if (type === 'click') {
            mouseData.detail = event.detail;
        }
        
        this.behavioralData.mouse_events.push(mouseData);
        this.updateProgress();
        
        // Limit buffer size
        if (this.behavioralData.mouse_events.length > 1500) {
            this.behavioralData.mouse_events = this.behavioralData.mouse_events.slice(-1000);
        }
    }
    
    getTargetInfo(target) {
        return {
            tagName: target.tagName.toLowerCase(),
            id: target.id || null,
            className: target.className || null,
            type: target.type || null
        };
    }
    
    getCurrentPhase() {
        if (!this.isActive) return 'inactive';
        return this.challengeType === 'verification' ? 'verification' : 'adaptation';
    }
    
    initializeChallenge() {
        this.updateChallengeUI();
        this.showInstructions();
    }
    
    updateChallengeUI() {
        const titleElement = document.getElementById('challengeTitle');
        const instructionsElement = document.getElementById('challengeInstructions');
        
        if (this.challengeType === 'verification') {
            if (titleElement) titleElement.textContent = 'Security Verification';
            if (instructionsElement) {
                instructionsElement.innerHTML = `
                    <p>Unusual behavior has been detected on your account. To verify your identity, please type the phrase shown below exactly as displayed.</p>
                    <p><strong>Type naturally</strong> - this helps us confirm it's really you.</p>
                `;
            }
        } else {
            if (titleElement) titleElement.textContent = 'Behavioral Recalibration';
            if (instructionsElement) {
                instructionsElement.innerHTML = `
                    <p>Your typing patterns have naturally evolved over time. To update your behavioral profile, please type the paragraph shown below.</p>
                    <p><strong>Type at your normal pace</strong> - this helps us learn your current patterns.</p>
                `;
            }
        }
    }
    
    showInstructions() {
        const instructionsModal = document.createElement('div');
        instructionsModal.className = 'modal-overlay instructions-modal active';
        instructionsModal.innerHTML = `
            <div class="modal">
                <div class="modal-header">
                    <h3>Challenge Instructions</h3>
                </div>
                <div class="modal-body">
                    <div class="instruction-content">
                        ${this.challengeType === 'verification' ? this.getVerificationInstructions() : this.getAdaptationInstructions()}
                    </div>
                    <div class="instruction-tips">
                        <h4>Important Tips:</h4>
                        <ul>
                            <li>Type naturally at your normal speed</li>
                            <li>Don't try to change your typing style</li>
                            <li>Focus on accuracy over speed</li>
                            <li>Take your time - there's no rush</li>
                        </ul>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-primary" onclick="this.startFromModal()">
                        I Understand - Start Challenge
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(instructionsModal);
        
        // Add method to modal
        instructionsModal.startFromModal = () => {
            document.body.removeChild(instructionsModal);
            this.startChallenge();
        };
    }
    
    getVerificationInstructions() {
        return `
            <p>This verification challenge helps confirm your identity by analyzing your unique typing patterns.</p>
            <p>You'll be asked to type a short phrase. Our system will compare your typing style to your established behavioral profile.</p>
            <p>This process typically takes 1-2 minutes and helps protect your account from unauthorized access.</p>
        `;
    }
    
    getAdaptationInstructions() {
        return `
            <p>Your behavioral patterns naturally change over time due to factors like:</p>
            <ul>
                <li>Increased familiarity with your device</li>
                <li>Changes in hand position or posture</li>
                <li>Different environmental conditions</li>
                <li>Natural variation in motor skills</li>
            </ul>
            <p>This recalibration process helps our system adapt to your current typing style while maintaining security.</p>
        `;
    }
    
    startChallenge() {
        this.isActive = true;
        this.behavioralData.challenge_start = Date.now();
        
        // Generate challenge content
        this.generateChallengeContent();
        
        // Update UI
        this.updateChallengeStatus('active');
        this.showChallengeContent();
        
        // Start progress tracking
        this.startProgressTimer();
        
        this.showMessage('Challenge started. Type naturally.', 'info');
    }
    
    generateChallengeContent() {
        if (this.challengeType === 'verification') {
            // Select random verification phrase
            const phrase = this.verificationPhrases[Math.floor(Math.random() * this.verificationPhrases.length)];
            this.challengeData = {
                type: 'verification',
                target_text: phrase,
                min_accuracy: 95,
                min_length: phrase.length
            };
        } else {
            // Select random adaptation paragraph
            const paragraph = this.adaptationParagraphs[Math.floor(Math.random() * this.adaptationParagraphs.length)];
            this.challengeData = {
                type: 'adaptation',
                target_text: paragraph,
                min_accuracy: 90,
                min_length: Math.floor(paragraph.length * 0.8) // Allow partial completion
            };
        }
    }
    
    showChallengeContent() {
        const contentContainer = document.getElementById('challengeContent');
        if (!contentContainer) return;
        
        contentContainer.innerHTML = `
            <div class="challenge-target">
                <h4>Please type the following text:</h4>
                <div class="target-text" id="targetText">${this.challengeData.target_text}</div>
            </div>
            
            <div class="challenge-input-area">
                <textarea 
                    id="challengeInput" 
                    class="challenge-textarea"
                    placeholder="Start typing here..."
                    rows="${this.challengeType === 'verification' ? 3 : 8}"
                ></textarea>
                
                <div class="challenge-stats">
                    <div class="stat-item">
                        <span class="stat-label">Progress:</span>
                        <span class="stat-value" id="progressPercent">0%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Accuracy:</span>
                        <span class="stat-value" id="accuracyPercent">100%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">WPM:</span>
                        <span class="stat-value" id="wpmValue">0</span>
                    </div>
                </div>
                
                <div class="progress-bar">
                    <div class="progress-fill" id="challengeProgress" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="challenge-actions">
                <button class="btn btn-primary" id="submitChallengeBtn" disabled>
                    Submit Response
                </button>
                <button class="btn btn-outline" id="cancelBtn">
                    Cancel
                </button>
            </div>
        `;
        
        // Setup input monitoring
        this.setupInputMonitoring();
        
        // Focus the textarea
        const textarea = document.getElementById('challengeInput');
        if (textarea) {
            textarea.focus();
        }
    }
    
    setupInputMonitoring() {
        const textarea = document.getElementById('challengeInput');
        if (!textarea) return;
        
        let lastUpdateTime = Date.now();
        let wordCount = 0;
        
        textarea.addEventListener('input', () => {
            const inputText = textarea.value;
            const targetText = this.challengeData.target_text;
            
            // Calculate progress
            const progress = Math.min(100, (inputText.length / targetText.length) * 100);
            this.updateProgressBar(progress);
            
            // Calculate accuracy
            const accuracy = this.calculateAccuracy(inputText, targetText);
            this.updateAccuracy(accuracy);
            
            // Calculate WPM
            const currentTime = Date.now();
            const timeDiff = (currentTime - this.behavioralData.challenge_start) / 1000 / 60; // minutes
            const currentWordCount = inputText.trim().split(/\s+/).filter(word => word.length > 0).length;
            const wpm = timeDiff > 0 ? Math.round(currentWordCount / timeDiff) : 0;
            this.updateWPM(wpm);
            
            // Update visual feedback
            this.updateInputFeedback(textarea, inputText, targetText);
            
            // Check if challenge can be submitted
            this.checkSubmissionEligibility(inputText, accuracy, progress);
            
            lastUpdateTime = currentTime;
            wordCount = currentWordCount;
        });
        
        // Handle paste events
        textarea.addEventListener('paste', (e) => {
            e.preventDefault();
            this.showMessage('Pasting is not allowed during the challenge', 'warning');
        });
        
        // Handle copy events
        textarea.addEventListener('copy', (e) => {
            e.preventDefault();
            this.showMessage('Copying is not allowed during the challenge', 'warning');
        });
    }
    
    calculateAccuracy(input, target) {
        if (input.length === 0) return 100;
        
        let matches = 0;
        const minLength = Math.min(input.length, target.length);
        
        for (let i = 0; i < minLength; i++) {
            if (input[i] === target[i]) {
                matches++;
            }
        }
        
        return Math.round((matches / Math.max(input.length, target.length)) * 100);
    }
    
    updateProgressBar(progress) {
        const progressBar = document.getElementById('challengeProgress');
        const progressText = document.getElementById('progressPercent');
        
        if (progressBar) {
            progressBar.style.width = progress + '%';
        }
        
        if (progressText) {
            progressText.textContent = Math.round(progress) + '%';
        }
    }
    
    updateAccuracy(accuracy) {
        const accuracyElement = document.getElementById('accuracyPercent');
        if (accuracyElement) {
            accuracyElement.textContent = accuracy + '%';
            
            // Color coding
            if (accuracy >= 95) {
                accuracyElement.className = 'stat-value accuracy-high';
            } else if (accuracy >= 85) {
                accuracyElement.className = 'stat-value accuracy-medium';
            } else {
                accuracyElement.className = 'stat-value accuracy-low';
            }
        }
    }
    
    updateWPM(wpm) {
        const wpmElement = document.getElementById('wpmValue');
        if (wpmElement) {
            wpmElement.textContent = wpm;
        }
    }
    
    updateInputFeedback(textarea, input, target) {
        // Highlight errors by changing background color
        const accuracy = this.calculateAccuracy(input, target);
        
        if (accuracy >= 95) {
            textarea.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
            textarea.style.borderColor = 'var(--success-color)';
        } else if (accuracy >= 85) {
            textarea.style.backgroundColor = 'rgba(245, 158, 11, 0.1)';
            textarea.style.borderColor = 'var(--warning-color)';
        } else {
            textarea.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
            textarea.style.borderColor = 'var(--error-color)';
        }
    }
    
    checkSubmissionEligibility(input, accuracy, progress) {
        const submitBtn = document.getElementById('submitChallengeBtn');
        if (!submitBtn) return;
        
        const isEligible = 
            input.length >= this.challengeData.min_length &&
            accuracy >= this.challengeData.min_accuracy &&
            progress >= (this.challengeType === 'verification' ? 95 : 80);
        
        submitBtn.disabled = !isEligible;
        submitBtn.classList.toggle('btn-disabled', !isEligible);
        
        if (isEligible && !submitBtn.dataset.eligibleShown) {
            this.showMessage('Challenge requirements met! You can now submit.', 'success');
            submitBtn.dataset.eligibleShown = 'true';
        }
    }
    
    updateProgress() {
        // Update general progress indicators
        const keystrokeCount = this.behavioralData.keystrokes.length;
        const mouseEventCount = this.behavioralData.mouse_events.length;
        const totalEvents = keystrokeCount + mouseEventCount;
        
        // Update progress indicators if they exist
        const eventCountElement = document.getElementById('eventCount');
        if (eventCountElement) {
            eventCountElement.textContent = totalEvents.toLocaleString();
        }
    }
    
    startProgressTimer() {
        const timerElement = document.getElementById('challengeTimer');
        if (!timerElement) return;
        
        const startTime = Date.now();
        
        const updateTimer = () => {
            if (!this.isActive) return;
            
            const elapsed = Date.now() - startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            
            timerElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            setTimeout(updateTimer, 1000);
        };
        
        updateTimer();
    }
    
    setupProgressTracking() {
        // Add timer element if it doesn't exist
        const statusArea = document.getElementById('challengeStatus');
        if (statusArea && !document.getElementById('challengeTimer')) {
            const timerElement = document.createElement('div');
            timerElement.id = 'challengeTimer';
            timerElement.className = 'challenge-timer';
            timerElement.textContent = '0:00';
            statusArea.appendChild(timerElement);
        }
    }
    
    async submitChallenge() {
        if (!this.isActive) return;
        
        const textarea = document.getElementById('challengeInput');
        if (!textarea) return;
        
        const inputText = textarea.value;
        const accuracy = this.calculateAccuracy(inputText, this.challengeData.target_text);
        
        // Final validation
        if (inputText.length < this.challengeData.min_length) {
            this.showMessage('Please complete more of the text before submitting.', 'warning');
            return;
        }
        
        if (accuracy < this.challengeData.min_accuracy) {
            this.showMessage(`Accuracy too low (${accuracy}%). Please check for errors.`, 'warning');
            return;
        }
        
        // Disable submission
        this.isActive = false;
        this.updateChallengeStatus('submitting');
        
        try {
            // Prepare submission data
            const submissionData = {
                type: this.challengeType,
                challenge_text: this.challengeData.target_text,
                user_input: inputText,
                behavioral_data: this.behavioralData,
                challenge_metrics: {
                    accuracy: accuracy,
                    completion_time: Date.now() - this.behavioralData.challenge_start,
                    input_length: inputText.length,
                    target_length: this.challengeData.target_text.length,
                    keystroke_count: this.behavioralData.keystrokes.length,
                    mouse_event_count: this.behavioralData.mouse_events.length
                }
            };
            
            // Send via WebSocket
            this.socket.emit('challenge_response', submissionData);
            
            this.showMessage('Challenge submitted. Processing response...', 'info');
            
        } catch (error) {
            console.error('Error submitting challenge:', error);
            this.showMessage('Error submitting challenge. Please try again.', 'error');
            this.isActive = true;
            this.updateChallengeStatus('active');
        }
    }
    
    handleChallengeResult(data) {
        this.updateChallengeStatus('completed');
        
        if (data.success) {
            this.showSuccessResult(data);
        } else {
            this.showFailureResult(data);
        }
    }
    
    showSuccessResult(data) {
        const resultContainer = document.getElementById('challengeResult');
        if (!resultContainer) return;
        
        resultContainer.innerHTML = `
            <div class="result-success">
                <div class="result-icon">✓</div>
                <h3>Challenge Completed Successfully</h3>
                <p>${data.message}</p>
                
                <div class="result-stats">
                    <div class="stat-item">
                        <span class="stat-label">Verification Confidence:</span>
                        <span class="stat-value">${Math.round((data.confidence || 0) * 100)}%</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Completion Time:</span>
                        <span class="stat-value">${this.formatDuration(Date.now() - this.behavioralData.challenge_start)}</span>
                    </div>
                </div>
                
                <div class="result-actions">
                    <button class="btn btn-primary" onclick="window.location.href='/dashboard'">
                        Return to Dashboard
                    </button>
                </div>
            </div>
        `;
        
        // Auto-redirect after 5 seconds
        setTimeout(() => {
            window.location.href = '/dashboard';
        }, 5000);
    }
    
    showFailureResult(data) {
        const resultContainer = document.getElementById('challengeResult');
        if (!resultContainer) return;
        
        resultContainer.innerHTML = `
            <div class="result-failure">
                <div class="result-icon">✗</div>
                <h3>Challenge Failed</h3>
                <p>${data.message || 'Verification unsuccessful. Please try again.'}</p>
                
                <div class="failure-reason">
                    <h4>Possible reasons:</h4>
                    <ul>
                        <li>Typing pattern doesn't match your profile</li>
                        <li>Too many errors in the text</li>
                        <li>Unusual typing speed or rhythm</li>
                        <li>Environmental factors affecting your typing</li>
                    </ul>
                </div>
                
                <div class="result-actions">
                    <button class="btn btn-primary" id="retryBtn" onclick="this.retryChallenge()">
                        Try Again
                    </button>
                    <button class="btn btn-outline" onclick="window.location.href='/dashboard'">
                        Return to Dashboard
                    </button>
                </div>
            </div>
        `;
        
        // If too many failures, force logout
        if (data.force_logout) {
            setTimeout(() => {
                this.forceLogout();
            }, 3000);
        }
    }
    
    retryChallenge() {
        // Reset challenge state
        this.isActive = false;
        this.challengeData = null;
        this.behavioralData = {
            keystrokes: [],
            mouse_events: [],
            challenge_start: Date.now()
        };
        
        // Clear UI
        const resultContainer = document.getElementById('challengeResult');
        if (resultContainer) {
            resultContainer.innerHTML = '';
        }
        
        const contentContainer = document.getElementById('challengeContent');
        if (contentContainer) {
            contentContainer.innerHTML = '';
        }
        
        // Restart challenge
        this.updateChallengeStatus('ready');
        this.initializeChallenge();
    }
    
    cancelChallenge() {
        if (confirm('Are you sure you want to cancel the challenge? This may affect your account security.')) {
            this.isActive = false;
            
            // Return to dashboard or logout
            if (this.challengeType === 'verification') {
                // High-security scenario - logout
                this.showMessage('Challenge cancelled. Logging out for security.', 'warning');
                setTimeout(() => {
                    this.forceLogout();
                }, 2000);
            } else {
                // Adaptation scenario - return to dashboard
                window.location.href = '/dashboard';
            }
        }
    }
    
    pauseChallenge() {
        if (this.isActive) {
            this.showMessage('Challenge paused. Return to this window to continue.', 'warning');
        }
    }
    
    resumeChallenge() {
        if (this.isActive) {
            this.showMessage('Challenge resumed.', 'info');
        }
    }
    
    updateChallengeStatus(status) {
        const statusElement = document.getElementById('challengeStatus');
        if (!statusElement) return;
        
        const statusMap = {
            'ready': { class: 'status-info', text: 'Ready to Start' },
            'active': { class: 'status-success', text: 'Challenge Active' },
            'submitting': { class: 'status-warning', text: 'Processing...' },
            'completed': { class: 'status-success', text: 'Completed' },
            'failed': { class: 'status-error', text: 'Failed' }
        };
        
        const statusInfo = statusMap[status] || statusMap['ready'];
        statusElement.className = `status-indicator ${statusInfo.class}`;
        statusElement.innerHTML = `
            <span class="status-dot"></span>
            ${statusInfo.text}
        `;
    }
    
    formatDuration(milliseconds) {
        const seconds = Math.floor(milliseconds / 1000);
        const minutes = Math.floor(seconds / 60);
        
        if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    }
    
    showMessage(message, type = 'info') {
        const messageContainer = document.getElementById('messageContainer');
        if (!messageContainer) return;
        
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.innerHTML = `
            <div class="alert-content">
                <div class="alert-icon">${this.getAlertIcon(type)}</div>
                <div class="alert-message">${message}</div>
            </div>
        `;
        
        messageContainer.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.style.opacity = '0';
                setTimeout(() => {
                    if (alertDiv.parentNode) {
                        alertDiv.parentNode.removeChild(alertDiv);
                    }
                }, 300);
            }
        }, 5000);
    }
    
    getAlertIcon(type) {
        const icons = {
            success: '✓',
            error: '✗',
            warning: '⚠',
            info: 'ℹ'
        };
        return icons[type] || icons.info;
    }
    
    forceLogout() {
        // Clear authentication
        localStorage.removeItem('auth_token');
        
        // Disconnect socket
        if (this.socket) {
            this.socket.disconnect();
        }
        
        // Redirect to login
        window.location.href = '/login';
    }
}

// Challenge-specific styles
const challengeStyles = `
<style>
.challenge-textarea {
    width: 100%;
    padding: 1rem;
    border: 2px solid var(--gray-200);
    border-radius: var(--radius-md);
    font-size: 1rem;
    font-family: 'Monaco', 'Consolas', monospace;
    background: var(--bg-primary);
    color: var(--text-primary);
    transition: all var(--transition-fast);
    resize: vertical;
    line-height: 1.6;
}

.challenge-textarea:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.target-text {
    background: var(--bg-secondary);
    padding: 1.5rem;
    border-radius: var(--radius-md);
    margin-bottom: 1.5rem;
    border-left: 4px solid var(--primary-color);
    font-family: 'Monaco', 'Consolas', monospace;
    line-height: 1.8;
    font-size: 1rem;
    color: var(--text-primary);
}

.challenge-stats {
    display: flex;
    justify-content: space-between;
    margin: 1rem 0;
    padding: 1rem;
    background: var(--bg-secondary);
    border-radius: var(--radius-md);
}

.stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
}

.stat-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.stat-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--primary-color);
}

.stat-value.accuracy-high {
    color: var(--success-color);
}

.stat-value.accuracy-medium {
    color: var(--warning-color);
}

.stat-value.accuracy-low {
    color: var(--error-color);
}

.challenge-actions {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 2rem;
}

.challenge-timer {
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--primary-color);
    text-align: center;
    padding: 0.5rem;
}

.result-success, .result-failure {
    text-align: center;
    padding: 2rem;
    background: var(--bg-secondary);
    border-radius: var(--radius-lg);
    margin-top: 2rem;
}

.result-success {
    border: 2px solid var(--success-color);
}

.result-failure {
    border: 2px solid var(--error-color);
}

.result-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

.result-success .result-icon {
    color: var(--success-color);
}

.result-failure .result-icon {
    color: var(--error-color);
}

.result-stats {
    display: flex;
    justify-content: space-around;
    margin: 2rem 0;
    padding: 1rem;
    background: var(--bg-primary);
    border-radius: var(--radius-md);
}

.result-actions {
    margin-top: 2rem;
    display: flex;
    gap: 1rem;
    justify-content: center;
}

.failure-reason {
    text-align: left;
    background: var(--bg-primary);
    padding: 1rem;
    border-radius: var(--radius-md);
    margin: 1.5rem 0;
}

.failure-reason h4 {
    margin-bottom: 0.5rem;
    color: var(--text-primary);
}

.failure-reason ul {
    margin: 0;
    padding-left: 1.5rem;
    color: var(--text-secondary);
}

.instructions-modal .modal {
    max-width: 600px;
}

.instruction-content {
    margin-bottom: 1.5rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

.instruction-tips {
    background: var(--bg-tertiary);
    padding: 1rem;
    border-radius: var(--radius-md);
    border-left: 4px solid var(--info-color);
}

.instruction-tips h4 {
    margin-bottom: 0.5rem;
    color: var(--info-color);
}

.instruction-tips ul {
    margin: 0;
    padding-left: 1.5rem;
    color: var(--text-secondary);
}

.challenge-target h4 {
    margin-bottom: 1rem;
    color: var(--text-primary);
}

@media (max-width: 768px) {
    .challenge-stats {
        flex-direction: column;
        gap: 1rem;
    }
    
    .stat-item {
        flex-direction: row;
        justify-content: space-between;
    }
    
    .result-stats {
        flex-direction: column;
        gap: 1rem;
    }
    
    .challenge-actions, .result-actions {
        flex-direction: column;
    }
}
</style>
`;

// Add styles to document head
document.head.insertAdjacentHTML('beforeend', challengeStyles);

// Initialize challenge manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChallengeManager();
});

// Export for testing
window.ChallengeManager = ChallengeManager;