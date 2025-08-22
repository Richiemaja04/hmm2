/**
 * Calibration Page JavaScript
 * Handles behavioral data collection during calibration phase
 */

class CalibrationManager {
    constructor() {
        this.calibrationContainer = document.getElementById('calibrationContainer');
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.currentTaskElement = document.getElementById('currentTask');
        this.instructionsElement = document.getElementById('instructions');
        this.statusElement = document.getElementById('status');
        
        // Calibration state
        this.currentPhase = 0;
        this.totalPhases = 4;
        this.samplesCollected = 0;
        this.targetSamples = 100;
        
        // Behavioral data collection
        this.behavioralData = {
            keystrokes: [],
            mouse_events: [],
            session_start: Date.now()
        };
        
        // Calibration phases
        this.phases = [
            {
                name: 'Free Typing',
                instruction: 'Type naturally about your day, interests, or anything that comes to mind. Aim for at least 200 characters.',
                minChars: 200,
                type: 'free_text'
            },
            {
                name: 'Structured Text',
                instruction: 'Please type the following text exactly as shown:',
                targetText: 'The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once. Behavioral biometrics uses typing patterns for authentication.',
                type: 'copy_text'
            },
            {
                name: 'Password Practice',
                instruction: 'Type a password that you might use (don\'t use your real password). Include uppercase, lowercase, numbers, and symbols.',
                minChars: 12,
                type: 'password_practice'
            },
            {
                name: 'Mouse Interaction',
                instruction: 'Perform various mouse actions: move around, click on different elements, scroll, and interact naturally with the interface.',
                duration: 30000, // 30 seconds
                type: 'mouse_interaction'
            }
        ];
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.startBehavioralCollection();
        this.startCalibration();
        this.setupMouseTargets();
    }
    
    setupEventListeners() {
        // Skip button
        const skipButton = document.getElementById('skipButton');
        if (skipButton) {
            skipButton.addEventListener('click', () => this.skipPhase());
        }
        
        // Next button
        const nextButton = document.getElementById('nextButton');
        if (nextButton) {
            nextButton.addEventListener('click', () => this.nextPhase());
        }
        
        // Complete button
        const completeButton = document.getElementById('completeButton');
        if (completeButton) {
            completeButton.addEventListener('click', () => this.completeCalibration());
        }
    }
    
    startBehavioralCollection() {
        // Keystroke monitoring
        document.addEventListener('keydown', (e) => {
            this.recordKeystroke(e, 'keydown');
        });
        
        document.addEventListener('keyup', (e) => {
            this.recordKeystroke(e, 'keyup');
        });
        
        // Mouse monitoring
        document.addEventListener('mousemove', (e) => {
            this.recordMouseEvent(e, 'mousemove');
        });
        
        document.addEventListener('mousedown', (e) => {
            this.recordMouseEvent(e, 'mousedown');
        });
        
        document.addEventListener('mouseup', (e) => {
            this.recordMouseEvent(e, 'mouseup');
        });
        
        document.addEventListener('click', (e) => {
            this.recordMouseEvent(e, 'click');
        });
        
        document.addEventListener('wheel', (e) => {
            this.recordMouseEvent(e, 'wheel');
        });
        
        document.addEventListener('dblclick', (e) => {
            this.recordMouseEvent(e, 'dblclick');
        });
        
        // Context menu (right-click)
        document.addEventListener('contextmenu', (e) => {
            this.recordMouseEvent(e, 'contextmenu');
        });
    }
    
    recordKeystroke(event, type) {
        const keystrokeData = {
            type: type,
            key: event.key,
            code: event.code,
            timestamp: Date.now(),
            phase: this.currentPhase,
            target: event.target.id || event.target.className || 'unknown',
            shiftKey: event.shiftKey,
            ctrlKey: event.ctrlKey,
            altKey: event.altKey,
            metaKey: event.metaKey,
            location: event.location,
            repeat: event.repeat
        };
        
        this.behavioralData.keystrokes.push(keystrokeData);
        this.samplesCollected++;
        this.updateProgress();
        
        // Limit memory usage
        if (this.behavioralData.keystrokes.length > 2000) {
            this.behavioralData.keystrokes = this.behavioralData.keystrokes.slice(-1500);
        }
    }
    
    recordMouseEvent(event, type) {
        const mouseData = {
            type: type,
            x: event.clientX,
            y: event.clientY,
            timestamp: Date.now(),
            phase: this.currentPhase,
            target: event.target.tagName.toLowerCase(),
            button: event.button,
            buttons: event.buttons
        };
        
        // Add specific data for different event types
        if (type === 'wheel') {
            mouseData.deltaX = event.deltaX;
            mouseData.deltaY = event.deltaY;
            mouseData.deltaZ = event.deltaZ;
            mouseData.deltaMode = event.deltaMode;
        }
        
        if (type === 'click' || type === 'dblclick') {
            mouseData.detail = event.detail;
        }
        
        this.behavioralData.mouse_events.push(mouseData);
        this.samplesCollected++;
        this.updateProgress();
        
        // Limit memory usage
        if (this.behavioralData.mouse_events.length > 3000) {
            this.behavioralData.mouse_events = this.behavioralData.mouse_events.slice(-2000);
        }
    }
    
    startCalibration() {
        this.updateStatus('Starting calibration...', 'info');
        this.showPhase(0);
    }
    
    showPhase(phaseIndex) {
        if (phaseIndex >= this.phases.length) {
            this.completeCalibration();
            return;
        }
        
        this.currentPhase = phaseIndex;
        const phase = this.phases[phaseIndex];
        
        // Update UI
        this.currentTaskElement.textContent = `Phase ${phaseIndex + 1}: ${phase.name}`;
        this.instructionsElement.textContent = phase.instruction;
        
        // Clear previous content
        const contentArea = document.getElementById('phaseContent');
        contentArea.innerHTML = '';
        
        // Create phase-specific content
        switch (phase.type) {
            case 'free_text':
                this.createFreeTextPhase(contentArea, phase);
                break;
            case 'copy_text':
                this.createCopyTextPhase(contentArea, phase);
                break;
            case 'password_practice':
                this.createPasswordPhase(contentArea, phase);
                break;
            case 'mouse_interaction':
                this.createMousePhase(contentArea, phase);
                break;
        }
        
        this.updateStatus(`Phase ${phaseIndex + 1} of ${this.phases.length}`, 'info');
    }
    
    createFreeTextPhase(container, phase) {
        const textarea = document.createElement('textarea');
        textarea.id = 'freeTextInput';
        textarea.className = 'calibration-textarea';
        textarea.placeholder = 'Start typing here... Write about your day, your hobbies, or anything that interests you.';
        textarea.rows = 8;
        
        const charCounter = document.createElement('div');
        charCounter.className = 'char-counter';
        charCounter.textContent = `0 / ${phase.minChars} characters`;
        
        const progressDiv = document.createElement('div');
        progressDiv.className = 'phase-progress';
        progressDiv.innerHTML = `
            <div class="progress-bar">
                <div class="progress-fill" id="textProgress" style="width: 0%"></div>
            </div>
        `;
        
        container.appendChild(textarea);
        container.appendChild(charCounter);
        container.appendChild(progressDiv);
        
        textarea.addEventListener('input', () => {
            const length = textarea.value.length;
            const progress = Math.min(100, (length / phase.minChars) * 100);
            
            charCounter.textContent = `${length} / ${phase.minChars} characters`;
            document.getElementById('textProgress').style.width = `${progress}%`;
            
            if (length >= phase.minChars) {
                this.enableNextButton();
            } else {
                this.disableNextButton();
            }
        });
        
        textarea.focus();
    }
    
    createCopyTextPhase(container, phase) {
        const targetDiv = document.createElement('div');
        targetDiv.className = 'target-text';
        targetDiv.textContent = phase.targetText;
        
        const textarea = document.createElement('textarea');
        textarea.id = 'copyTextInput';
        textarea.className = 'calibration-textarea';
        textarea.placeholder = 'Type the text above exactly as shown...';
        textarea.rows = 6;
        
        const accuracyDiv = document.createElement('div');
        accuracyDiv.className = 'accuracy-indicator';
        accuracyDiv.textContent = 'Accuracy: 0%';
        
        const progressDiv = document.createElement('div');
        progressDiv.className = 'phase-progress';
        progressDiv.innerHTML = `
            <div class="progress-bar">
                <div class="progress-fill" id="copyProgress" style="width: 0%"></div>
            </div>
        `;
        
        container.appendChild(targetDiv);
        container.appendChild(textarea);
        container.appendChild(accuracyDiv);
        container.appendChild(progressDiv);
        
        textarea.addEventListener('input', () => {
            const typed = textarea.value;
            const target = phase.targetText;
            const accuracy = this.calculateAccuracy(typed, target);
            const progress = (typed.length / target.length) * 100;
            
            accuracyDiv.textContent = `Accuracy: ${accuracy.toFixed(1)}%`;
            document.getElementById('copyProgress').style.width = `${Math.min(100, progress)}%`;
            
            // Highlight errors
            this.highlightErrors(textarea, typed, target);
            
            if (typed.length >= target.length && accuracy >= 95) {
                this.enableNextButton();
            } else {
                this.disableNextButton();
            }
        });
        
        textarea.focus();
    }
    
    createPasswordPhase(container, phase) {
        const input = document.createElement('input');
        input.type = 'password';
        input.id = 'passwordInput';
        input.className = 'calibration-input';
        input.placeholder = 'Type a practice password...';
        
        const strengthDiv = document.createElement('div');
        strengthDiv.className = 'password-strength-indicator';
        
        const requirementsDiv = document.createElement('div');
        requirementsDiv.className = 'password-requirements';
        requirementsDiv.innerHTML = `
            <div class="requirement" id="req-length">✗ At least 12 characters</div>
            <div class="requirement" id="req-upper">✗ Uppercase letter</div>
            <div class="requirement" id="req-lower">✗ Lowercase letter</div>
            <div class="requirement" id="req-number">✗ Number</div>
            <div class="requirement" id="req-symbol">✗ Symbol</div>
        `;
        
        container.appendChild(input);
        container.appendChild(strengthDiv);
        container.appendChild(requirementsDiv);
        
        input.addEventListener('input', () => {
            const password = input.value;
            const strength = this.analyzePasswordStrength(password);
            
            this.updatePasswordRequirements(password);
            this.updatePasswordStrength(strengthDiv, strength);
            
            if (password.length >= phase.minChars && strength.score >= 4) {
                this.enableNextButton();
            } else {
                this.disableNextButton();
            }
        });
        
        input.focus();
    }
    
    createMousePhase(container, phase) {
        const instructionDiv = document.createElement('div');
        instructionDiv.className = 'mouse-instructions';
        instructionDiv.innerHTML = `
            <h3>Mouse Interaction Tasks</h3>
            <p>Perform the following actions naturally:</p>
            <ul>
                <li>Move your mouse around the screen</li>
                <li>Click on the colored targets below</li>
                <li>Scroll up and down</li>
                <li>Double-click on elements</li>
                <li>Right-click to open context menus</li>
            </ul>
        `;
        
        const targetsDiv = document.createElement('div');
        targetsDiv.className = 'mouse-targets';
        targetsDiv.id = 'mouseTargets';
        
        const timerDiv = document.createElement('div');
        timerDiv.className = 'phase-timer';
        timerDiv.id = 'mouseTimer';
        
        container.appendChild(instructionDiv);
        container.appendChild(targetsDiv);
        container.appendChild(timerDiv);
        
        this.createMouseTargets(targetsDiv);
        this.startMouseTimer(phase.duration);
    }
    
    setupMouseTargets() {
        // This will be called when mouse targets are created
    }
    
    createMouseTargets(container) {
        const colors = ['red', 'blue', 'green', 'orange', 'purple', 'teal'];
        const targetCount = 12;
        
        for (let i = 0; i < targetCount; i++) {
            const target = document.createElement('div');
            target.className = 'mouse-target';
            target.style.backgroundColor = colors[i % colors.length];
            target.style.left = Math.random() * 80 + 10 + '%';
            target.style.top = Math.random() * 60 + 20 + '%';
            target.dataset.targetId = i;
            
            target.addEventListener('click', (e) => {
                target.classList.add('clicked');
                this.createRippleEffect(e);
                setTimeout(() => {
                    target.style.left = Math.random() * 80 + 10 + '%';
                    target.style.top = Math.random() * 60 + 20 + '%';
                    target.classList.remove('clicked');
                }, 500);
            });
            
            target.addEventListener('dblclick', (e) => {
                target.classList.add('double-clicked');
                this.createStarEffect(e);
            });
            
            container.appendChild(target);
        }
        
        // Add scrollable content
        const scrollArea = document.createElement('div');
        scrollArea.className = 'scroll-area';
        scrollArea.innerHTML = `
            <div class="scroll-content">
                <h4>Scroll through this content</h4>
                ${Array(20).fill(0).map((_, i) => 
                    `<p>This is paragraph ${i + 1}. Scroll up and down to generate mouse wheel events for behavioral analysis.</p>`
                ).join('')}
            </div>
        `;
        container.appendChild(scrollArea);
    }
    
    createRippleEffect(event) {
        const ripple = document.createElement('div');
        ripple.className = 'ripple-effect';
        ripple.style.left = event.clientX + 'px';
        ripple.style.top = event.clientY + 'px';
        
        document.body.appendChild(ripple);
        
        setTimeout(() => {
            document.body.removeChild(ripple);
        }, 600);
    }
    
    createStarEffect(event) {
        for (let i = 0; i < 5; i++) {
            const star = document.createElement('div');
            star.className = 'star-effect';
            star.style.left = event.clientX + (Math.random() - 0.5) * 100 + 'px';
            star.style.top = event.clientY + (Math.random() - 0.5) * 100 + 'px';
            star.textContent = '★';
            
            document.body.appendChild(star);
            
            setTimeout(() => {
                document.body.removeChild(star);
            }, 1000);
        }
    }
    
    startMouseTimer(duration) {
        const timerElement = document.getElementById('mouseTimer');
        let timeLeft = duration / 1000;
        
        const updateTimer = () => {
            timerElement.textContent = `Time remaining: ${timeLeft}s`;
            
            if (timeLeft <= 0) {
                this.enableNextButton();
                timerElement.textContent = 'Phase completed!';
                return;
            }
            
            timeLeft--;
            setTimeout(updateTimer, 1000);
        };
        
        updateTimer();
    }
    
    calculateAccuracy(typed, target) {
        if (typed.length === 0) return 0;
        
        let matches = 0;
        const maxLength = Math.max(typed.length, target.length);
        
        for (let i = 0; i < maxLength; i++) {
            if (typed[i] === target[i]) {
                matches++;
            }
        }
        
        return (matches / maxLength) * 100;
    }
    
    highlightErrors(textarea, typed, target) {
        // This would require more complex implementation to highlight in textarea
        // For now, we'll use background color to indicate accuracy
        const accuracy = this.calculateAccuracy(typed, target);
        
        if (accuracy >= 95) {
            textarea.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
        } else if (accuracy >= 80) {
            textarea.style.backgroundColor = 'rgba(245, 158, 11, 0.1)';
        } else {
            textarea.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
        }
    }
    
    analyzePasswordStrength(password) {
        let score = 0;
        const checks = {
            length: password.length >= 12,
            upper: /[A-Z]/.test(password),
            lower: /[a-z]/.test(password),
            number: /[0-9]/.test(password),
            symbol: /[^A-Za-z0-9]/.test(password)
        };
        
        Object.values(checks).forEach(check => {
            if (check) score++;
        });
        
        return { score, checks };
    }
    
    updatePasswordRequirements(password) {
        const requirements = {
            'req-length': password.length >= 12,
            'req-upper': /[A-Z]/.test(password),
            'req-lower': /[a-z]/.test(password),
            'req-number': /[0-9]/.test(password),
            'req-symbol': /[^A-Za-z0-9]/.test(password)
        };
        
        Object.entries(requirements).forEach(([id, met]) => {
            const element = document.getElementById(id);
            element.textContent = (met ? '✓' : '✗') + element.textContent.substring(1);
            element.className = met ? 'requirement met' : 'requirement';
        });
    }
    
    updatePasswordStrength(container, strength) {
        const levels = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong'];
        const colors = ['#ef4444', '#f59e0b', '#eab308', '#22c55e', '#16a34a'];
        
        container.innerHTML = `
            <div class="strength-bar">
                <div class="strength-fill" style="width: ${strength.score * 20}%; background-color: ${colors[strength.score - 1] || colors[0]}"></div>
            </div>
            <div class="strength-label">${levels[strength.score - 1] || levels[0]}</div>
        `;
    }
    
    updateProgress() {
        const totalProgress = ((this.currentPhase / this.totalPhases) + 
                             (this.samplesCollected / this.targetSamples / this.totalPhases)) * 100;
        
        this.progressBar.style.width = Math.min(100, totalProgress) + '%';
        this.progressText.textContent = `${Math.floor(totalProgress)}% Complete`;
    }
    
    updateStatus(message, type = 'info') {
        this.statusElement.textContent = message;
        this.statusElement.className = `status status-${type}`;
    }
    
    enableNextButton() {
        const nextButton = document.getElementById('nextButton');
        if (nextButton) {
            nextButton.disabled = false;
            nextButton.classList.remove('disabled');
        }
    }
    
    disableNextButton() {
        const nextButton = document.getElementById('nextButton');
        if (nextButton) {
            nextButton.disabled = true;
            nextButton.classList.add('disabled');
        }
    }
    
    skipPhase() {
        if (confirm('Are you sure you want to skip this phase? This may affect the accuracy of your behavioral profile.')) {
            this.nextPhase();
        }
    }
    
    nextPhase() {
        if (this.currentPhase < this.phases.length - 1) {
            this.showPhase(this.currentPhase + 1);
            this.disableNextButton();
        } else {
            this.completeCalibration();
        }
    }
    
    async completeCalibration() {
        this.updateStatus('Processing calibration data...', 'info');
        
        // Show completion animation
        this.calibrationContainer.classList.add('calibration-complete');
        
        try {
            const response = await fetch('/api/calibration', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
                },
                body: JSON.stringify({
                    behavioral_data: this.behavioralData,
                    calibration_summary: this.getCalibrationSummary()
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.updateStatus('Calibration completed successfully!', 'success');
                
                // Show training status
                this.showTrainingStatus();
                
            } else {
                this.updateStatus('Calibration failed: ' + result.message, 'error');
            }
            
        } catch (error) {
            console.error('Calibration error:', error);
            this.updateStatus('Connection error during calibration.', 'error');
        }
    }
    
    async showTrainingStatus() {
        const statusContainer = document.createElement('div');
        statusContainer.className = 'training-status';
        statusContainer.innerHTML = `
            <div class="training-progress">
                <h3>Training Your Behavioral Models</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="trainingProgress" style="width: 0%"></div>
                </div>
                <div class="progress-text" id="trainingText">Initializing...</div>
            </div>
        `;
        
        this.calibrationContainer.appendChild(statusContainer);
        
        // Poll for training status
        const pollTrainingStatus = async () => {
            try {
                const response = await fetch('/api/training-status', {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
                    }
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const status = result.status;
                    const progressBar = document.getElementById('trainingProgress');
                    const progressText = document.getElementById('trainingText');
                    
                    if (progressBar && progressText) {
                        progressBar.style.width = status.progress + '%';
                        
                        if (status.status === 'training') {
                            progressText.textContent = `Training models... ${status.progress}%`;
                            setTimeout(pollTrainingStatus, 2000);
                        } else if (status.status === 'completed') {
                            progressText.textContent = 'Training completed!';
                            setTimeout(() => {
                                window.location.href = '/dashboard';
                            }, 2000);
                        } else if (status.status === 'failed') {
                            progressText.textContent = 'Training failed. Please try again.';
                        }
                    }
                }
                
            } catch (error) {
                console.error('Error polling training status:', error);
            }
        };
        
        pollTrainingStatus();
    }
    
    getCalibrationSummary() {
        return {
            phases_completed: this.currentPhase + 1,
            total_keystrokes: this.behavioralData.keystrokes.length,
            total_mouse_events: this.behavioralData.mouse_events.length,
            session_duration: Date.now() - this.behavioralData.session_start,
            typing_analysis: this.analyzeTypingBehavior(),
            mouse_analysis: this.analyzeMouseBehavior(),
            completion_time: new Date().toISOString()
        };
    }
    
    analyzeTypingBehavior() {
        const keydowns = this.behavioralData.keystrokes.filter(k => k.type === 'keydown');
        
        if (keydowns.length < 2) return {};
        
        const intervals = [];
        for (let i = 1; i < keydowns.length; i++) {
            intervals.push(keydowns[i].timestamp - keydowns[i-1].timestamp);
        }
        
        return {
            average_typing_interval: intervals.reduce((a, b) => a + b, 0) / intervals.length,
            typing_variance: this.calculateVariance(intervals),
            special_key_usage: this.countSpecialKeys(keydowns),
            typing_rhythm_score: this.calculateTypingRhythm(intervals)
        };
    }
    
    analyzeMouseBehavior() {
        const moves = this.behavioralData.mouse_events.filter(m => m.type === 'mousemove');
        
        if (moves.length < 2) return {};
        
        const velocities = [];
        const distances = [];
        
        for (let i = 1; i < moves.length; i++) {
            const dx = moves[i].x - moves[i-1].x;
            const dy = moves[i].y - moves[i-1].y;
            const dt = moves[i].timestamp - moves[i-1].timestamp;
            
            const distance = Math.sqrt(dx*dx + dy*dy);
            const velocity = dt > 0 ? distance / dt : 0;
            
            distances.push(distance);
            velocities.push(velocity);
        }
        
        return {
            average_velocity: velocities.reduce((a, b) => a + b, 0) / velocities.length,
            movement_variance: this.calculateVariance(distances),
            click_patterns: this.analyzeClickPatterns(),
            scroll_behavior: this.analyzeScrollBehavior()
        };
    }
    
    calculateVariance(values) {
        if (values.length < 2) return 0;
        
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        
        return variance;
    }
    
    countSpecialKeys(keydowns) {
        const specialKeys = {
            shift: 0,
            ctrl: 0,
            alt: 0,
            meta: 0,
            backspace: 0,
            enter: 0
        };
        
        keydowns.forEach(key => {
            if (key.shiftKey) specialKeys.shift++;
            if (key.ctrlKey) specialKeys.ctrl++;
            if (key.altKey) specialKeys.alt++;
            if (key.metaKey) specialKeys.meta++;
            if (key.key === 'Backspace') specialKeys.backspace++;
            if (key.key === 'Enter') specialKeys.enter++;
        });
        
        return specialKeys;
    }
    
    calculateTypingRhythm(intervals) {
        if (intervals.length < 3) return 0;
        
        let consistentPairs = 0;
        for (let i = 1; i < intervals.length - 1; i++) {
            const prev = intervals[i-1];
            const curr = intervals[i];
            const next = intervals[i+1];
            
            if (Math.abs(curr - prev) < 50 && Math.abs(next - curr) < 50) {
                consistentPairs++;
            }
        }
        
        return consistentPairs / (intervals.length - 2);
    }
    
    analyzeClickPatterns() {
        const clicks = this.behavioralData.mouse_events.filter(m => m.type === 'click');
        const doubleClicks = this.behavioralData.mouse_events.filter(m => m.type === 'dblclick');
        
        return {
            total_clicks: clicks.length,
            double_clicks: doubleClicks.length,
            click_timing: this.analyzeClickTiming(clicks)
        };
    }
    
    analyzeClickTiming(clicks) {
        if (clicks.length < 2) return {};
        
        const intervals = [];
        for (let i = 1; i < clicks.length; i++) {
            intervals.push(clicks[i].timestamp - clicks[i-1].timestamp);
        }
        
        return {
            average_interval: intervals.reduce((a, b) => a + b, 0) / intervals.length,
            interval_variance: this.calculateVariance(intervals)
        };
    }
    
    analyzeScrollBehavior() {
        const scrolls = this.behavioralData.mouse_events.filter(m => m.type === 'wheel');
        
        return {
            total_scrolls: scrolls.length,
            scroll_directions: this.countScrollDirections(scrolls),
            scroll_intensity: this.calculateScrollIntensity(scrolls)
        };
    }
    
    countScrollDirections(scrolls) {
        const directions = { up: 0, down: 0, left: 0, right: 0 };
        
        scrolls.forEach(scroll => {
            if (scroll.deltaY < 0) directions.up++;
            if (scroll.deltaY > 0) directions.down++;
            if (scroll.deltaX < 0) directions.left++;
            if (scroll.deltaX > 0) directions.right++;
        });
        
        return directions;
    }
    
    calculateScrollIntensity(scrolls) {
        if (scrolls.length === 0) return 0;
        
        const intensities = scrolls.map(scroll => 
            Math.sqrt(scroll.deltaX*scroll.deltaX + scroll.deltaY*scroll.deltaY)
        );
        
        return intensities.reduce((a, b) => a + b, 0) / intensities.length;
    }
}

// Additional CSS for calibration-specific styles
const calibrationStyles = `
<style>
.calibration-textarea, .calibration-input {
    width: 100%;
    padding: 1rem;
    border: 2px solid var(--gray-200);
    border-radius: var(--radius-md);
    font-size: 1rem;
    font-family: inherit;
    background: var(--bg-primary);
    color: var(--text-primary);
    transition: all var(--transition-fast);
    resize: vertical;
}

.calibration-textarea:focus, .calibration-input:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.char-counter {
    margin-top: 0.5rem;
    font-size: 0.875rem;
    color: var(--text-secondary);
    text-align: right;
}

.target-text {
    background: var(--bg-secondary);
    padding: 1rem;
    border-radius: var(--radius-md);
    margin-bottom: 1rem;
    border-left: 4px solid var(--primary-color);
    font-family: monospace;
    line-height: 1.6;
}

.accuracy-indicator {
    margin-top: 0.5rem;
    font-weight: 600;
    color: var(--primary-color);
}

.password-requirements {
    margin-top: 1rem;
    padding: 1rem;
    background: var(--bg-secondary);
    border-radius: var(--radius-md);
}

.requirement {
    padding: 0.25rem 0;
    font-size: 0.875rem;
    color: var(--error-color);
}

.requirement.met {
    color: var(--success-color);
}

.password-strength-indicator {
    margin-top: 0.5rem;
}

.strength-bar {
    height: 6px;
    background: var(--gray-200);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}

.strength-fill {
    height: 100%;
    transition: all var(--transition-normal);
    border-radius: 3px;
}

.strength-label {
    font-size: 0.875rem;
    font-weight: 600;
}

.mouse-targets {
    position: relative;
    height: 400px;
    background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
    border-radius: var(--radius-lg);
    margin: 1rem 0;
    overflow: hidden;
}

.mouse-target {
    position: absolute;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    cursor: pointer;
    transition: all var(--transition-fast);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    box-shadow: var(--shadow-md);
}

.mouse-target:hover {
    transform: scale(1.1);
    box-shadow: var(--shadow-lg);
}

.mouse-target.clicked {
    transform: scale(1.3);
    box-shadow: 0 0 20px currentColor;
}

.mouse-target.double-clicked {
    animation: bounce 0.6s ease-in-out;
}

@keyframes bounce {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.5); }
}

.scroll-area {
    height: 200px;
    overflow-y: auto;
    background: var(--bg-primary);
    border-radius: var(--radius-md);
    padding: 1rem;
    margin-top: 1rem;
    border: 1px solid var(--gray-200);
}

.phase-timer {
    text-align: center;
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--primary-color);
    margin-top: 1rem;
}

.ripple-effect {
    position: fixed;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: rgba(37, 99, 235, 0.6);
    pointer-events: none;
    animation: ripple 0.6s ease-out;
    z-index: 1000;
}

@keyframes ripple {
    0% {
        transform: scale(0);
        opacity: 1;
    }
    100% {
        transform: scale(4);
        opacity: 0;
    }
}

.star-effect {
    position: fixed;
    font-size: 1.5rem;
    color: gold;
    pointer-events: none;
    animation: starFall 1s ease-out;
    z-index: 1000;
}

@keyframes starFall {
    0% {
        transform: translateY(0) rotate(0deg);
        opacity: 1;
    }
    100% {
        transform: translateY(50px) rotate(360deg);
        opacity: 0;
    }
}

.calibration-complete {
    animation: completePulse 1s ease-in-out;
}

@keyframes completePulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

.training-status {
    background: var(--bg-secondary);
    padding: 2rem;
    border-radius: var(--radius-lg);
    margin-top: 2rem;
    text-align: center;
}

.training-progress h3 {
    margin-bottom: 1.5rem;
    color: var(--primary-color);
}

.phase-progress {
    margin-top: 1rem;
}

.disabled {
    opacity: 0.6;
    cursor: not-allowed;
}
</style>
`;

// Add styles to document head
document.head.insertAdjacentHTML('beforeend', calibrationStyles);

// Initialize calibration manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CalibrationManager();
});

// Export for testing
window.CalibrationManager = CalibrationManager;