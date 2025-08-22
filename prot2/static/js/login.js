/**
 * Login Page JavaScript - FIXED VERSION
 * Handles user authentication and initial behavioral data collection
 */

class LoginManager {
    constructor() {
        this.loginForm = document.getElementById('loginForm');
        this.usernameInput = document.getElementById('username');
        this.passwordInput = document.getElementById('password');
        this.loginButton = document.getElementById('loginButton');
        this.messageContainer = document.getElementById('messageContainer');
        
        // Behavioral data collection
        this.behavioralData = {
            keystrokes: [],
            mouse_events: [],
            session_start: Date.now()
        };
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.startBehavioralCollection();
        this.setupFormValidation();
        this.addVisualEnhancements();
        
        console.log('LoginManager initialized');
    }
    
    setupEventListeners() {
        // Form submission
        this.loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        
        // Real-time form validation
        this.usernameInput.addEventListener('input', () => this.validateForm());
        this.passwordInput.addEventListener('input', () => this.validateForm());
        
        // Enter key handling
        this.usernameInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.passwordInput.focus();
            }
        });
        
        this.passwordInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleLogin(e);
            }
        });
        
        // Visual feedback on focus
        [this.usernameInput, this.passwordInput].forEach(input => {
            input.addEventListener('focus', () => {
                input.parentElement.classList.add('focused');
            });
            
            input.addEventListener('blur', () => {
                input.parentElement.classList.remove('focused');
            });
        });
    }
    
    startBehavioralCollection() {
        console.log('Starting behavioral data collection for login...');
        
        // Keystroke data collection - capture all keystrokes on page
        document.addEventListener('keydown', (e) => {
            this.recordKeystroke(e, 'keydown');
        });
        
        document.addEventListener('keyup', (e) => {
            this.recordKeystroke(e, 'keyup');
        });
        
        // Mouse data collection - capture all mouse events on page
        document.addEventListener('mousemove', (e) => {
            // Throttle mousemove events to prevent overwhelming data
            if (Math.random() < 0.2) { // Sample 20% of mousemove events
                this.recordMouseEvent(e, 'mousemove');
            }
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
    }
    
    recordKeystroke(event, type) {
        // Record keystrokes from any element on the page for better behavioral profiling
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
            repeat: event.repeat
        };
        
        this.behavioralData.keystrokes.push(keystrokeData);
        
        // Limit data collection to prevent memory issues
        if (this.behavioralData.keystrokes.length > 500) {
            this.behavioralData.keystrokes = this.behavioralData.keystrokes.slice(-400);
        }
        
        console.debug(`Recorded keystroke: ${type} - ${event.key}`);
    }
    
    recordMouseEvent(event, type) {
        const mouseData = {
            type: type,
            x: event.clientX,
            y: event.clientY,
            timestamp: Date.now(),
            target: this.getTargetInfo(event.target),
            button: event.button,
            buttons: event.buttons
        };
        
        // Add additional data for specific event types
        if (type === 'click') {
            mouseData.detail = event.detail; // Click count
        }
        
        if (type === 'wheel') {
            mouseData.deltaX = event.deltaX;
            mouseData.deltaY = event.deltaY;
            mouseData.deltaZ = event.deltaZ;
        }
        
        this.behavioralData.mouse_events.push(mouseData);
        
        // Limit data collection
        if (this.behavioralData.mouse_events.length > 1000) {
            this.behavioralData.mouse_events = this.behavioralData.mouse_events.slice(-800);
        }
        
        console.debug(`Recorded mouse event: ${type} at (${event.clientX}, ${event.clientY})`);
    }
    
    getTargetInfo(target) {
        return {
            tagName: target.tagName?.toLowerCase() || 'unknown',
            id: target.id || null,
            className: target.className || null,
            type: target.type || null
        };
    }
    
    setupFormValidation() {
        const validateField = (field, minLength = 1) => {
            const value = field.value.trim();
            const isValid = value.length >= minLength;
            
            field.classList.toggle('invalid', !isValid && value.length > 0);
            field.classList.toggle('valid', isValid);
            
            return isValid;
        };
        
        this.usernameInput.addEventListener('input', () => {
            validateField(this.usernameInput, 3);
            this.validateForm();
        });
        
        this.passwordInput.addEventListener('input', () => {
            validateField(this.passwordInput, 6);
            this.validateForm();
        });
    }
    
    validateForm() {
        const username = this.usernameInput.value.trim();
        const password = this.passwordInput.value.trim();
        
        const isValid = username.length >= 3 && password.length >= 6;
        
        this.loginButton.disabled = !isValid;
        this.loginButton.classList.toggle('btn-disabled', !isValid);
        
        return isValid;
    }
    
    addVisualEnhancements() {
        // Add floating label effect
        [this.usernameInput, this.passwordInput].forEach(input => {
            const updateLabel = () => {
                const hasValue = input.value.length > 0;
                input.parentElement.classList.toggle('has-value', hasValue);
            };
            
            input.addEventListener('input', updateLabel);
            input.addEventListener('blur', updateLabel);
            
            // Initial check
            updateLabel();
        });
        
        // Add password strength indicator
        this.addPasswordStrengthIndicator();
        
        // Add animated placeholder effect
        this.addAnimatedPlaceholders();
    }
    
    addPasswordStrengthIndicator() {
        const strengthIndicator = document.createElement('div');
        strengthIndicator.className = 'password-strength';
        strengthIndicator.innerHTML = `
            <div class="strength-bar">
                <div class="strength-fill"></div>
            </div>
            <div class="strength-text">Password Strength</div>
        `;
        
        this.passwordInput.parentElement.appendChild(strengthIndicator);
        
        this.passwordInput.addEventListener('input', () => {
            const password = this.passwordInput.value;
            const strength = this.calculatePasswordStrength(password);
            
            const fill = strengthIndicator.querySelector('.strength-fill');
            const text = strengthIndicator.querySelector('.strength-text');
            
            fill.style.width = `${strength.percentage}%`;
            fill.className = `strength-fill strength-${strength.level}`;
            text.textContent = `Password Strength: ${strength.label}`;
            
            strengthIndicator.style.opacity = password.length > 0 ? '1' : '0';
        });
    }
    
    calculatePasswordStrength(password) {
        let score = 0;
        let feedback = [];
        
        // Length check
        if (password.length >= 8) score += 25;
        else feedback.push('8+ characters');
        
        // Uppercase check
        if (/[A-Z]/.test(password)) score += 25;
        else feedback.push('uppercase letter');
        
        // Lowercase check
        if (/[a-z]/.test(password)) score += 25;
        else feedback.push('lowercase letter');
        
        // Number or special character check
        if (/[0-9]/.test(password) || /[^A-Za-z0-9]/.test(password)) score += 25;
        else feedback.push('number or special character');
        
        let level, label;
        if (score <= 25) {
            level = 'weak';
            label = 'Weak';
        } else if (score <= 50) {
            level = 'fair';
            label = 'Fair';
        } else if (score <= 75) {
            level = 'good';
            label = 'Good';
        } else {
            level = 'strong';
            label = 'Strong';
        }
        
        return {
            percentage: score,
            level: level,
            label: label,
            feedback: feedback
        };
    }
    
    addAnimatedPlaceholders() {
        const placeholders = [
            'Enter your username...',
            'Type your username here...',
            'Username or email...'
        ];
        
        let currentIndex = 0;
        const cyclePlaceholder = () => {
            if (this.usernameInput.value.length === 0) {
                this.usernameInput.placeholder = placeholders[currentIndex];
                currentIndex = (currentIndex + 1) % placeholders.length;
            }
        };
        
        setInterval(cyclePlaceholder, 3000);
    }
    
    async handleLogin(event) {
        event.preventDefault();
        
        if (!this.validateForm()) {
            this.showMessage('Please fill in all required fields correctly.', 'error');
            return;
        }
        
        const username = this.usernameInput.value.trim();
        const password = this.passwordInput.value.trim();
        
        console.log('Attempting login...', {
            username,
            keystrokeCount: this.behavioralData.keystrokes.length,
            mouseEventCount: this.behavioralData.mouse_events.length
        });
        
        // Show loading state
        this.setLoadingState(true);
        
        try {
            // Prepare behavioral data summary
            const behavioralSummary = this.getBehavioralSummary();
            console.log('Behavioral data summary:', behavioralSummary);
            
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    password: password,
                    behavioral_data: {
                        keystrokes: this.behavioralData.keystrokes,
                        mouse_events: this.behavioralData.mouse_events,
                        session_start: this.behavioralData.session_start,
                        session_duration: Date.now() - this.behavioralData.session_start,
                        summary: behavioralSummary
                    }
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('Login response:', result);
            
            if (result.success) {
                // Store authentication token
                if (result.token) {
                    localStorage.setItem('auth_token', result.token);
                    console.log('Token stored successfully');
                }
                
                this.showMessage(result.message, 'success');
                
                // Add success animation
                this.loginForm.classList.add('login-success');
                
                // Redirect after animation
                setTimeout(() => {
                    console.log('Redirecting to:', result.redirect);
                    window.location.href = result.redirect;
                }, 1500);
                
            } else {
                this.showMessage(result.message, 'error');
                this.shakeForm();
                
                // Clear password on failed login
                this.passwordInput.value = '';
                this.passwordInput.focus();
            }
            
        } catch (error) {
            console.error('Login error:', error);
            this.showMessage('Connection error. Please try again.', 'error');
            this.shakeForm();
        } finally {
            this.setLoadingState(false);
        }
    }
    
    setLoadingState(loading) {
        this.loginButton.disabled = loading;
        this.loginButton.classList.toggle('loading', loading);
        
        if (loading) {
            this.loginButton.innerHTML = `
                <span class="btn-text">Authenticating...</span>
            `;
        } else {
            this.loginButton.innerHTML = `
                <span class="btn-text">Sign In</span>
            `;
        }
        
        // Disable form inputs during loading
        this.usernameInput.disabled = loading;
        this.passwordInput.disabled = loading;
    }
    
    showMessage(message, type = 'info') {
        // Clear existing messages
        this.messageContainer.innerHTML = '';
        
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.innerHTML = `
            <div class="alert-content">
                <div class="alert-icon">
                    ${this.getAlertIcon(type)}
                </div>
                <div class="alert-message">${message}</div>
            </div>
        `;
        
        this.messageContainer.appendChild(alertDiv);
        
        // Auto-hide success messages
        if (type === 'success') {
            setTimeout(() => {
                alertDiv.style.opacity = '0';
                setTimeout(() => {
                    if (alertDiv.parentNode) {
                        alertDiv.parentNode.removeChild(alertDiv);
                    }
                }, 300);
            }, 3000);
        }
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
    
    shakeForm() {
        this.loginForm.classList.add('shake');
        setTimeout(() => {
            this.loginForm.classList.remove('shake');
        }, 600);
    }
    
    // Utility method to get behavioral data summary
    getBehavioralSummary() {
        const keystrokeCount = this.behavioralData.keystrokes.length;
        const mouseEventCount = this.behavioralData.mouse_events.length;
        const sessionDuration = Date.now() - this.behavioralData.session_start;
        
        return {
            keystroke_count: keystrokeCount,
            mouse_event_count: mouseEventCount,
            session_duration: sessionDuration,
            typing_patterns: this.analyzeTypingPatterns(),
            mouse_patterns: this.analyzeMousePatterns(),
            data_quality: this.assessDataQuality()
        };
    }
    
    analyzeTypingPatterns() {
        const keydowns = this.behavioralData.keystrokes.filter(k => k.type === 'keydown');
        if (keydowns.length < 2) return { insufficient_data: true };
        
        const intervals = [];
        for (let i = 1; i < keydowns.length; i++) {
            intervals.push(keydowns[i].timestamp - keydowns[i-1].timestamp);
        }
        
        return {
            average_interval: intervals.reduce((a, b) => a + b, 0) / intervals.length,
            typing_rhythm: this.calculateTypingRhythm(intervals),
            special_keys: this.countSpecialKeys(keydowns)
        };
    }
    
    analyzeMousePatterns() {
        const moves = this.behavioralData.mouse_events.filter(m => m.type === 'mousemove');
        if (moves.length < 2) return { insufficient_data: true };
        
        const distances = [];
        const velocities = [];
        
        for (let i = 1; i < moves.length; i++) {
            const dx = moves[i].x - moves[i-1].x;
            const dy = moves[i].y - moves[i-1].y;
            const dt = moves[i].timestamp - moves[i-1].timestamp;
            
            const distance = Math.sqrt(dx*dx + dy*dy);
            distances.push(distance);
            
            if (dt > 0) {
                velocities.push(distance / dt);
            }
        }
        
        return {
            average_movement: distances.reduce((a, b) => a + b, 0) / distances.length,
            average_velocity: velocities.reduce((a, b) => a + b, 0) / velocities.length,
            movement_variability: this.calculateVariability(distances),
            click_patterns: this.analyzeClickPatterns()
        };
    }
    
    calculateTypingRhythm(intervals) {
        if (intervals.length < 3) return 0;
        
        let rhythmScore = 0;
        for (let i = 1; i < intervals.length - 1; i++) {
            const prev = intervals[i-1];
            const curr = intervals[i];
            const next = intervals[i+1];
            
            // Check for consistent rhythm patterns
            if (Math.abs(curr - prev) < 50 && Math.abs(next - curr) < 50) {
                rhythmScore++;
            }
        }
        
        return rhythmScore / (intervals.length - 2);
    }
    
    countSpecialKeys(keydowns) {
        const specialKeys = {
            shift: 0,
            ctrl: 0,
            alt: 0,
            meta: 0,
            backspace: 0,
            enter: 0,
            space: 0
        };
        
        keydowns.forEach(key => {
            if (key.shiftKey) specialKeys.shift++;
            if (key.ctrlKey) specialKeys.ctrl++;
            if (key.altKey) specialKeys.alt++;
            if (key.metaKey) specialKeys.meta++;
            if (key.key === 'Backspace') specialKeys.backspace++;
            if (key.key === 'Enter') specialKeys.enter++;
            if (key.key === ' ') specialKeys.space++;
        });
        
        return specialKeys;
    }
    
    calculateVariability(values) {
        if (values.length < 2) return 0;
        
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        
        return Math.sqrt(variance);
    }
    
    analyzeClickPatterns() {
        const clicks = this.behavioralData.mouse_events.filter(m => m.type === 'click');
        const doubleClicks = clicks.filter(c => c.detail >= 2);
        
        return {
            total_clicks: clicks.length,
            double_clicks: doubleClicks.length,
            click_timing: this.analyzeClickTiming(clicks)
        };
    }
    
    analyzeClickTiming(clicks) {
        if (clicks.length < 2) return { insufficient_data: true };
        
        const intervals = [];
        for (let i = 1; i < clicks.length; i++) {
            intervals.push(clicks[i].timestamp - clicks[i-1].timestamp);
        }
        
        return {
            average_interval: intervals.reduce((a, b) => a + b, 0) / intervals.length,
            interval_variance: this.calculateVariability(intervals)
        };
    }
    
    assessDataQuality() {
        const keystrokeCount = this.behavioralData.keystrokes.length;
        const mouseEventCount = this.behavioralData.mouse_events.length;
        const sessionDuration = Date.now() - this.behavioralData.session_start;
        
        let quality = 'poor';
        
        if (keystrokeCount >= 50 && mouseEventCount >= 20 && sessionDuration >= 10000) {
            quality = 'excellent';
        } else if (keystrokeCount >= 20 && mouseEventCount >= 10 && sessionDuration >= 5000) {
            quality = 'good';
        } else if (keystrokeCount >= 10 || mouseEventCount >= 5) {
            quality = 'fair';
        }
        
        return {
            quality: quality,
            keystroke_count: keystrokeCount,
            mouse_event_count: mouseEventCount,
            session_duration: sessionDuration
        };
    }
}

// CSS for additional animations and effects
const additionalStyles = `
<style>
.form-group.focused .form-input {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.form-input.valid {
    border-color: var(--success-color);
}

.form-input.invalid {
    border-color: var(--error-color);
    box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1);
}

.password-strength {
    margin-top: 0.5rem;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.strength-bar {
    height: 4px;
    background: var(--gray-200);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 0.25rem;
}

.strength-fill {
    height: 100%;
    transition: all 0.3s ease;
    border-radius: 2px;
}

.strength-fill.strength-weak {
    background: var(--error-color);
}

.strength-fill.strength-fair {
    background: var(--warning-color);
}

.strength-fill.strength-good {
    background: var(--info-color);
}

.strength-fill.strength-strong {
    background: var(--success-color);
}

.strength-text {
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.login-success {
    animation: successPulse 0.6s ease-out;
}

@keyframes successPulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

.shake {
    animation: shake 0.6s ease-in-out;
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}

.alert {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    animation: slideInDown 0.3s ease-out;
}

.alert-content {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
}

.alert-icon {
    font-weight: bold;
    font-size: 1.1rem;
}

.btn-disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.form-group.has-value .form-label {
    transform: translateY(-0.5rem) scale(0.85);
    color: var(--primary-color);
}
</style>
`;

// Add styles to document head
document.head.insertAdjacentHTML('beforeend', additionalStyles);

// Initialize login manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing LoginManager...');
    new LoginManager();
});

// Export for testing purposes
window.LoginManager = LoginManager;