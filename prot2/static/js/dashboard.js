/**
 * Dashboard Page JavaScript - FIXED VERSION
 * Handles real-time behavioral monitoring and dashboard visualization
 */

class DashboardManager {
    constructor() {
        this.socket = null;
        this.isMonitoring = false;
        this.monitoringInterval = null;
        
        // Behavioral data collection
        this.behavioralBuffer = {
            keystrokes: [],
            mouse_events: []
        };
        
        // Data batching configuration
        this.batchSize = 15000; // 15 seconds (reduced for more frequent updates)
        this.lastBatchSent = Date.now();
        
        // Chart instances
        this.charts = {};
        
        // Dashboard state
        this.stats = {
            session_start: Date.now(),
            total_keystrokes: 0,
            total_mouse_events: 0,
            anomalies_detected: 0,
            current_status: 'monitoring'
        };
        
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.startBehavioralMonitoring();
        this.initializeCharts();
        this.loadUserStats();
        this.setupDashboardUpdates();
    }
    
    setupWebSocket() {
        // Connect to WebSocket for real-time communication
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/socket.io/`;
        
        const token = localStorage.getItem('auth_token');
        
        this.socket = io(wsUrl, {
            auth: {
                token: token
            },
            query: {
                token: token
            }
        });
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
            this.showNotification('Connected to monitoring server', 'success');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
            this.showNotification('Disconnected from monitoring server', 'warning');
        });
        
        this.socket.on('analysis_result', (data) => {
            this.handleAnalysisResult(data);
        });
        
        this.socket.on('challenge_required', (data) => {
            this.handleChallengeRequired(data);
        });
        
        this.socket.on('security_alert', (data) => {
            this.showSecurityAlert(data);
        });
        
        this.socket.on('force_logout', (data) => {
            this.handleForceLogout(data);
        });
        
        this.socket.on('training_status', (data) => {
            this.updateTrainingStatus(data);
        });
        
        this.socket.on('connection_status', (data) => {
            console.log('Connection status received:', data);
        });
        
        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            this.showNotification('Connection error', 'error');
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('Socket connection error:', error);
            this.showNotification('Failed to connect to monitoring server', 'error');
        });
    }
    
    setupEventListeners() {
        // Logout button
        const logoutBtn = document.getElementById('logoutBtn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.logout());
        }
        
        // Monitoring toggle
        const monitoringToggle = document.getElementById('monitoringToggle');
        if (monitoringToggle) {
            monitoringToggle.addEventListener('change', (e) => {
                this.toggleMonitoring(e.target.checked);
            });
        }
        
        // Settings button
        const settingsBtn = document.getElementById('settingsBtn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => this.showSettings());
        }
        
        // Export data button
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportData());
        }
        
        // Window focus/blur for monitoring
        window.addEventListener('focus', () => {
            if (this.isMonitoring) {
                console.log('Window focused, resuming monitoring');
                this.resumeMonitoring();
            }
        });
        
        window.addEventListener('blur', () => {
            console.log('Window blurred, pausing monitoring');
            this.pauseMonitoring();
        });
        
        // Page visibility change
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseMonitoring();
            } else if (this.isMonitoring) {
                this.resumeMonitoring();
            }
        });
    }
    
    startBehavioralMonitoring() {
        console.log('Starting behavioral monitoring...');
        
        // Global keystroke monitoring
        document.addEventListener('keydown', (e) => {
            if (this.isMonitoring) {
                this.recordKeystroke(e, 'keydown');
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (this.isMonitoring) {
                this.recordKeystroke(e, 'keyup');
            }
        });
        
        // Global mouse monitoring
        document.addEventListener('mousemove', (e) => {
            if (this.isMonitoring) {
                // Throttle mousemove events
                if (Math.random() < 0.1) { // Sample 10% of mousemove events
                    this.recordMouseEvent(e, 'mousemove');
                }
            }
        });
        
        document.addEventListener('mousedown', (e) => {
            if (this.isMonitoring) {
                this.recordMouseEvent(e, 'mousedown');
            }
        });
        
        document.addEventListener('mouseup', (e) => {
            if (this.isMonitoring) {
                this.recordMouseEvent(e, 'mouseup');
            }
        });
        
        document.addEventListener('click', (e) => {
            if (this.isMonitoring) {
                this.recordMouseEvent(e, 'click');
            }
        });
        
        document.addEventListener('wheel', (e) => {
            if (this.isMonitoring) {
                this.recordMouseEvent(e, 'wheel');
            }
        });
        
        // Start monitoring by default
        this.toggleMonitoring(true);
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
            repeat: event.repeat
        };
        
        this.behavioralBuffer.keystrokes.push(keystrokeData);
        this.stats.total_keystrokes++;
        
        // Update real-time stats
        this.updateRealtimeStats();
        
        // Check if batch should be sent
        this.checkBatchSend();
        
        // Limit buffer size
        if (this.behavioralBuffer.keystrokes.length > 1000) {
            this.behavioralBuffer.keystrokes = this.behavioralBuffer.keystrokes.slice(-800);
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
            buttons: event.buttons
        };
        
        // Add event-specific data
        if (type === 'wheel') {
            mouseData.deltaX = event.deltaX;
            mouseData.deltaY = event.deltaY;
            mouseData.deltaZ = event.deltaZ;
            mouseData.deltaMode = event.deltaMode;
        }
        
        if (type === 'click') {
            mouseData.detail = event.detail;
        }
        
        this.behavioralBuffer.mouse_events.push(mouseData);
        this.stats.total_mouse_events++;
        
        // Update real-time stats
        this.updateRealtimeStats();
        
        // Check if batch should be sent
        this.checkBatchSend();
        
        // Limit buffer size
        if (this.behavioralBuffer.mouse_events.length > 1500) {
            this.behavioralBuffer.mouse_events = this.behavioralBuffer.mouse_events.slice(-1000);
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
    
    checkBatchSend() {
        const now = Date.now();
        const timeSinceLastBatch = now - this.lastBatchSent;
        
        // Send batch every 15 seconds or when buffer is large
        if (timeSinceLastBatch >= this.batchSize || 
            this.behavioralBuffer.keystrokes.length > 30 ||
            this.behavioralBuffer.mouse_events.length > 50) {
            
            this.sendBehavioralData();
        }
    }
    
    sendBehavioralData() {
        if (!this.socket || !this.socket.connected) {
            console.warn('Socket not connected, buffering data');
            return;
        }
        
        if (this.behavioralBuffer.keystrokes.length === 0 && 
            this.behavioralBuffer.mouse_events.length === 0) {
            return;
        }
        
        const dataToSend = {
            keystrokes: [...this.behavioralBuffer.keystrokes],
            mouse_events: [...this.behavioralBuffer.mouse_events],
            timestamp: Date.now(),
            session_info: {
                session_duration: Date.now() - this.stats.session_start,
                total_keystrokes: this.stats.total_keystrokes,
                total_mouse_events: this.stats.total_mouse_events
            }
        };
        
        console.log(`Sending behavioral data: ${dataToSend.keystrokes.length} keystrokes, ${dataToSend.mouse_events.length} mouse events`);
        
        // Send data to server
        this.socket.emit('behavioral_data', dataToSend);
        
        // Clear buffers
        this.behavioralBuffer.keystrokes = [];
        this.behavioralBuffer.mouse_events = [];
        this.lastBatchSent = Date.now();
        
        // Update dashboard indicator
        this.showDataSentIndicator();
    }
    
    handleAnalysisResult(data) {
        console.log('Analysis result received:', data);
        
        // Update anomaly statistics
        if (data.anomaly && data.anomaly.is_anomaly) {
            this.stats.anomalies_detected++;
            this.updateAnomalyChart(data.anomaly);
            
            // Show visual indicator for anomalies
            this.showAnomalyIndicator(data.anomaly.severity);
        }
        
        // Update real-time monitoring display
        this.updateMonitoringDisplay(data);
        
        // Update charts with new data
        this.updateChartsWithData(data);
    }
    
    handleChallengeRequired(data) {
        console.log('Challenge required:', data);
        
        // Pause monitoring during challenge
        this.pauseMonitoring();
        
        // Show challenge modal
        this.showChallengeModal(data);
    }
    
    showChallengeModal(challengeData) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay challenge-modal';
        modal.innerHTML = `
            <div class="modal">
                <div class="modal-header">
                    <h3>Security Verification Required</h3>
                </div>
                <div class="modal-body">
                    <div class="challenge-message">
                        ${challengeData.message}
                    </div>
                    <div class="challenge-reason">
                        ${challengeData.type === 'verification' ? 
                          'Unusual behavior detected. Please verify your identity.' :
                          'Your behavioral patterns have changed. Please recalibrate.'}
                    </div>
                    <div class="challenge-actions">
                        <button class="btn btn-primary" onclick="this.redirectToChallenge('${challengeData.type}')">
                            Start Verification
                        </button>
                        <button class="btn btn-outline" onclick="this.closeModal()">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Add methods to modal
        modal.redirectToChallenge = (type) => {
            window.location.href = `/challenge?type=${type}`;
        };
        
        modal.closeModal = () => {
            modal.classList.remove('active');
            setTimeout(() => {
                document.body.removeChild(modal);
                this.resumeMonitoring();
            }, 300);
        };
        
        document.body.appendChild(modal);
        setTimeout(() => modal.classList.add('active'), 10);
        
        // Auto-redirect if high severity
        if (challengeData.type === 'verification') {
            setTimeout(() => {
                window.location.href = '/challenge?type=verification';
            }, 5000);
        }
    }
    
    showSecurityAlert(alertData) {
        this.showNotification(
            `Security Alert: ${alertData.message}`,
            alertData.level || 'warning'
        );
        
        // Update security status indicator
        this.updateSecurityStatus(alertData.level);
    }
    
    handleForceLogout(data) {
        this.showNotification(
            'Session terminated for security reasons',
            'error'
        );
        
        setTimeout(() => {
            this.logout();
        }, 2000);
    }
    
    updateTrainingStatus(data) {
        const statusElement = document.getElementById('trainingStatus');
        if (statusElement && data.status !== 'completed') {
            statusElement.textContent = `Training: ${data.progress}%`;
            statusElement.className = 'status-indicator status-warning';
        } else if (statusElement && data.status === 'completed') {
            statusElement.textContent = 'Models Ready';
            statusElement.className = 'status-indicator status-success';
        }
    }
    
    initializeCharts() {
        this.initTypingSpeedChart();
        this.initAnomalyChart();
        this.initFeatureChart();
        this.initStatusPieChart();
    }
    
    initTypingSpeedChart() {
        const ctx = document.getElementById('typingSpeedChart');
        if (!ctx) return;
        
        this.charts.typingSpeed = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Typing Speed (CPM)',
                    data: [],
                    borderColor: 'rgb(37, 99, 235)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Characters per Minute'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                },
                animation: {
                    duration: 750
                }
            }
        });
    }
    
    initAnomalyChart() {
        const ctx = document.getElementById('anomalyChart');
        if (!ctx) return;
        
        this.charts.anomaly = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Normal', 'Low Risk', 'Medium Risk', 'High Risk'],
                datasets: [{
                    data: [95, 3, 1.5, 0.5],
                    backgroundColor: [
                        '#10b981',
                        '#f59e0b',
                        '#ef4444',
                        '#7c2d12'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    initFeatureChart() {
        const ctx = document.getElementById('featureChart');
        if (!ctx) return;
        
        this.charts.feature = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [
                    'Typing Speed', 'Key Hold Time', 'Flight Time', 
                    'Mouse Speed', 'Click Pattern', 'Scroll Behavior'
                ],
                datasets: [{
                    label: 'Deviation Score',
                    data: [0.2, 0.1, 0.3, 0.15, 0.25, 0.18],
                    backgroundColor: 'rgba(37, 99, 235, 0.8)',
                    borderColor: 'rgb(37, 99, 235)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Deviation from Baseline'
                        }
                    }
                }
            }
        });
    }
    
    initStatusPieChart() {
        const ctx = document.getElementById('statusChart');
        if (!ctx) return;
        
        this.charts.status = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Active Monitoring', 'Processing', 'Idle'],
                datasets: [{
                    data: [85, 10, 5],
                    backgroundColor: [
                        '#10b981',
                        '#3b82f6',
                        '#6b7280'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    updateChartsWithData(data) {
        // Update typing speed chart
        if (this.charts.typingSpeed && data.typing_speed) {
            const chart = this.charts.typingSpeed;
            const now = new Date().toLocaleTimeString();
            
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(data.typing_speed);
            
            // Keep only last 20 data points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update('none');
        }
        
        // Update feature chart if deviation data is available
        if (this.charts.feature && data.feature_deviations) {
            this.updateFeatureDeviations(data.feature_deviations);
        }
    }
    
    updateFeatureDeviations(deviations) {
        if (!this.charts.feature) return;
        
        const chart = this.charts.feature;
        const newData = [
            deviations.typing_speed || 0,
            deviations.key_hold_time || 0,
            deviations.flight_time || 0,
            deviations.mouse_speed || 0,
            deviations.click_pattern || 0,
            deviations.scroll_behavior || 0
        ];
        
        chart.data.datasets[0].data = newData;
        chart.update('none');
    }
    
    updateAnomalyChart(anomalyData) {
        if (!this.charts.anomaly) return;
        
        // Update anomaly distribution based on recent data
        const chart = this.charts.anomaly;
        
        if (anomalyData.severity === 'high') {
            chart.data.datasets[0].data[3] += 0.1;
            chart.data.datasets[0].data[0] -= 0.1;
        } else if (anomalyData.severity === 'medium') {
            chart.data.datasets[0].data[2] += 0.1;
            chart.data.datasets[0].data[0] -= 0.1;
        } else if (anomalyData.severity === 'low') {
            chart.data.datasets[0].data[1] += 0.1;
            chart.data.datasets[0].data[0] -= 0.1;
        }
        
        chart.update('none');
    }
    
    async loadUserStats() {
        try {
            const response = await fetch('/api/user-stats', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.updateDashboardStats(result.stats);
            } else {
                console.error('Failed to load user stats:', result.message);
            }
            
        } catch (error) {
            console.error('Error loading user stats:', error);
        }
    }
    
    updateDashboardStats(stats) {
        console.log('Updating dashboard stats:', stats);
        
        // Update stat cards
        const elements = {
            'sessionsCount': stats.sessions_last_7_days || 0,
            'anomaliesCount': stats.total_anomalies || 0,
            'uptime': this.formatUptime(Date.now() - this.stats.session_start),
            'accuracy': '99.2%' // This would come from server calculations
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
        
        // Update typing speed trend chart
        if (stats.typing_speed_trend && this.charts.typingSpeed) {
            const chart = this.charts.typingSpeed;
            chart.data.labels = stats.typing_speed_trend.timestamps.map(ts => 
                new Date(ts).toLocaleTimeString()
            );
            chart.data.datasets[0].data = stats.typing_speed_trend.speeds;
            chart.update();
        }
    }
    
    setupDashboardUpdates() {
        // Update real-time stats every second
        setInterval(() => {
            this.updateRealtimeStats();
        }, 1000);
        
        // Refresh user stats every 30 seconds
        setInterval(() => {
            this.loadUserStats();
        }, 30000);
        
        // Send heartbeat every 10 seconds
        setInterval(() => {
            this.sendHeartbeat();
        }, 10000);
        
        // Force send data every 30 seconds even if buffer is small
        setInterval(() => {
            if (this.isMonitoring && this.socket && this.socket.connected) {
                this.sendBehavioralData();
            }
        }, 30000);
    }
    
    updateRealtimeStats() {
        // Update session duration
        const sessionDuration = Date.now() - this.stats.session_start;
        const uptimeElement = document.getElementById('sessionUptime');
        if (uptimeElement) {
            uptimeElement.textContent = this.formatUptime(sessionDuration);
        }
        
        // Update event counters
        const keystrokeElement = document.getElementById('keystrokeCount');
        if (keystrokeElement) {
            keystrokeElement.textContent = this.stats.total_keystrokes.toLocaleString();
        }
        
        const mouseElement = document.getElementById('mouseEventCount');
        if (mouseElement) {
            mouseElement.textContent = this.stats.total_mouse_events.toLocaleString();
        }
        
        // Update monitoring status
        this.updateMonitoringStatus();
    }
    
    updateMonitoringDisplay(data) {
        const monitoringIndicator = document.getElementById('monitoringIndicator');
        if (monitoringIndicator) {
            if (data.anomaly && data.anomaly.is_anomaly) {
                monitoringIndicator.className = 'status-indicator status-warning';
                monitoringIndicator.innerHTML = `
                    <span class="status-dot"></span>
                    Anomaly Detected
                `;
            } else {
                monitoringIndicator.className = 'status-indicator status-online';
                monitoringIndicator.innerHTML = `
                    <span class="status-dot"></span>
                    Normal Behavior
                `;
            }
        }
    }
    
    updateMonitoringStatus() {
        const statusElement = document.getElementById('monitoringStatus');
        if (statusElement) {
            if (this.isMonitoring) {
                statusElement.className = 'status-indicator status-online';
                statusElement.innerHTML = `
                    <span class="status-dot"></span>
                    Active Monitoring
                `;
            } else {
                statusElement.className = 'status-indicator status-error';
                statusElement.innerHTML = `
                    <span class="status-dot"></span>
                    Monitoring Paused
                `;
            }
        }
    }
    
    updateConnectionStatus(connected) {
        const connectionElement = document.getElementById('connectionStatus');
        if (connectionElement) {
            if (connected) {
                connectionElement.className = 'status-indicator status-online';
                connectionElement.innerHTML = `
                    <span class="status-dot"></span>
                    Connected
                `;
            } else {
                connectionElement.className = 'status-indicator status-error';
                connectionElement.innerHTML = `
                    <span class="status-dot"></span>
                    Disconnected
                `;
            }
        }
    }
    
    updateSecurityStatus(level) {
        const securityElement = document.getElementById('securityStatus');
        if (securityElement) {
            const statusMap = {
                'low': { class: 'status-online', text: 'Secure' },
                'medium': { class: 'status-warning', text: 'Alert' },
                'high': { class: 'status-error', text: 'Risk Detected' }
            };
            
            const status = statusMap[level] || statusMap['low'];
            securityElement.className = `status-indicator ${status.class}`;
            securityElement.innerHTML = `
                <span class="status-dot"></span>
                ${status.text}
            `;
        }
    }
    
    showDataSentIndicator() {
        const indicator = document.getElementById('dataSentIndicator');
        if (indicator) {
            indicator.style.opacity = '1';
            indicator.classList.add('show');
            setTimeout(() => {
                indicator.style.opacity = '0';
                indicator.classList.remove('show');
            }, 1500);
        }
    }
    
    showAnomalyIndicator(severity) {
        const anomalyAlert = document.createElement('div');
        anomalyAlert.className = `anomaly-indicator severity-${severity}`;
        anomalyAlert.innerHTML = `
            <div class="anomaly-icon">⚠</div>
            <div class="anomaly-text">${severity.toUpperCase()} ANOMALY</div>
        `;
        
        document.body.appendChild(anomalyAlert);
        
        setTimeout(() => {
            anomalyAlert.classList.add('show');
        }, 10);
        
        setTimeout(() => {
            anomalyAlert.classList.remove('show');
            setTimeout(() => {
                if (anomalyAlert.parentNode) {
                    document.body.removeChild(anomalyAlert);
                }
            }, 300);
        }, 3000);
    }
    
    toggleMonitoring(enabled) {
        this.isMonitoring = enabled;
        
        const toggle = document.getElementById('monitoringToggle');
        if (toggle) {
            toggle.checked = enabled;
        }
        
        console.log(`Monitoring ${enabled ? 'enabled' : 'disabled'}`);
        
        if (enabled) {
            this.resumeMonitoring();
        } else {
            this.pauseMonitoring();
        }
        
        this.updateMonitoringStatus();
    }
    
    resumeMonitoring() {
        if (!this.isMonitoring) return;
        
        console.log('Resuming behavioral monitoring');
        
        // Resume data collection
        this.stats.current_status = 'monitoring';
        
        // Send any buffered data
        if (this.behavioralBuffer.keystrokes.length > 0 || 
            this.behavioralBuffer.mouse_events.length > 0) {
            this.sendBehavioralData();
        }
    }
    
    pauseMonitoring() {
        console.log('Pausing behavioral monitoring');
        
        this.stats.current_status = 'paused';
        
        // Send current buffer before pausing
        this.sendBehavioralData();
    }
    
    sendHeartbeat() {
        if (this.socket && this.socket.connected) {
            this.socket.emit('heartbeat', {
                timestamp: Date.now(),
                status: this.stats.current_status,
                session_duration: Date.now() - this.stats.session_start
            });
        }
    }
    
    formatUptime(milliseconds) {
        const seconds = Math.floor(milliseconds / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes % 60}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">${this.getNotificationIcon(type)}</div>
                <div class="notification-message">
                    <div class="notification-title">${this.getNotificationTitle(type)}</div>
                    <div class="notification-text">${message}</div>
                </div>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.classList.remove('show');
                setTimeout(() => {
                    if (notification.parentNode) {
                        document.body.removeChild(notification);
                    }
                }, 300);
            }
        }, 5000);
    }
    
    getNotificationIcon(type) {
        const icons = {
            success: '✓',
            error: '✗',
            warning: '⚠',
            info: 'ℹ'
        };
        return icons[type] || icons.info;
    }
    
    getNotificationTitle(type) {
        const titles = {
            success: 'Success',
            error: 'Error',
            warning: 'Warning',
            info: 'Information'
        };
        return titles[type] || titles.info;
    }
    
    showSettings() {
        console.log('Settings modal would open here');
    }
    
    async exportData() {
        try {
            const response = await fetch('/api/export-user-data', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
                }
            });
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `behavioral_data_${new Date().getTime()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            this.showNotification('Data exported successfully', 'success');
            
        } catch (error) {
            console.error('Error exporting data:', error);
            this.showNotification('Failed to export data', 'error');
        }
    }
    
    async logout() {
        try {
            console.log('Logging out...');
            
            // Stop monitoring
            this.toggleMonitoring(false);
            
            // Disconnect socket
            if (this.socket) {
                this.socket.disconnect();
            }
            
            // Send logout request
            const response = await fetch('/api/logout', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
                }
            });
            
            // Clear local storage
            localStorage.removeItem('auth_token');
            
            // Redirect to login
            window.location.href = '/login';
            
        } catch (error) {
            console.error('Logout error:', error);
            // Force redirect even if logout fails
            localStorage.removeItem('auth_token');
            window.location.href = '/login';
        }
    }
}

// Additional dashboard-specific styles
const dashboardStyles = `
<style>
.anomaly-indicator {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) scale(0);
    background: rgba(0, 0, 0, 0.9);
    color: white;
    padding: 2rem;
    border-radius: var(--radius-xl);
    display: flex;
    align-items: center;
    gap: 1rem;
    z-index: var(--z-notification);
    transition: all var(--transition-normal);
    pointer-events: none;
}

.anomaly-indicator.show {
    transform: translate(-50%, -50%) scale(1);
}

.anomaly-indicator.severity-high {
    border: 3px solid var(--error-color);
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.5);
}

.anomaly-indicator.severity-medium {
    border: 3px solid var(--warning-color);
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.5);
}

.anomaly-indicator.severity-low {
    border: 3px solid var(--info-color);
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
}

.anomaly-icon {
    font-size: 2rem;
    animation: pulse 1s infinite;
}

.anomaly-text {
    font-weight: 700;
    font-size: 1.25rem;
    letter-spacing: 0.1em;
}

.modal-overlay.active {
    opacity: 1;
    visibility: visible;
}

.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(4px);
    z-index: var(--z-modal);
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    visibility: hidden;
    transition: all var(--transition-normal);
}
</style>
`;

// Add styles to document head
document.head.insertAdjacentHTML('beforeend', dashboardStyles);

// Initialize dashboard manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Dashboard Manager...');
    new DashboardManager();
});

// Export for testing
window.DashboardManager = DashboardManager;