class BehavioralCollector {
    constructor(socket) {
        this.socket = socket;
        this.isMonitoring = false;
        this.buffer = {
            keystrokes: [],
            mouseEvents: []
        };
        this.batchInterval = 5000; // Send data every 5 seconds
        this.batchTimer = null;

        this.eventListeners = {
            keydown: this.recordKeystroke.bind(this),
            mousemove: this.throttle(this.recordMouseEvent.bind(this), 50), // Throttle mousemove
            mousedown: this.recordMouseEvent.bind(this),
            mouseup: this.recordMouseEvent.bind(this),
            click: this.recordMouseEvent.bind(this),
            wheel: this.recordMouseEvent.bind(this),
        };
    }

    start() {
        if (this.isMonitoring) return;
        this.isMonitoring = true;
        Object.keys(this.eventListeners).forEach(event => {
            document.addEventListener(event, this.eventListeners[event]);
        });
        this.batchTimer = setInterval(() => this.sendBatch(), this.batchInterval);
        console.log("Behavioral monitoring started.");
    }

    stop() {
        if (!this.isMonitoring) return;
        this.isMonitoring = false;
        Object.keys(this.eventListeners).forEach(event => {
            document.removeEventListener(event, this.eventListeners[event]);
        });
        clearInterval(this.batchTimer);
        this.sendBatch(); // Send any remaining data
        console.log("Behavioral monitoring stopped.");
    }

    recordKeystroke(e) {
        this.buffer.keystrokes.push({
            type: e.type,
            key: e.key,
            code: e.code,
            timestamp: Date.now()
        });
    }

    recordMouseEvent(e) {
        this.buffer.mouseEvents.push({
            type: e.type,
            x: e.clientX,
            y: e.clientY,
            timestamp: Date.now()
        });
    }

    sendBatch() {
        if (this.buffer.keystrokes.length === 0 && this.buffer.mouseEvents.length === 0) {
            return;
        }

        if (this.socket && this.socket.connected) {
            this.socket.emit('behavioral_data', {
                keystrokes: this.buffer.keystrokes,
                mouse_events: this.buffer.mouseEvents
            });

            // Clear buffer after sending
            this.buffer.keystrokes = [];
            this.buffer.mouseEvents = [];
        }
    }

    // Utility to throttle high-frequency events like mousemove
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}