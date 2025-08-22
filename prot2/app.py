#!/usr/bin/env python3
"""
Behavioral Biometrics Continuous Authentication Agent
Main Flask Application - FIXED VERSION

This module serves as the core application entry point for the behavioral
biometrics authentication system. It handles routing, WebSocket communication,
and orchestrates the entire authentication workflow.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, disconnect
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import threading
import time
import logging
from datetime import datetime, timedelta
import json
import os

# Local imports
from config import Config
from database.db_manager import DatabaseManager
from models.behavioral_models import BehavioralModels
from utils.feature_extractor import FeatureExtractor
from utils.anomaly_finder import AnomalyFinder
from utils.drift_detector import DriftDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config.from_object(Config)

# Ensure secret key is set
if not app.config.get('SECRET_KEY'):
    app.config['SECRET_KEY'] = 'dev-key-change-in-production-2024'
    logger.warning("Using default SECRET_KEY. Change this in production!")

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize components with error handling
try:
    db_manager = DatabaseManager(app.config['DATABASE_URL'])
    # Initialize database immediately
    db_manager.init_database()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise

try:
    behavioral_models = BehavioralModels()
    feature_extractor = FeatureExtractor()
    anomaly_finder = AnomalyFinder()
    drift_detector = DriftDetector()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise

# Global state tracking
user_sessions = {}
training_status = {}

@app.route('/')
def index():
    """Redirect to login page."""
    return redirect(url_for('login'))

@app.route('/login')
def login():
    """Render login page."""
    return render_template('login.html')

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'database': 'connected',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/login', methods=['POST'])
def api_login():
    """Handle user login authentication."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
            
        username = data.get('username')
        password = data.get('password')
        behavioral_data = data.get('behavioral_data', {})
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'})
        
        logger.info(f"Login attempt for username: {username}")
        
        # Check user credentials
        user = db_manager.get_user(username)
        
        if not user:
            # Create new user
            logger.info(f"Creating new user: {username}")
            hashed_password = generate_password_hash(password)
            user_id = db_manager.create_user(username, hashed_password)
            
            # Store initial behavioral data if provided
            if behavioral_data:
                try:
                    features = feature_extractor.extract_features(behavioral_data)
                    db_manager.store_behavioral_data(
                        user_id, 
                        behavioral_data, 
                        features, 
                        {'is_anomaly': False, 'score': 0.0, 'session_id': 'login'}
                    )
                    logger.info(f"Stored initial behavioral data for user {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to store initial behavioral data: {e}")
            
            # Generate JWT token
            token = jwt.encode({
                'user_id': user_id,
                'username': username,
                'exp': datetime.utcnow() + timedelta(hours=24)
            }, app.config['SECRET_KEY'], algorithm='HS256')
            
            session.permanent = True
            session['token'] = token
            session['user_id'] = user_id
            session['username'] = username
            
            logger.info(f"New user created successfully: {username} (ID: {user_id})")
            
            return jsonify({
                'success': True, 
                'message': 'Account created successfully',
                'redirect': '/calibration',
                'new_user': True,
                'token': token
            })
        
        else:
            # Verify existing user
            if check_password_hash(user['password_hash'], password):
                logger.info(f"Successful login for user: {username}")
                
                # Store login behavioral data
                if behavioral_data:
                    try:
                        features = feature_extractor.extract_features(behavioral_data)
                        db_manager.store_behavioral_data(
                            user['id'], 
                            behavioral_data, 
                            features, 
                            {'is_anomaly': False, 'score': 0.0, 'session_id': 'login'}
                        )
                        logger.info(f"Stored login behavioral data for user {user['id']}")
                    except Exception as e:
                        logger.warning(f"Failed to store login behavioral data: {e}")
                
                # Update last login
                db_manager.update_last_login(user['id'])
                
                token = jwt.encode({
                    'user_id': user['id'],
                    'username': username,
                    'exp': datetime.utcnow() + timedelta(hours=24)
                }, app.config['SECRET_KEY'], algorithm='HS256')
                
                session.permanent = True
                session['token'] = token
                session['user_id'] = user['id']
                session['username'] = username
                
                return jsonify({
                    'success': True,
                    'message': 'Login successful',
                    'redirect': '/dashboard',
                    'new_user': False,
                    'token': token
                })
            else:
                logger.warning(f"Failed login attempt for user: {username}")
                return jsonify({'success': False, 'message': 'Invalid credentials'})
                
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'})

@app.route('/calibration')
def calibration():
    """Render calibration page for new users."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('calib.html')

@app.route('/dashboard')
def dashboard():
    """Render main dashboard."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/challenge')
def challenge():
    """Render challenge page for anomaly verification."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('challenge.html')

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Handle user logout."""
    user_id = session.get('user_id')
    if user_id and user_id in user_sessions:
        del user_sessions[user_id]
    
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/calibration', methods=['POST'])
def api_calibration():
    """Process calibration data and train initial models."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'message': 'Unauthorized'})
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
            
        behavioral_data = data.get('behavioral_data', {})
        
        # Extract events from behavioral data
        keystrokes = behavioral_data.get('keystrokes', [])
        mouse_events = behavioral_data.get('mouse_events', [])
        
        total_events = len(keystrokes) + len(mouse_events)
        
        if total_events < 50:  # Minimum samples for training
            return jsonify({
                'success': False, 
                'message': f'Insufficient calibration data. Got {total_events} events, need at least 50.'
            })
        
        logger.info(f"Processing calibration data for user {user_id}: {total_events} events")
        
        # Extract features from calibration data
        features = feature_extractor.extract_features(behavioral_data)
        
        # Store calibration data
        db_manager.store_calibration_data(user_id, behavioral_data, [features])
        logger.info(f"Stored calibration data for user {user_id}")
        
        # Start asynchronous model training
        training_status[user_id] = {'status': 'training', 'progress': 0}
        
        def train_models():
            try:
                logger.info(f"Starting model training for user {user_id}")
                training_status[user_id] = {'status': 'training', 'progress': 25}
                
                # Create multiple feature samples for training
                feature_samples = [features] * 10  # Replicate for initial training
                
                # Train ensemble models
                behavioral_models.train_user_models(user_id, feature_samples)
                training_status[user_id] = {'status': 'training', 'progress': 75}
                
                # Update database
                db_manager.update_model_training_status(user_id, True)
                training_status[user_id] = {'status': 'completed', 'progress': 100}
                
                logger.info(f"Model training completed for user {user_id}")
            except Exception as e:
                logger.error(f"Model training failed for user {user_id}: {e}")
                training_status[user_id] = {'status': 'failed', 'progress': 0, 'error': str(e)}
        
        threading.Thread(target=train_models, daemon=True).start()
        
        return jsonify({
            'success': True,
            'message': 'Calibration completed. Training models...',
            'training_started': True,
            'events_processed': total_events
        })
        
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return jsonify({'success': False, 'message': f'Calibration failed: {str(e)}'})

@app.route('/api/training-status')
def api_training_status():
    """Get model training status."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    status = training_status.get(user_id, {'status': 'unknown', 'progress': 0})
    return jsonify({'success': True, 'status': status})

@app.route('/api/user-stats')
def api_user_stats():
    """Get user behavioral statistics for dashboard."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized'})
    
    try:
        stats = db_manager.get_user_stats(user_id)
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return jsonify({'success': False, 'message': 'Failed to get user stats'})

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    # Get user info from session or token
    user_id = session.get('user_id')
    
    if not user_id:
        # Try to get from token in auth header
        try:
            auth_data = request.args.get('token') or request.headers.get('Authorization')
            if auth_data:
                if auth_data.startswith('Bearer '):
                    auth_data = auth_data[7:]
                payload = jwt.decode(auth_data, app.config['SECRET_KEY'], algorithms=['HS256'])
                user_id = payload.get('user_id')
                session['user_id'] = user_id
                session['username'] = payload.get('username')
        except:
            pass
    
    if user_id:
        user_sessions[user_id] = {
            'session_id': request.sid,
            'connected_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'anomaly_count': 0,
            'authenticated': True
        }
        logger.info(f"User {user_id} connected via WebSocket")
        emit('connection_status', {'status': 'connected', 'user_id': user_id})
    else:
        logger.warning("WebSocket connection without valid user session")
        disconnect()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    user_id = session.get('user_id')
    if user_id and user_id in user_sessions:
        logger.info(f"User {user_id} disconnected")
        del user_sessions[user_id]

@socketio.on('behavioral_data')
def handle_behavioral_data(data):
    """Process incoming behavioral data from frontend."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            logger.warning("Behavioral data received without valid user session")
            disconnect()
            return
        
        logger.debug(f"Received behavioral data from user {user_id}: {len(data.get('keystrokes', []))} keystrokes, {len(data.get('mouse_events', []))} mouse events")
        
        # Check if models are trained
        status = training_status.get(user_id, {'status': 'unknown'})
        if status['status'] != 'completed':
            emit('training_status', status)
            return
        
        # Update session activity
        if user_id in user_sessions:
            user_sessions[user_id]['last_activity'] = datetime.utcnow()
        
        # Extract features
        features = feature_extractor.extract_features(data)
        logger.debug(f"Extracted {len(features)} features for user {user_id}")
        
        # Run anomaly detection
        anomaly_result = anomaly_finder.detect_anomaly(user_id, features)
        
        # Store behavioral data
        db_manager.store_behavioral_data(user_id, data, features, anomaly_result)
        logger.debug(f"Stored behavioral data for user {user_id}")
        
        # Check for drift
        drift_result = drift_detector.check_drift(user_id, features)
        
        # Handle anomalies
        if anomaly_result['is_anomaly']:
            handle_anomaly(user_id, anomaly_result, drift_result)
        
        # Send real-time updates to frontend
        emit('analysis_result', {
            'anomaly': anomaly_result,
            'drift': drift_result,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing behavioral data: {e}")
        emit('error', {'message': 'Processing error'})

def handle_anomaly(user_id, anomaly_result, drift_result):
    """Handle detected anomalies based on severity."""
    try:
        severity = anomaly_result.get('severity', 'low')
        
        if user_id in user_sessions:
            user_sessions[user_id]['anomaly_count'] += 1
        
        logger.info(f"Anomaly detected for user {user_id}: severity={severity}")
        
        if severity == 'high' or (user_sessions.get(user_id, {}).get('anomaly_count', 0) > 3):
            # Trigger high-risk challenge
            emit('challenge_required', {
                'type': 'verification',
                'message': 'Security verification required'
            })
            
        elif drift_result.get('drift_detected', False):
            # Trigger drift adaptation challenge
            emit('challenge_required', {
                'type': 'adaptation', 
                'message': 'Behavioral calibration update needed'
            })
            
        elif severity == 'medium':
            # Send warning
            emit('security_alert', {
                'level': 'warning',
                'message': 'Unusual behavior detected'
            })
            
    except Exception as e:
        logger.error(f"Error handling anomaly: {e}")

@socketio.on('challenge_response')
def handle_challenge_response(data):
    """Process challenge response data."""
    try:
        user_id = session.get('user_id')
        if not user_id:
            disconnect()
            return
        
        challenge_type = data.get('type')
        behavioral_data = data.get('behavioral_data', {})
        
        logger.info(f"Processing challenge response for user {user_id}: type={challenge_type}")
        
        if challenge_type == 'verification':
            # Verify user identity with challenge data
            features = feature_extractor.extract_features(behavioral_data)
            verification_result = anomaly_finder.verify_identity(user_id, features)
            
            if verification_result['verified']:
                # Reset anomaly count
                if user_id in user_sessions:
                    user_sessions[user_id]['anomaly_count'] = 0
                
                emit('challenge_result', {
                    'success': True,
                    'message': 'Verification successful'
                })
                logger.info(f"Challenge verification successful for user {user_id}")
            else:
                # Force logout
                emit('force_logout', {
                    'message': 'Verification failed. Session terminated.'
                })
                logger.warning(f"Challenge verification failed for user {user_id}")
                
        elif challenge_type == 'adaptation':
            # Use data for model retraining
            features = feature_extractor.extract_features(behavioral_data)
            
            # Retrain models with new data
            behavioral_models.update_user_models(user_id, [features])
            
            emit('challenge_result', {
                'success': True,
                'message': 'Behavioral profile updated'
            })
            logger.info(f"Behavioral profile updated for user {user_id}")
            
    except Exception as e:
        logger.error(f"Error processing challenge response: {e}")
        emit('error', {'message': 'Challenge processing error'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Behavioral Biometrics Authentication Agent")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)