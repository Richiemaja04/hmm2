#!/usr/bin/env python3
"""
Configuration Module for Behavioral Biometrics Authentication Agent

This module contains all configuration settings for the application,
including security parameters, ML model configurations, and system thresholds.
"""

import os
from datetime import timedelta

class Config:
    """Main configuration class containing all application settings."""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production-2024'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database Configuration
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///behavioral_auth.db'
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Security Configuration
    JWT_SECRET_KEY = SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    SESSION_TIMEOUT = timedelta(hours=8)
    MAX_LOGIN_ATTEMPTS = 5
    ACCOUNT_LOCKOUT_DURATION = timedelta(minutes=30)
    
    # Behavioral Analysis Configuration
    MONITORING_WINDOW_SIZE = 30  # seconds
    MIN_CALIBRATION_SAMPLES = 50
    MIN_CHALLENGE_SAMPLES = 20
    FEATURE_VECTOR_SIZE = 40
    
    # Anomaly Detection Thresholds
    ANOMALY_THRESHOLDS = {
        'low': 0.3,      # Normal variations
        'medium': 0.6,   # Suspicious behavior
        'high': 0.8,     # Likely intrusion
        'critical': 0.9  # Definite threat
    }
    
    # Model Configuration
    MODEL_CONFIGS = {
        'gru': {
            'sequence_length': 20,
            'hidden_units': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        },
        'autoencoder': {
            'encoding_dim': 20,
            'hidden_layers': [32, 16],
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'reconstruction_threshold': 0.1
        },
        'isolation_forest': {
            'n_estimators': 100,
            'contamination': 0.1,
            'random_state': 42
        },
        'one_class_svm': {
            'kernel': 'rbf',
            'gamma': 'scale',
            'nu': 0.1
        },
        'passive_aggressive': {
            'C': 1.0,
            'random_state': 42,
            'max_iter': 1000
        },
        'incremental_knn': {
            'n_neighbors': 5,
            'contamination': 0.1
        }
    }
    
    # Feature Extraction Configuration
    FEATURE_CONFIG = {
        'keystroke_features': [
            'key_hold_time_mean', 'key_hold_time_std',
            'flight_time_mean', 'flight_time_std',
            'typing_speed_mean', 'typing_speed_std',
            'backspace_frequency', 'delete_frequency',
            'shift_usage', 'enter_usage', 'arrow_usage',
            'digraph_latency_mean', 'trigraph_latency_mean',
            'error_correction_rate', 'capitalization_method',
            'punctuation_frequency', 'word_count',
            'special_char_frequency', 'session_uptime',
            'numeric_keypad_usage', 'typing_rhythm_score'
        ],
        'mouse_features': [
            'mouse_speed_mean', 'mouse_speed_peak',
            'mouse_acceleration_mean', 'movement_curvature',
            'movement_jitter', 'path_straightness',
            'pause_count', 'pause_duration_mean',
            'click_rate', 'double_click_frequency',
            'right_click_frequency', 'click_duration_mean',
            'scroll_speed_mean', 'scroll_direction_ratio',
            'drag_drop_count', 'drag_distance_mean',
            'hover_duration_mean', 'idle_time',
            'movement_angle_variance', 'wheel_click_frequency'
        ]
    }
    
    # Drift Detection Configuration
    DRIFT_CONFIG = {
        'window_size': 100,  # Number of samples for drift analysis
        'sensitivity': 0.05,  # Statistical significance level
        'adaptation_threshold': 0.1,  # When to trigger adaptation
        'methods': ['ks_test', 'wasserstein_distance', 'population_stability_index']
    }
    
    # Real-time Processing Configuration
    REALTIME_CONFIG = {
        'data_batch_size': 10,  # Number of events to batch before processing
        'processing_interval': 5,  # seconds between batch processing
        'max_queue_size': 1000,  # Maximum events in processing queue
        'websocket_timeout': 60,  # seconds
        'heartbeat_interval': 30  # seconds
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_file': 'behavioral_auth.log',
        'max_bytes': 10485760,  # 10MB
        'backup_count': 5
    }
    
    # Challenge Configuration
    CHALLENGE_CONFIG = {
        'verification_phrases': [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How razorback-jumping frogs can level six piqued gymnasts.",
            "Jinxed wizards pluck ivy from the big quilt.",
            "The five boxing wizards jump quickly."
        ],
        'adaptation_paragraphs': [
            """In the digital age, cybersecurity has become paramount for protecting 
            sensitive information and maintaining user privacy. Behavioral biometrics 
            represents a cutting-edge approach to continuous authentication, analyzing 
            unique patterns in how users interact with their devices.""",
            
            """Machine learning algorithms can detect subtle changes in typing patterns, 
            mouse movements, and other behavioral characteristics that are difficult 
            for attackers to replicate. This technology provides an additional layer 
            of security beyond traditional password-based systems.""",
            
            """The integration of artificial intelligence in security systems enables 
            real-time threat detection and adaptive responses to emerging cyber threats. 
            By continuously monitoring user behavior, organizations can identify 
            potential security breaches before they cause significant damage."""
        ]
    }
    
    # Performance Monitoring Configuration
    PERFORMANCE_CONFIG = {
        'max_response_time': 200,  # milliseconds
        'memory_threshold': 512,   # MB
        'cpu_threshold': 80,       # percentage
        'disk_usage_threshold': 90  # percentage
    }
    
    # Data Retention Configuration
    DATA_RETENTION = {
        'behavioral_data_days': 30,
        'session_logs_days': 7,
        'anomaly_reports_days': 90,
        'calibration_data_days': 365
    }

class DevelopmentConfig(Config):
    """Development-specific configuration."""
    DEBUG = True
    TESTING = False
    
class ProductionConfig(Config):
    """Production-specific configuration."""
    DEBUG = False
    TESTING = False
    # Additional production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class TestingConfig(Config):
    """Testing-specific configuration."""
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Configuration dictionary for easy access
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}