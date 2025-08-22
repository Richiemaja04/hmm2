#!/usr/bin/env python3
"""
Database Manager for Behavioral Biometrics Authentication Agent - FIXED VERSION

This module handles all database operations including user management,
behavioral data storage, model persistence, and analytics queries.
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager
import threading
from typing import Dict, List, Optional, Tuple, Any
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database operations for the behavioral authentication system."""
    
    def __init__(self, database_url: str):
        """Initialize database manager with connection URL."""
        # Fix database URL handling
        if database_url.startswith('sqlite:///'):
            # Keep the full path for absolute paths or use relative path
            if database_url.startswith('sqlite:////'):
                # Absolute path
                self.database_url = database_url[10:]  # Remove 'sqlite:///'
            else:
                # Relative path
                self.database_url = database_url[10:]  # Remove 'sqlite:///'
        elif database_url.startswith('sqlite://'):
            self.database_url = database_url[9:]  # Remove 'sqlite://'
        else:
            self.database_url = database_url
        
        # Ensure directory exists
        db_dir = os.path.dirname(self.database_url)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        self.lock = threading.Lock()
        logger.info(f"DatabaseManager initialized with database: {self.database_url}")
        
        # Test connection
        try:
            with self.get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(self.database_url, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database schema with all required tables."""
        logger.info("Initializing database schema...")
        
        with self.lock:
            with self.get_connection() as conn:
                try:
                    # Users table
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT UNIQUE NOT NULL,
                            password_hash TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_login TIMESTAMP,
                            is_active BOOLEAN DEFAULT TRUE,
                            failed_login_attempts INTEGER DEFAULT 0,
                            locked_until TIMESTAMP NULL
                        )
                    """)
                    
                    # User profiles table for behavioral metadata
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS user_profiles (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            calibration_completed BOOLEAN DEFAULT FALSE,
                            models_trained BOOLEAN DEFAULT FALSE,
                            last_model_update TIMESTAMP,
                            total_sessions INTEGER DEFAULT 0,
                            total_keystrokes INTEGER DEFAULT 0,
                            total_mouse_events INTEGER DEFAULT 0,
                            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Behavioral data table
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS behavioral_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            session_id TEXT NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            event_type TEXT NOT NULL,  -- 'keystroke' or 'mouse'
                            raw_data TEXT NOT NULL,    -- JSON of raw events
                            features TEXT NOT NULL,    -- JSON of extracted features
                            anomaly_score REAL,
                            is_anomaly BOOLEAN DEFAULT FALSE,
                            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Calibration data table
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS calibration_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            keystroke_data TEXT,    -- JSON of keystroke calibration
                            mouse_data TEXT,        -- JSON of mouse calibration
                            feature_vectors TEXT,   -- JSON of extracted features
                            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Model data table for storing trained models
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS model_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            model_type TEXT NOT NULL,  -- 'gru', 'autoencoder', etc.
                            model_data BLOB,           -- Serialized model
                            parameters TEXT,           -- JSON of model parameters
                            performance_metrics TEXT,  -- JSON of performance data
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            is_active BOOLEAN DEFAULT TRUE,
                            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Sessions table
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS user_sessions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            session_id TEXT UNIQUE NOT NULL,
                            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            end_time TIMESTAMP,
                            ip_address TEXT,
                            user_agent TEXT,
                            anomaly_count INTEGER DEFAULT 0,
                            status TEXT DEFAULT 'active',  -- 'active', 'terminated', 'expired'
                            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Anomaly reports table
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS anomaly_reports (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            session_id TEXT NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            anomaly_type TEXT NOT NULL,    -- 'behavioral', 'drift', 'challenge'
                            severity TEXT NOT NULL,        -- 'low', 'medium', 'high', 'critical'
                            confidence_score REAL NOT NULL,
                            feature_deviations TEXT,       -- JSON of specific deviations
                            action_taken TEXT,             -- 'alert', 'challenge', 'terminate'
                            resolved BOOLEAN DEFAULT FALSE,
                            resolution_method TEXT,
                            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Challenge responses table
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS challenge_responses (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            session_id TEXT NOT NULL,
                            challenge_type TEXT NOT NULL,  -- 'verification', 'adaptation'
                            challenge_text TEXT NOT NULL,
                            response_data TEXT NOT NULL,   -- JSON of behavioral response
                            success BOOLEAN NOT NULL,
                            confidence_score REAL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                        )
                    """)
                    
                    # System metrics table
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS system_metrics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metric_type TEXT NOT NULL,
                            metric_value REAL NOT NULL,
                            additional_data TEXT  -- JSON for extra context
                        )
                    """)
                    
                    # Create indexes for better performance
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_behavioral_data_user_id ON behavioral_data(user_id)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_behavioral_data_timestamp ON behavioral_data(timestamp)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_reports_user_id ON anomaly_reports(user_id)")
                    
                    conn.commit()
                    logger.info("Database schema initialized successfully")
                    
                    # Log table counts for verification
                    tables = ['users', 'user_profiles', 'behavioral_data', 'calibration_data', 
                             'model_data', 'user_sessions', 'anomaly_reports', 'challenge_responses']
                    for table in tables:
                        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        logger.debug(f"Table {table}: {count} records")
                        
                except Exception as e:
                    logger.error(f"Error initializing database schema: {e}")
                    raise
    
    # User Management Methods
    
    def create_user(self, username: str, password_hash: str) -> int:
        """Create a new user and return user ID."""
        with self.lock:
            with self.get_connection() as conn:
                try:
                    cursor = conn.execute(
                        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (username, password_hash)
                    )
                    user_id = cursor.lastrowid
                    
                    # Create user profile
                    conn.execute(
                        "INSERT INTO user_profiles (user_id) VALUES (?)",
                        (user_id,)
                    )
                    
                    conn.commit()
                    logger.info(f"Created new user: {username} (ID: {user_id})")
                    return user_id
                except Exception as e:
                    logger.error(f"Error creating user {username}: {e}")
                    raise
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Retrieve user by username."""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM users WHERE username = ? AND is_active = TRUE",
                    (username,)
                ).fetchone()
                
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error getting user {username}: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Retrieve user by ID."""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM users WHERE id = ? AND is_active = TRUE",
                    (user_id,)
                ).fetchone()
                
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {e}")
            return None
    
    def update_last_login(self, user_id: int):
        """Update user's last login timestamp."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (user_id,)
                )
                conn.commit()
                logger.debug(f"Updated last login for user {user_id}")
        except Exception as e:
            logger.error(f"Error updating last login for user {user_id}: {e}")
    
    def increment_failed_login(self, username: str) -> int:
        """Increment failed login attempts and return current count."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "UPDATE users SET failed_login_attempts = failed_login_attempts + 1 WHERE username = ?",
                    (username,)
                )
                
                row = conn.execute(
                    "SELECT failed_login_attempts FROM users WHERE username = ?",
                    (username,)
                ).fetchone()
                
                conn.commit()
                return row[0] if row else 0
        except Exception as e:
            logger.error(f"Error incrementing failed login for {username}: {e}")
            return 0
    
    def reset_failed_logins(self, username: str):
        """Reset failed login attempts for user."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "UPDATE users SET failed_login_attempts = 0, locked_until = NULL WHERE username = ?",
                    (username,)
                )
                conn.commit()
                logger.debug(f"Reset failed logins for {username}")
        except Exception as e:
            logger.error(f"Error resetting failed logins for {username}: {e}")
    
    # User Profile Methods
    
    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get user profile information."""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM user_profiles WHERE user_id = ?",
                    (user_id,)
                ).fetchone()
                
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error getting user profile for {user_id}: {e}")
            return None
    
    def update_calibration_status(self, user_id: int, completed: bool = True):
        """Update user's calibration completion status."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "UPDATE user_profiles SET calibration_completed = ? WHERE user_id = ?",
                    (completed, user_id)
                )
                conn.commit()
                logger.info(f"Updated calibration status for user {user_id}: {completed}")
        except Exception as e:
            logger.error(f"Error updating calibration status for user {user_id}: {e}")
    
    def update_model_training_status(self, user_id: int, trained: bool = True):
        """Update user's model training status."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """UPDATE user_profiles SET models_trained = ?, 
                       last_model_update = CURRENT_TIMESTAMP WHERE user_id = ?""",
                    (trained, user_id)
                )
                conn.commit()
                logger.info(f"Updated model training status for user {user_id}: {trained}")
        except Exception as e:
            logger.error(f"Error updating model training status for user {user_id}: {e}")
    
    # Behavioral Data Methods
    
    def store_calibration_data(self, user_id: int, behavioral_data: Dict, 
                             features: List[List[float]]):
        """Store calibration data for initial model training."""
        with self.lock:
            with self.get_connection() as conn:
                try:
                    # Separate keystroke and mouse data
                    keystroke_data = behavioral_data.get('keystrokes', [])
                    mouse_data = behavioral_data.get('mouse_events', [])
                    
                    conn.execute(
                        """INSERT INTO calibration_data 
                           (user_id, keystroke_data, mouse_data, feature_vectors) 
                           VALUES (?, ?, ?, ?)""",
                        (user_id, 
                         json.dumps(keystroke_data),
                         json.dumps(mouse_data),
                         json.dumps(features))
                    )
                    
                    # Update profile
                    self.update_calibration_status(user_id, True)
                    
                    conn.commit()
                    logger.info(f"Stored calibration data for user {user_id}: {len(keystroke_data)} keystrokes, {len(mouse_data)} mouse events")
                except Exception as e:
                    logger.error(f"Error storing calibration data for user {user_id}: {e}")
                    raise
    
    def store_behavioral_data(self, user_id: int, raw_data: Dict, 
                            features: List[float], anomaly_result: Dict):
        """Store real-time behavioral data and analysis results."""
        try:
            with self.get_connection() as conn:
                # Determine event type
                keystroke_count = len(raw_data.get('keystrokes', []))
                mouse_count = len(raw_data.get('mouse_events', []))
                
                if keystroke_count > mouse_count:
                    event_type = 'keystroke'
                elif mouse_count > keystroke_count:
                    event_type = 'mouse'
                else:
                    event_type = 'mixed'
                
                conn.execute(
                    """INSERT INTO behavioral_data 
                       (user_id, session_id, event_type, raw_data, features, 
                        anomaly_score, is_anomaly) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (user_id,
                     anomaly_result.get('session_id', 'unknown'),
                     event_type,
                     json.dumps(raw_data),
                     json.dumps(features),
                     anomaly_result.get('score', 0.0),
                     anomaly_result.get('is_anomaly', False))
                )
                
                # Update profile counters
                conn.execute(
                    """UPDATE user_profiles 
                       SET total_keystrokes = total_keystrokes + ?, 
                           total_mouse_events = total_mouse_events + ?
                       WHERE user_id = ?""",
                    (keystroke_count, mouse_count, user_id)
                )
                
                conn.commit()
                logger.debug(f"Stored behavioral data for user {user_id}: {event_type}, anomaly={anomaly_result.get('is_anomaly', False)}")
        except Exception as e:
            logger.error(f"Error storing behavioral data for user {user_id}: {e}")
            raise
    
    def get_user_behavioral_data(self, user_id: int, limit: int = 1000) -> List[Dict]:
        """Retrieve recent behavioral data for a user."""
        try:
            with self.get_connection() as conn:
                rows = conn.execute(
                    """SELECT * FROM behavioral_data 
                       WHERE user_id = ? 
                       ORDER BY timestamp DESC 
                       LIMIT ?""",
                    (user_id, limit)
                ).fetchall()
                
                result = []
                for row in rows:
                    data = dict(row)
                    # Parse JSON fields
                    try:
                        data['raw_data'] = json.loads(data['raw_data'])
                        data['features'] = json.loads(data['features'])
                    except:
                        pass
                    result.append(data)
                
                return result
        except Exception as e:
            logger.error(f"Error getting behavioral data for user {user_id}: {e}")
            return []
    
    # Model Management Methods
    
    def store_model(self, user_id: int, model_type: str, model_data: bytes,
                   parameters: Dict, performance_metrics: Dict):
        """Store trained model for a user."""
        with self.lock:
            with self.get_connection() as conn:
                try:
                    # Deactivate old models of the same type
                    conn.execute(
                        "UPDATE model_data SET is_active = FALSE WHERE user_id = ? AND model_type = ?",
                        (user_id, model_type)
                    )
                    
                    # Insert new model
                    conn.execute(
                        """INSERT INTO model_data 
                           (user_id, model_type, model_data, parameters, performance_metrics) 
                           VALUES (?, ?, ?, ?, ?)""",
                        (user_id, model_type, model_data,
                         json.dumps(parameters), json.dumps(performance_metrics))
                    )
                    
                    conn.commit()
                    logger.info(f"Stored {model_type} model for user {user_id}")
                except Exception as e:
                    logger.error(f"Error storing model for user {user_id}: {e}")
                    raise
    
    def get_user_model(self, user_id: int, model_type: str) -> Optional[Dict]:
        """Retrieve active model for a user."""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    """SELECT * FROM model_data 
                       WHERE user_id = ? AND model_type = ? AND is_active = TRUE 
                       ORDER BY created_at DESC LIMIT 1""",
                    (user_id, model_type)
                ).fetchone()
                
                if row:
                    result = dict(row)
                    result['parameters'] = json.loads(result['parameters'])
                    result['performance_metrics'] = json.loads(result['performance_metrics'])
                    return result
                return None
        except Exception as e:
            logger.error(f"Error getting model for user {user_id}: {e}")
            return None
    
    # Analytics and Statistics Methods
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get comprehensive user statistics for dashboard."""
        try:
            with self.get_connection() as conn:
                # Basic profile stats
                profile = conn.execute(
                    "SELECT * FROM user_profiles WHERE user_id = ?",
                    (user_id,)
                ).fetchone()
                
                # Recent session count
                sessions_last_7_days = conn.execute(
                    """SELECT COUNT(*) FROM user_sessions 
                       WHERE user_id = ? AND start_time > datetime('now', '-7 days')""",
                    (user_id,)
                ).fetchone()[0]
                
                # Anomaly statistics
                anomaly_stats = conn.execute(
                    """SELECT severity, COUNT(*) as count 
                       FROM anomaly_reports 
                       WHERE user_id = ? AND timestamp > datetime('now', '-30 days')
                       GROUP BY severity""",
                    (user_id,)
                ).fetchall()
                
                # Behavioral data count
                behavioral_data_count = conn.execute(
                    """SELECT COUNT(*) FROM behavioral_data 
                       WHERE user_id = ? AND timestamp > datetime('now', '-24 hours')""",
                    (user_id,)
                ).fetchone()[0]
                
                # Recent behavioral data for charts (simplified)
                recent_data = conn.execute(
                    """SELECT timestamp, anomaly_score, features 
                       FROM behavioral_data 
                       WHERE user_id = ? AND timestamp > datetime('now', '-24 hours')
                       ORDER BY timestamp ASC
                       LIMIT 50""",
                    (user_id,)
                ).fetchall()
                
                # Calculate basic metrics
                typing_speeds = []
                timestamps = []
                
                for row in recent_data:
                    try:
                        features = json.loads(row[2])
                        if len(features) > 4:  # Ensure typing speed features exist
                            typing_speeds.append(features[4])  # typing_speed_mean
                            timestamps.append(row[0])
                    except (json.JSONDecodeError, IndexError):
                        continue
                
                return {
                    'profile': dict(profile) if profile else {},
                    'sessions_last_7_days': sessions_last_7_days,
                    'anomaly_distribution': {row[0]: row[1] for row in anomaly_stats},
                    'behavioral_data_count': behavioral_data_count,
                    'typing_speed_trend': {
                        'timestamps': timestamps,
                        'speeds': typing_speeds
                    },
                    'total_anomalies': sum(row[1] for row in anomaly_stats),
                    'last_updated': datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting user stats for {user_id}: {e}")
            return {
                'profile': {},
                'sessions_last_7_days': 0,
                'anomaly_distribution': {},
                'behavioral_data_count': 0,
                'typing_speed_trend': {'timestamps': [], 'speeds': []},
                'total_anomalies': 0,
                'last_updated': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def store_anomaly_report(self, user_id: int, session_id: str, 
                           anomaly_type: str, severity: str, 
                           confidence_score: float, feature_deviations: Dict,
                           action_taken: str):
        """Store anomaly detection report."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """INSERT INTO anomaly_reports 
                       (user_id, session_id, anomaly_type, severity, confidence_score, 
                        feature_deviations, action_taken) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, session_id, anomaly_type, severity, confidence_score,
                     json.dumps(feature_deviations), action_taken)
                )
                conn.commit()
                logger.debug(f"Stored anomaly report for user {user_id}")
        except Exception as e:
            logger.error(f"Error storing anomaly report for user {user_id}: {e}")
    
    def store_challenge_response(self, user_id: int, session_id: str,
                               challenge_type: str, challenge_text: str,
                               response_data: Dict, success: bool,
                               confidence_score: float):
        """Store challenge response data."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """INSERT INTO challenge_responses 
                       (user_id, session_id, challenge_type, challenge_text, 
                        response_data, success, confidence_score) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, session_id, challenge_type, challenge_text,
                     json.dumps(response_data), success, confidence_score)
                )
                conn.commit()
                logger.debug(f"Stored challenge response for user {user_id}")
        except Exception as e:
            logger.error(f"Error storing challenge response for user {user_id}: {e}")
    
    # Session Management Methods
    
    def create_session(self, user_id: int, session_id: str, 
                      ip_address: str = None, user_agent: str = None):
        """Create new user session record."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """INSERT INTO user_sessions 
                       (user_id, session_id, ip_address, user_agent) 
                       VALUES (?, ?, ?, ?)""",
                    (user_id, session_id, ip_address, user_agent)
                )
                conn.commit()
                logger.debug(f"Created session for user {user_id}")
        except Exception as e:
            logger.error(f"Error creating session for user {user_id}: {e}")
    
    def end_session(self, session_id: str, status: str = 'terminated'):
        """End user session."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """UPDATE user_sessions 
                       SET end_time = CURRENT_TIMESTAMP, status = ? 
                       WHERE session_id = ?""",
                    (status, session_id)
                )
                conn.commit()
                logger.debug(f"Ended session {session_id} with status {status}")
        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
    
    # Cleanup Methods
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policies."""
        with self.lock:
            with self.get_connection() as conn:
                try:
                    # Clean old behavioral data (30 days)
                    result = conn.execute(
                        "DELETE FROM behavioral_data WHERE timestamp < datetime('now', '-30 days')"
                    )
                    behavioral_deleted = result.rowcount
                    
                    # Clean old session logs (7 days)  
                    result = conn.execute(
                        "DELETE FROM user_sessions WHERE start_time < datetime('now', '-7 days')"
                    )
                    sessions_deleted = result.rowcount
                    
                    # Clean resolved anomaly reports (90 days)
                    result = conn.execute(
                        """DELETE FROM anomaly_reports 
                           WHERE timestamp < datetime('now', '-90 days') AND resolved = TRUE"""
                    )
                    anomalies_deleted = result.rowcount
                    
                    conn.commit()
                    logger.info(f"Data cleanup completed: {behavioral_deleted} behavioral records, {sessions_deleted} sessions, {anomalies_deleted} anomaly reports deleted")
                except Exception as e:
                    logger.error(f"Error during data cleanup: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics for monitoring."""
        try:
            with self.get_connection() as conn:
                stats = {}
                
                # Table row counts
                tables = ['users', 'behavioral_data', 'calibration_data', 
                         'model_data', 'anomaly_reports', 'user_sessions']
                
                for table in tables:
                    try:
                        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        stats[f'{table}_count'] = count
                    except Exception as e:
                        logger.warning(f"Error counting table {table}: {e}")
                        stats[f'{table}_count'] = -1
                
                # Database size (SQLite specific)
                try:
                    size = conn.execute("PRAGMA page_count").fetchone()[0]
                    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
                    stats['database_size_bytes'] = size * page_size
                except Exception as e:
                    logger.warning(f"Error getting database size: {e}")
                    stats['database_size_bytes'] = -1
                
                return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}