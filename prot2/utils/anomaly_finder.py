#!/usr/bin/env python3
"""
Anomaly Finder for Behavioral Biometrics

This module implements the anomaly detection engine that coordinates with
the machine learning models to identify suspicious behavioral patterns
and trigger appropriate security responses.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque

from config import Config
from models.behavioral_models import BehavioralModels

logger = logging.getLogger(__name__)

class AnomalyFinder:
    """Advanced anomaly detection engine for continuous authentication."""
    
    def __init__(self):
        """Initialize the anomaly detection system."""
        self.config = Config.ANOMALY_THRESHOLDS
        self.behavioral_models = BehavioralModels()
        
        # User-specific anomaly history
        self.user_anomaly_history = defaultdict(lambda: deque(maxlen=100))
        self.user_baseline_stats = defaultdict(dict)
        self.user_recent_scores = defaultdict(lambda: deque(maxlen=20))
        
        # Verification thresholds
        self.verification_thresholds = {
            'strict': 0.9,
            'moderate': 0.7,
            'lenient': 0.5
        }
        
        logger.info("AnomalyFinder initialized with ensemble detection methods")
    
    def detect_anomaly(self, user_id: int, features: List[float], 
                      session_id: str = None) -> Dict[str, Any]:
        """
        Main anomaly detection method that coordinates all detection algorithms.
        
        Args:
            user_id: User identifier
            features: Extracted behavioral features
            session_id: Current session identifier
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            timestamp = datetime.utcnow()
            
            # Get ensemble prediction from behavioral models
            model_result = self.behavioral_models.predict_anomaly(user_id, features)
            
            # Enhanced analysis with historical context
            contextual_analysis = self._analyze_behavioral_context(
                user_id, features, model_result
            )
            
            # Risk scoring and severity assessment
            risk_assessment = self._assess_risk_level(
                user_id, model_result, contextual_analysis
            )
            
            # Generate comprehensive anomaly report
            anomaly_result = {
                'user_id': user_id,
                'session_id': session_id or 'unknown',
                'timestamp': timestamp.isoformat(),
                'is_anomaly': risk_assessment['is_anomaly'],
                'severity': risk_assessment['severity'],
                'confidence': risk_assessment['confidence'],
                'score': risk_assessment['anomaly_score'],
                'model_results': model_result,
                'contextual_analysis': contextual_analysis,
                'risk_factors': risk_assessment['risk_factors'],
                'recommended_action': risk_assessment['recommended_action'],
                'feature_deviations': self._analyze_feature_deviations(user_id, features)
            }
            
            # Update user history
            self._update_user_history(user_id, anomaly_result)
            
            # Log significant anomalies
            if anomaly_result['is_anomaly'] and anomaly_result['severity'] in ['high', 'critical']:
                logger.warning(
                    f"High-severity anomaly detected for user {user_id}: "
                    f"confidence={anomaly_result['confidence']:.3f}, "
                    f"severity={anomaly_result['severity']}"
                )
            
            return anomaly_result
            
        except Exception as e:
            logger.error(f"Error in anomaly detection for user {user_id}: {e}")
            return {
                'user_id': user_id,
                'session_id': session_id or 'unknown',
                'timestamp': datetime.utcnow().isoformat(),
                'is_anomaly': False,
                'severity': 'unknown',
                'confidence': 0.0,
                'score': 0.0,
                'error': str(e)
            }
    
    def _analyze_behavioral_context(self, user_id: int, features: List[float], 
                                   model_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral context using historical patterns."""
        
        context_analysis = {
            'temporal_consistency': 0.0,
            'session_stability': 0.0,
            'pattern_deviation': 0.0,
            'adaptation_score': 0.0
        }
        
        try:
            user_key = f"user_{user_id}"
            recent_scores = self.user_recent_scores[user_key]
            
            # Add current score to history
            current_score = model_result.get('confidence', 0.0)
            recent_scores.append(current_score)
            
            # Temporal consistency analysis
            if len(recent_scores) >= 5:
                score_variance = np.var(list(recent_scores))
                context_analysis['temporal_consistency'] = max(0, 1.0 - score_variance)
            
            # Session stability analysis
            context_analysis['session_stability'] = self._calculate_session_stability(
                user_id, features
            )
            
            # Pattern deviation analysis
            context_analysis['pattern_deviation'] = self._calculate_pattern_deviation(
                user_id, features
            )
            
            # Adaptation score (how well user adapts to detected changes)
            context_analysis['adaptation_score'] = self._calculate_adaptation_score(
                user_id
            )
            
        except Exception as e:
            logger.error(f"Error in behavioral context analysis: {e}")
        
        return context_analysis
    
    def _assess_risk_level(self, user_id: int, model_result: Dict[str, Any], 
                          contextual_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk level combining model and contextual analysis."""
        
        # Base confidence from ensemble models
        base_confidence = model_result.get('confidence', 0.0)
        
        # Contextual modifiers
        temporal_modifier = 1.0 - contextual_analysis.get('temporal_consistency', 0.0)
        stability_modifier = 1.0 - contextual_analysis.get('session_stability', 0.0)
        pattern_modifier = contextual_analysis.get('pattern_deviation', 0.0)
        
        # Calculate weighted risk score
        risk_weights = {
            'base_confidence': 0.4,
            'temporal_inconsistency': 0.2,
            'session_instability': 0.2,
            'pattern_deviation': 0.2
        }
        
        anomaly_score = (
            risk_weights['base_confidence'] * base_confidence +
            risk_weights['temporal_inconsistency'] * temporal_modifier +
            risk_weights['session_instability'] * stability_modifier +
            risk_weights['pattern_deviation'] * pattern_modifier
        )
        
        # Determine severity level
        severity = 'normal'
        is_anomaly = False
        
        if anomaly_score >= self.config['critical']:
            severity = 'critical'
            is_anomaly = True
        elif anomaly_score >= self.config['high']:
            severity = 'high'
            is_anomaly = True
        elif anomaly_score >= self.config['medium']:
            severity = 'medium'
            is_anomaly = True
        elif anomaly_score >= self.config['low']:
            severity = 'low'
            is_anomaly = True
        
        # Identify specific risk factors
        risk_factors = []
        
        if base_confidence > 0.7:
            risk_factors.append('high_model_confidence')
        if temporal_modifier > 0.5:
            risk_factors.append('temporal_inconsistency')
        if stability_modifier > 0.5:
            risk_factors.append('session_instability')
        if pattern_modifier > 0.5:
            risk_factors.append('significant_pattern_deviation')
        
        # Determine recommended action
        recommended_action = self._determine_recommended_action(severity, risk_factors)
        
        return {
            'is_anomaly': is_anomaly,
            'severity': severity,
            'confidence': float(anomaly_score),
            'anomaly_score': float(anomaly_score),
            'risk_factors': risk_factors,
            'recommended_action': recommended_action,
            'base_confidence': float(base_confidence),
            'contextual_modifiers': {
                'temporal': float(temporal_modifier),
                'stability': float(stability_modifier),
                'pattern': float(pattern_modifier)
            }
        }
    
    def _calculate_session_stability(self, user_id: int, features: List[float]) -> float:
        """Calculate session stability based on feature consistency."""
        
        user_key = f"user_{user_id}"
        
        if user_key not in self.user_baseline_stats:
            # Initialize baseline stats
            self.user_baseline_stats[user_key] = {
                'feature_means': np.array(features),
                'feature_stds': np.zeros(len(features)),
                'sample_count': 1
            }
            return 1.0  # Perfect stability for first sample
        
        baseline = self.user_baseline_stats[user_key]
        
        # Calculate feature deviations from baseline
        deviations = np.abs(np.array(features) - baseline['feature_means'])
        
        # Normalize by baseline standard deviations (avoid division by zero)
        normalized_deviations = deviations / np.maximum(baseline['feature_stds'], 0.1)
        
        # Calculate stability score (lower deviation = higher stability)
        stability_score = 1.0 / (1.0 + np.mean(normalized_deviations))
        
        # Update baseline stats incrementally
        self._update_baseline_stats(user_key, features)
        
        return float(stability_score)
    
    def _calculate_pattern_deviation(self, user_id: int, features: List[float]) -> float:
        """Calculate deviation from established behavioral patterns."""
        
        try:
            # Use statistical methods to measure pattern deviation
            user_history = self.user_anomaly_history[f"user_{user_id}"]
            
            if len(user_history) < 5:
                return 0.0  # Not enough history for pattern analysis
            
            # Extract recent feature vectors
            recent_features = []
            for record in list(user_history)[-10:]:  # Last 10 records
                if 'features' in record:
                    recent_features.append(record['features'])
            
            if not recent_features:
                return 0.0
            
            # Calculate statistical deviation
            recent_mean = np.mean(recent_features, axis=0)
            recent_std = np.std(recent_features, axis=0)
            
            # Calculate z-scores for current features
            z_scores = np.abs((np.array(features) - recent_mean) / 
                            np.maximum(recent_std, 0.1))
            
            # Return normalized deviation score
            deviation_score = np.mean(z_scores) / 3.0  # Normalize by 3-sigma rule
            return min(1.0, deviation_score)
            
        except Exception as e:
            logger.error(f"Error calculating pattern deviation: {e}")
            return 0.0
    
    def _calculate_adaptation_score(self, user_id: int) -> float:
        """Calculate how well user adapts to behavioral changes."""
        
        user_history = self.user_anomaly_history[f"user_{user_id}"]
        
        if len(user_history) < 10:
            return 0.5  # Neutral score for insufficient data
        
        # Analyze trend in anomaly scores over time
        recent_scores = [record.get('score', 0.0) for record in list(user_history)[-10:]]
        
        # Calculate trend (positive = increasing anomalies, negative = adapting)
        if len(recent_scores) >= 3:
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]
            
            # Convert slope to adaptation score (negative slope = good adaptation)
            adaptation_score = max(0.0, min(1.0, 0.5 - slope))
            return adaptation_score
        
        return 0.5
    
    def _determine_recommended_action(self, severity: str, 
                                    risk_factors: List[str]) -> str:
        """Determine recommended security action based on risk assessment."""
        
        if severity == 'critical':
            return 'terminate_session'
        elif severity == 'high':
            if 'high_model_confidence' in risk_factors:
                return 'challenge_verification'
            else:
                return 'enhanced_monitoring'
        elif severity == 'medium':
            if len(risk_factors) >= 2:
                return 'challenge_verification'
            else:
                return 'alert_user'
        elif severity == 'low':
            return 'log_incident'
        else:
            return 'continue_monitoring'
    
    def _analyze_feature_deviations(self, user_id: int, 
                                   features: List[float]) -> Dict[str, Any]:
        """Analyze which specific features deviate most from normal patterns."""
        
        deviations = {}
        
        try:
            user_key = f"user_{user_id}"
            
            if user_key in self.user_baseline_stats:
                baseline = self.user_baseline_stats[user_key]
                feature_means = baseline['feature_means']
                feature_stds = baseline['feature_stds']
                
                # Calculate z-scores for each feature
                z_scores = np.abs((np.array(features) - feature_means) / 
                                np.maximum(feature_stds, 0.1))
                
                # Identify significantly deviant features (z-score > 2)
                significant_deviations = {}
                feature_names = (
                    Config.FEATURE_CONFIG['keystroke_features'] + 
                    Config.FEATURE_CONFIG['mouse_features']
                )
                
                for i, (name, z_score) in enumerate(zip(feature_names, z_scores)):
                    if z_score > 2.0:  # 2-sigma threshold
                        significant_deviations[name] = {
                            'z_score': float(z_score),
                            'current_value': float(features[i]),
                            'baseline_mean': float(feature_means[i]),
                            'baseline_std': float(feature_stds[i])
                        }
                
                deviations = {
                    'significant_deviations': significant_deviations,
                    'max_z_score': float(np.max(z_scores)),
                    'mean_z_score': float(np.mean(z_scores)),
                    'deviation_count': int(np.sum(z_scores > 2.0))
                }
        
        except Exception as e:
            logger.error(f"Error analyzing feature deviations: {e}")
            deviations = {'error': str(e)}
        
        return deviations
    
    def _update_baseline_stats(self, user_key: str, features: List[float]):
        """Update baseline statistics incrementally."""
        
        baseline = self.user_baseline_stats[user_key]
        n = baseline['sample_count']
        
        # Incremental mean update
        new_mean = (baseline['feature_means'] * n + np.array(features)) / (n + 1)
        
        # Incremental standard deviation update
        if n > 1:
            old_variance = baseline['feature_stds'] ** 2
            new_variance = (old_variance * (n - 1) + 
                          (np.array(features) - baseline['feature_means']) * 
                          (np.array(features) - new_mean)) / n
            new_std = np.sqrt(np.maximum(new_variance, 0.0))
        else:
            new_std = np.abs(np.array(features) - new_mean)
        
        # Update baseline
        baseline['feature_means'] = new_mean
        baseline['feature_stds'] = new_std
        baseline['sample_count'] = n + 1
    
    def _update_user_history(self, user_id: int, anomaly_result: Dict[str, Any]):
        """Update user's anomaly detection history."""
        
        user_key = f"user_{user_id}"
        history_record = {
            'timestamp': anomaly_result['timestamp'],
            'score': anomaly_result['score'],
            'severity': anomaly_result['severity'],
            'is_anomaly': anomaly_result['is_anomaly'],
            'confidence': anomaly_result['confidence'],
            'features': anomaly_result.get('features'),  # May not always be present
            'risk_factors': anomaly_result['risk_factors']
        }
        
        self.user_anomaly_history[user_key].append(history_record)
    
    def verify_identity(self, user_id: int, challenge_features: List[float], 
                       verification_mode: str = 'moderate') -> Dict[str, Any]:
        """
        Verify user identity using challenge response data.
        
        Args:
            user_id: User identifier
            challenge_features: Features extracted from challenge response
            verification_mode: Verification strictness ('strict', 'moderate', 'lenient')
            
        Returns:
            Verification result dictionary
        """
        try:
            # Get verification threshold
            threshold = self.verification_thresholds.get(verification_mode, 0.7)
            
            # Run anomaly detection on challenge data
            challenge_result = self.behavioral_models.predict_anomaly(user_id, challenge_features)
            
            # Calculate verification confidence
            base_confidence = challenge_result.get('confidence', 0.0)
            
            # Contextual verification factors
            user_key = f"user_{user_id}"
            recent_history = list(self.user_anomaly_history[user_key])[-5:]
            
            # Historical consistency check
            historical_consistency = 1.0
            if recent_history:
                recent_scores = [r['score'] for r in recent_history]
                consistency = 1.0 - np.std(recent_scores + [base_confidence])
                historical_consistency = max(0.0, consistency)
            
            # Combined verification score
            verification_score = (
                0.7 * (1.0 - base_confidence) +  # Lower anomaly score = higher verification
                0.3 * historical_consistency
            )
            
            # Verification decision
            verified = verification_score >= threshold
            
            verification_result = {
                'verified': verified,
                'confidence': float(verification_score),
                'threshold': threshold,
                'verification_mode': verification_mode,
                'base_anomaly_score': float(base_confidence),
                'historical_consistency': float(historical_consistency),
                'timestamp': datetime.utcnow().isoformat(),
                'challenge_analysis': challenge_result
            }
            
            logger.info(
                f"Identity verification for user {user_id}: "
                f"verified={verified}, confidence={verification_score:.3f}"
            )
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Error in identity verification for user {user_id}: {e}")
            return {
                'verified': False,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_user_anomaly_summary(self, user_id: int, 
                                days: int = 7) -> Dict[str, Any]:
        """Get summary of user's anomaly history over specified period."""
        
        user_key = f"user_{user_id}"
        history = list(self.user_anomaly_history[user_key])
        
        if not history:
            return {
                'user_id': user_id,
                'period_days': days,
                'total_events': 0,
                'anomaly_count': 0,
                'anomaly_rate': 0.0,
                'severity_distribution': {},
                'average_confidence': 0.0
            }
        
        # Filter by time period
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        recent_history = [
            record for record in history 
            if datetime.fromisoformat(record['timestamp']) >= cutoff_time
        ]
        
        if not recent_history:
            return {
                'user_id': user_id,
                'period_days': days,
                'total_events': 0,
                'anomaly_count': 0,
                'anomaly_rate': 0.0,
                'severity_distribution': {},
                'average_confidence': 0.0
            }
        
        # Calculate statistics
        total_events = len(recent_history)
        anomaly_count = sum(1 for r in recent_history if r['is_anomaly'])
        anomaly_rate = anomaly_count / total_events
        
        # Severity distribution
        severity_counts = defaultdict(int)
        for record in recent_history:
            severity_counts[record['severity']] += 1
        
        # Average confidence
        avg_confidence = np.mean([r['confidence'] for r in recent_history])
        
        # Risk factor analysis
        all_risk_factors = []
        for record in recent_history:
            all_risk_factors.extend(record.get('risk_factors', []))
        
        risk_factor_counts = Counter(all_risk_factors)
        
        return {
            'user_id': user_id,
            'period_days': days,
            'total_events': total_events,
            'anomaly_count': anomaly_count,
            'anomaly_rate': float(anomaly_rate),
            'severity_distribution': dict(severity_counts),
            'average_confidence': float(avg_confidence),
            'top_risk_factors': dict(risk_factor_counts.most_common(5)),
            'recent_trend': self._calculate_trend(recent_history),
            'summary_generated': datetime.utcnow().isoformat()
        }
    
    def _calculate_trend(self, history: List[Dict]) -> str:
        """Calculate trend in anomaly scores over time."""
        
        if len(history) < 3:
            return 'insufficient_data'
        
        scores = [record['score'] for record in history]
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def reset_user_baseline(self, user_id: int):
        """Reset user's behavioral baseline (useful after significant life changes)."""
        
        user_key = f"user_{user_id}"
        
        # Clear baseline statistics
        if user_key in self.user_baseline_stats:
            del self.user_baseline_stats[user_key]
        
        # Clear recent scores
        if user_key in self.user_recent_scores:
            self.user_recent_scores[user_key].clear()
        
        # Keep anomaly history but mark reset point
        self.user_anomaly_history[user_key].append({
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'baseline_reset',
            'note': 'User baseline statistics reset'
        })
        
        logger.info(f"Reset behavioral baseline for user {user_id}")
    
    def export_user_data(self, user_id: int) -> Dict[str, Any]:
        """Export user's anomaly detection data for analysis or transfer."""
        
        user_key = f"user_{user_id}"
        
        export_data = {
            'user_id': user_id,
            'export_timestamp': datetime.utcnow().isoformat(),
            'anomaly_history': list(self.user_anomaly_history[user_key]),
            'baseline_stats': self.user_baseline_stats.get(user_key, {}),
            'recent_scores': list(self.user_recent_scores[user_key]),
            'summary': self.get_user_anomaly_summary(user_id, days=30)
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if 'baseline_stats' in export_data and export_data['baseline_stats']:
            baseline = export_data['baseline_stats']
            if 'feature_means' in baseline:
                baseline['feature_means'] = baseline['feature_means'].tolist()
            if 'feature_stds' in baseline:
                baseline['feature_stds'] = baseline['feature_stds'].tolist()
        
        return export_data