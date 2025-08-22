#!/usr/bin/env python3
"""
Drift Detector for Behavioral Biometrics

This module implements statistical drift detection methods to identify
gradual changes in user behavior patterns that require model adaptation.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from scipy import stats
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

from config import Config

logger = logging.getLogger(__name__)

class DriftDetector:
    """Advanced drift detection system for behavioral pattern changes."""
    
    def __init__(self):
        """Initialize the drift detection system."""
        self.config = Config.DRIFT_CONFIG
        self.window_size = self.config['window_size']
        self.sensitivity = self.config['sensitivity']
        self.adaptation_threshold = self.config['adaptation_threshold']
        
        # User-specific drift tracking
        self.user_feature_windows = defaultdict(lambda: deque(maxlen=self.window_size))
        self.user_reference_distributions = defaultdict(dict)
        self.user_drift_history = defaultdict(list)
        self.user_adaptation_status = defaultdict(dict)
        
        # Drift detection methods
        self.detection_methods = {
            'ks_test': self._kolmogorov_smirnov_test,
            'wasserstein_distance': self._wasserstein_distance_test,
            'population_stability_index': self._population_stability_index,
            'statistical_variance': self._statistical_variance_test
        }
        
        logger.info("DriftDetector initialized with multiple statistical methods")
    
    def check_drift(self, user_id: int, features: List[float]) -> Dict[str, Any]:
        """
        Main drift detection method that analyzes behavioral pattern changes.
        
        Args:
            user_id: User identifier
            features: Current behavioral feature vector
            
        Returns:
            Drift detection results dictionary
        """
        try:
            user_key = f"user_{user_id}"
            current_time = datetime.utcnow()
            
            # Add features to rolling window
            self.user_feature_windows[user_key].append({
                'features': features,
                'timestamp': current_time.isoformat()
            })
            
            # Initialize reference distribution if needed
            if user_key not in self.user_reference_distributions:
                self._initialize_reference_distribution(user_key)
            
            # Check if we have enough data for drift detection
            if len(self.user_feature_windows[user_key]) < self.window_size // 2:
                return {
                    'drift_detected': False,
                    'confidence': 0.0,
                    'drift_type': 'none',
                    'affected_features': [],
                    'recommendation': 'continue_monitoring',
                    'status': 'insufficient_data',
                    'window_size': len(self.user_feature_windows[user_key])
                }
            
            # Perform drift detection analysis
            drift_analysis = self._perform_drift_analysis(user_key, features)
            
            # Assess drift significance and type
            drift_assessment = self._assess_drift_significance(user_key, drift_analysis)
            
            # Update drift history
            self._update_drift_history(user_key, drift_assessment)
            
            # Generate drift report
            drift_result = {
                'user_id': user_id,
                'timestamp': current_time.isoformat(),
                'drift_detected': drift_assessment['drift_detected'],
                'confidence': drift_assessment['confidence'],
                'drift_type': drift_assessment['drift_type'],
                'severity': drift_assessment['severity'],
                'affected_features': drift_assessment['affected_features'],
                'drift_methods': drift_analysis,
                'recommendation': drift_assessment['recommendation'],
                'adaptation_required': drift_assessment['adaptation_required'],
                'window_status': {
                    'current_size': len(self.user_feature_windows[user_key]),
                    'max_size': self.window_size,
                    'data_span_hours': self._calculate_data_span(user_key)
                }
            }
            
            # Log significant drift events
            if drift_result['drift_detected'] and drift_result['severity'] in ['moderate', 'high']:
                logger.info(
                    f"Behavioral drift detected for user {user_id}: "
                    f"type={drift_result['drift_type']}, "
                    f"confidence={drift_result['confidence']:.3f}, "
                    f"severity={drift_result['severity']}"
                )
            
            return drift_result
            
        except Exception as e:
            logger.error(f"Error in drift detection for user {user_id}: {e}")
            return {
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat(),
                'drift_detected': False,
                'confidence': 0.0,
                'drift_type': 'error',
                'error': str(e)
            }
    
    def _initialize_reference_distribution(self, user_key: str):
        """Initialize reference distribution for a new user."""
        
        self.user_reference_distributions[user_key] = {
            'initialized': False,
            'feature_means': None,
            'feature_stds': None,
            'feature_distributions': {},
            'last_updated': datetime.utcnow().isoformat(),
            'sample_count': 0
        }
        
        logger.info(f"Initialized reference distribution for {user_key}")
    
    def _perform_drift_analysis(self, user_key: str, current_features: List[float]) -> Dict[str, Any]:
        """Perform comprehensive drift analysis using multiple methods."""
        
        analysis_results = {}
        
        try:
            # Get current and reference data
            current_window = list(self.user_feature_windows[user_key])
            reference_dist = self.user_reference_distributions[user_key]
            
            # Extract feature matrices
            current_features_matrix = np.array([item['features'] for item in current_window])
            
            # Update reference distribution if needed
            if not reference_dist['initialized'] or len(current_window) >= self.window_size:
                self._update_reference_distribution(user_key, current_features_matrix)
                reference_dist = self.user_reference_distributions[user_key]
            
            if not reference_dist['initialized']:
                return {'status': 'reference_not_ready'}
            
            # Apply each drift detection method
            for method_name, method_func in self.detection_methods.items():
                try:
                    method_result = method_func(
                        reference_dist, 
                        current_features_matrix,
                        current_features
                    )
                    analysis_results[method_name] = method_result
                except Exception as e:
                    logger.warning(f"Error in drift method {method_name}: {e}")
                    analysis_results[method_name] = {
                        'drift_score': 0.0,
                        'p_value': 1.0,
                        'status': 'error',
                        'error': str(e)
                    }
            
        except Exception as e:
            logger.error(f"Error in drift analysis: {e}")
            analysis_results = {'error': str(e)}
        
        return analysis_results
    
    def _kolmogorov_smirnov_test(self, reference_dist: Dict, 
                                current_matrix: np.ndarray, 
                                current_features: List[float]) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for drift detection."""
        
        try:
            if reference_dist['feature_distributions'] is None:
                return {'drift_score': 0.0, 'p_value': 1.0, 'status': 'no_reference'}
            
            ks_scores = []
            p_values = []
            feature_drift_scores = {}
            
            # Test each feature dimension
            for i in range(min(len(current_features), len(reference_dist['feature_means']))):
                reference_values = reference_dist['feature_distributions'].get(f'feature_{i}', [])
                current_values = current_matrix[:, i]
                
                if len(reference_values) > 5 and len(current_values) > 5:
                    ks_stat, p_value = stats.ks_2samp(reference_values, current_values)
                    ks_scores.append(ks_stat)
                    p_values.append(p_value)
                    feature_drift_scores[f'feature_{i}'] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'drift_detected': p_value < self.sensitivity
                    }
            
            # Aggregate results
            overall_drift_score = np.mean(ks_scores) if ks_scores else 0.0
            overall_p_value = np.mean(p_values) if p_values else 1.0
            
            return {
                'drift_score': float(overall_drift_score),
                'p_value': float(overall_p_value),
                'feature_results': feature_drift_scores,
                'drift_detected': overall_p_value < self.sensitivity,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in KS test: {e}")
            return {'drift_score': 0.0, 'p_value': 1.0, 'status': 'error', 'error': str(e)}
    
    def _wasserstein_distance_test(self, reference_dist: Dict, 
                                  current_matrix: np.ndarray, 
                                  current_features: List[float]) -> Dict[str, Any]:
        """Perform Wasserstein distance test for drift detection."""
        
        try:
            if reference_dist['feature_distributions'] is None:
                return {'drift_score': 0.0, 'status': 'no_reference'}
            
            distances = []
            feature_distances = {}
            
            # Calculate Wasserstein distance for each feature
            for i in range(min(len(current_features), len(reference_dist['feature_means']))):
                reference_values = reference_dist['feature_distributions'].get(f'feature_{i}', [])
                current_values = current_matrix[:, i]
                
                if len(reference_values) > 1 and len(current_values) > 1:
                    distance = wasserstein_distance(reference_values, current_values)
                    distances.append(distance)
                    feature_distances[f'feature_{i}'] = {
                        'distance': float(distance),
                        'normalized_distance': float(distance / (np.std(reference_values) + 1e-6))
                    }
            
            # Aggregate results
            overall_distance = np.mean(distances) if distances else 0.0
            normalized_distance = overall_distance / (np.mean([
                np.std(reference_dist['feature_distributions'].get(f'feature_{i}', [1]))
                for i in range(len(current_features))
            ]) + 1e-6)
            
            # Drift detection based on normalized distance threshold
            drift_detected = normalized_distance > 1.0  # 1 standard deviation threshold
            
            return {
                'drift_score': float(normalized_distance),
                'raw_distance': float(overall_distance),
                'feature_distances': feature_distances,
                'drift_detected': drift_detected,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in Wasserstein distance test: {e}")
            return {'drift_score': 0.0, 'status': 'error', 'error': str(e)}
    
    def _population_stability_index(self, reference_dist: Dict, 
                                   current_matrix: np.ndarray, 
                                   current_features: List[float]) -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI) for drift detection."""
        
        try:
            if reference_dist['feature_distributions'] is None:
                return {'drift_score': 0.0, 'status': 'no_reference'}
            
            psi_scores = []
            feature_psi_scores = {}
            
            # Calculate PSI for each feature
            for i in range(min(len(current_features), len(reference_dist['feature_means']))):
                reference_values = reference_dist['feature_distributions'].get(f'feature_{i}', [])
                current_values = current_matrix[:, i]
                
                if len(reference_values) > 10 and len(current_values) > 10:
                    psi_score = self._calculate_psi(reference_values, current_values)
                    psi_scores.append(psi_score)
                    feature_psi_scores[f'feature_{i}'] = {
                        'psi_score': float(psi_score),
                        'drift_level': self._interpret_psi_score(psi_score)
                    }
            
            # Aggregate PSI scores
            overall_psi = np.mean(psi_scores) if psi_scores else 0.0
            
            # PSI interpretation
            if overall_psi < 0.1:
                drift_level = 'no_drift'
                drift_detected = False
            elif overall_psi < 0.2:
                drift_level = 'minor_drift'
                drift_detected = False
            else:
                drift_level = 'significant_drift'
                drift_detected = True
            
            return {
                'drift_score': float(overall_psi),
                'drift_level': drift_level,
                'feature_psi_scores': feature_psi_scores,
                'drift_detected': drift_detected,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in PSI calculation: {e}")
            return {'drift_score': 0.0, 'status': 'error', 'error': str(e)}
    
    def _statistical_variance_test(self, reference_dist: Dict, 
                                  current_matrix: np.ndarray, 
                                  current_features: List[float]) -> Dict[str, Any]:
        """Perform statistical variance test for drift detection."""
        
        try:
            if reference_dist['feature_means'] is None:
                return {'drift_score': 0.0, 'status': 'no_reference'}
            
            # Calculate z-scores for current features vs reference
            reference_means = reference_dist['feature_means']
            reference_stds = reference_dist['feature_stds']
            
            z_scores = np.abs((np.array(current_features) - reference_means) / 
                            (reference_stds + 1e-6))
            
            # Calculate drift metrics
            max_z_score = np.max(z_scores)
            mean_z_score = np.mean(z_scores)
            
            # Count features with significant drift (z-score > 2)
            significant_drift_count = np.sum(z_scores > 2.0)
            drift_ratio = significant_drift_count / len(z_scores)
            
            # Overall drift score based on statistical significance
            drift_score = mean_z_score / 3.0  # Normalize by 3-sigma rule
            drift_detected = drift_ratio > 0.2 or max_z_score > 3.0
            
            return {
                'drift_score': float(drift_score),
                'max_z_score': float(max_z_score),
                'mean_z_score': float(mean_z_score),
                'significant_drift_count': int(significant_drift_count),
                'drift_ratio': float(drift_ratio),
                'z_scores': z_scores.tolist(),
                'drift_detected': drift_detected,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error in variance test: {e}")
            return {'drift_score': 0.0, 'status': 'error', 'error': str(e)}
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, 
                      buckets: int = 10) -> float:
        """Calculate Population Stability Index between two distributions."""
        
        try:
            # Create bins based on reference distribution percentiles
            bin_edges = np.percentile(reference, np.linspace(0, 100, buckets + 1))
            bin_edges[0] = -np.inf  # Handle edge cases
            bin_edges[-1] = np.inf
            
            # Calculate distributions
            ref_dist, _ = np.histogram(reference, bins=bin_edges)
            cur_dist, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize to percentages
            ref_dist = ref_dist / len(reference)
            cur_dist = cur_dist / len(current)
            
            # Handle zero values (add small epsilon)
            ref_dist = np.where(ref_dist == 0, 1e-6, ref_dist)
            cur_dist = np.where(cur_dist == 0, 1e-6, cur_dist)
            
            # Calculate PSI
            psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
            
            return float(psi)
            
        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0
    
    def _interpret_psi_score(self, psi_score: float) -> str:
        """Interpret PSI score levels."""
        
        if psi_score < 0.1:
            return 'no_drift'
        elif psi_score < 0.2:
            return 'minor_drift'
        elif psi_score < 0.3:
            return 'moderate_drift'
        else:
            return 'significant_drift'
    
    def _update_reference_distribution(self, user_key: str, features_matrix: np.ndarray):
        """Update reference distribution with new data."""
        
        try:
            reference_dist = self.user_reference_distributions[user_key]
            
            # Calculate distribution statistics
            feature_means = np.mean(features_matrix, axis=0)
            feature_stds = np.std(features_matrix, axis=0)
            
            # Store individual feature distributions
            feature_distributions = {}
            for i in range(features_matrix.shape[1]):
                feature_distributions[f'feature_{i}'] = features_matrix[:, i].tolist()
            
            # Update reference distribution
            reference_dist.update({
                'initialized': True,
                'feature_means': feature_means,
                'feature_stds': feature_stds,
                'feature_distributions': feature_distributions,
                'last_updated': datetime.utcnow().isoformat(),
                'sample_count': len(features_matrix)
            })
            
            logger.info(f"Updated reference distribution for {user_key} with {len(features_matrix)} samples")
            
        except Exception as e:
            logger.error(f"Error updating reference distribution: {e}")
    
    def _assess_drift_significance(self, user_key: str, 
                                  drift_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall drift significance from multiple detection methods."""
        
        # Extract drift indicators from each method
        drift_indicators = {}
        method_scores = []
        
        for method_name, method_result in drift_analysis.items():
            if isinstance(method_result, dict) and 'drift_score' in method_result:
                score = method_result['drift_score']
                detected = method_result.get('drift_detected', False)
                
                drift_indicators[method_name] = {
                    'score': score,
                    'detected': detected
                }
                method_scores.append(score)
        
        # Calculate ensemble drift confidence
        if method_scores:
            confidence = np.mean(method_scores)
            detection_consensus = sum(1 for method in drift_indicators.values() 
                                    if method['detected']) / len(drift_indicators)
        else:
            confidence = 0.0
            detection_consensus = 0.0
        
        # Determine drift type and severity
        drift_detected = detection_consensus >= 0.5 or confidence > self.adaptation_threshold
        
        if confidence >= 0.7:
            severity = 'high'
            drift_type = 'abrupt_change'
        elif confidence >= 0.4:
            severity = 'moderate'
            drift_type = 'gradual_shift'
        elif confidence >= 0.2:
            severity = 'low'
            drift_type = 'minor_variation'
        else:
            severity = 'none'
            drift_type = 'stable'
        
        # Identify affected features
        affected_features = self._identify_affected_features(drift_analysis)
        
        # Determine recommendation
        recommendation = self._determine_drift_recommendation(
            drift_detected, severity, affected_features
        )
        
        # Check if adaptation is required
        adaptation_required = (
            drift_detected and 
            severity in ['moderate', 'high'] and 
            len(affected_features) > 2
        )
        
        return {
            'drift_detected': drift_detected,
            'confidence': float(confidence),
            'detection_consensus': float(detection_consensus),
            'severity': severity,
            'drift_type': drift_type,
            'affected_features': affected_features,
            'method_results': drift_indicators,
            'recommendation': recommendation,
            'adaptation_required': adaptation_required
        }
    
    def _identify_affected_features(self, drift_analysis: Dict[str, Any]) -> List[str]:
        """Identify which specific features show drift."""
        
        affected_features = set()
        
        # Collect feature-level results from different methods
        for method_name, method_result in drift_analysis.items():
            if isinstance(method_result, dict):
                # Check KS test results
                if 'feature_results' in method_result:
                    for feature_name, feature_result in method_result['feature_results'].items():
                        if feature_result.get('drift_detected', False):
                            affected_features.add(feature_name)
                
                # Check Wasserstein distance results
                if 'feature_distances' in method_result:
                    for feature_name, feature_result in method_result['feature_distances'].items():
                        if feature_result.get('normalized_distance', 0) > 1.0:
                            affected_features.add(feature_name)
                
                # Check PSI results
                if 'feature_psi_scores' in method_result:
                    for feature_name, feature_result in method_result['feature_psi_scores'].items():
                        if feature_result.get('psi_score', 0) > 0.2:
                            affected_features.add(feature_name)
        
        return sorted(list(affected_features))
    
    def _determine_drift_recommendation(self, drift_detected: bool, 
                                       severity: str, 
                                       affected_features: List[str]) -> str:
        """Determine recommended action based on drift analysis."""
        
        if not drift_detected:
            return 'continue_monitoring'
        
        if severity == 'high':
            return 'immediate_adaptation'
        elif severity == 'moderate':
            if len(affected_features) > 5:
                return 'scheduled_adaptation'
            else:
                return 'feature_specific_adaptation'
        elif severity == 'low':
            return 'enhanced_monitoring'
        else:
            return 'continue_monitoring'
    
    def _calculate_data_span(self, user_key: str) -> float:
        """Calculate time span of data in the current window."""
        
        window = self.user_feature_windows[user_key]
        if len(window) < 2:
            return 0.0
        
        oldest = datetime.fromisoformat(window[0]['timestamp'])
        newest = datetime.fromisoformat(window[-1]['timestamp'])
        
        return (newest - oldest).total_seconds() / 3600  # Hours
    
    def _update_drift_history(self, user_key: str, drift_assessment: Dict[str, Any]):
        """Update user's drift detection history."""
        
        history_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_detected': drift_assessment['drift_detected'],
            'confidence': drift_assessment['confidence'],
            'severity': drift_assessment['severity'],
            'drift_type': drift_assessment['drift_type'],
            'affected_features': drift_assessment['affected_features'],
            'recommendation': drift_assessment['recommendation']
        }
        
        # Keep only recent history (last 30 records)
        self.user_drift_history[user_key].append(history_record)
        if len(self.user_drift_history[user_key]) > 30:
            self.user_drift_history[user_key].pop(0)
    
    def trigger_adaptation(self, user_id: int, adaptation_type: str = 'full') -> Dict[str, Any]:
        """Trigger behavioral model adaptation for a user."""
        
        user_key = f"user_{user_id}"
        
        adaptation_result = {
            'user_id': user_id,
            'adaptation_type': adaptation_type,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'initiated'
        }
        
        try:
            if adaptation_type == 'full':
                # Reset reference distribution
                self._initialize_reference_distribution(user_key)
                adaptation_result['actions'] = ['reference_distribution_reset']
                
            elif adaptation_type == 'partial':
                # Update reference with recent data
                if user_key in self.user_feature_windows:
                    current_window = list(self.user_feature_windows[user_key])
                    if len(current_window) >= self.window_size // 2:
                        features_matrix = np.array([item['features'] for item in current_window])
                        self._update_reference_distribution(user_key, features_matrix)
                        adaptation_result['actions'] = ['reference_distribution_updated']
                    else:
                        adaptation_result['status'] = 'insufficient_data'
                        adaptation_result['actions'] = []
                else:
                    adaptation_result['status'] = 'no_data'
                    adaptation_result['actions'] = []
            
            # Update adaptation status
            self.user_adaptation_status[user_key] = {
                'last_adaptation': adaptation_result['timestamp'],
                'adaptation_type': adaptation_type,
                'status': adaptation_result['status']
            }
            
            logger.info(f"Triggered {adaptation_type} adaptation for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error in adaptation trigger: {e}")
            adaptation_result.update({
                'status': 'error',
                'error': str(e)
            })
        
        return adaptation_result
    
    def get_drift_summary(self, user_id: int, days: int = 7) -> Dict[str, Any]:
        """Get summary of drift detection results for a user."""
        
        user_key = f"user_{user_id}"
        history = self.user_drift_history.get(user_key, [])
        
        if not history:
            return {
                'user_id': user_id,
                'period_days': days,
                'total_checks': 0,
                'drift_events': 0,
                'drift_rate': 0.0
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
                'total_checks': 0,
                'drift_events': 0,
                'drift_rate': 0.0
            }
        
        # Calculate statistics
        total_checks = len(recent_history)
        drift_events = sum(1 for r in recent_history if r['drift_detected'])
        drift_rate = drift_events / total_checks
        
        # Severity distribution
        severity_counts = defaultdict(int)
        for record in recent_history:
            severity_counts[record['severity']] += 1
        
        # Most affected features
        all_affected_features = []
        for record in recent_history:
            all_affected_features.extend(record.get('affected_features', []))
        
        from collections import Counter
        feature_impact_counts = Counter(all_affected_features)
        
        return {
            'user_id': user_id,
            'period_days': days,
            'total_checks': total_checks,
            'drift_events': drift_events,
            'drift_rate': float(drift_rate),
            'severity_distribution': dict(severity_counts),
            'most_affected_features': dict(feature_impact_counts.most_common(5)),
            'current_window_size': len(self.user_feature_windows[user_key]),
            'adaptation_status': self.user_adaptation_status.get(user_key, {}),
            'summary_generated': datetime.utcnow().isoformat()
        }
    
    def export_drift_data(self, user_id: int) -> Dict[str, Any]:
        """Export drift detection data for a user."""
        
        user_key = f"user_{user_id}"
        
        export_data = {
            'user_id': user_id,
            'export_timestamp': datetime.utcnow().isoformat(),
            'drift_history': self.user_drift_history.get(user_key, []),
            'reference_distribution': self.user_reference_distributions.get(user_key, {}),
            'adaptation_status': self.user_adaptation_status.get(user_key, {}),
            'current_window_size': len(self.user_feature_windows[user_key]),
            'configuration': self.config
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if 'reference_distribution' in export_data and export_data['reference_distribution']:
            ref_dist = export_data['reference_distribution']
            if 'feature_means' in ref_dist and ref_dist['feature_means'] is not None:
                ref_dist['feature_means'] = ref_dist['feature_means'].tolist()
            if 'feature_stds' in ref_dist and ref_dist['feature_stds'] is not None:
                ref_dist['feature_stds'] = ref_dist['feature_stds'].tolist()
        
        return export_data