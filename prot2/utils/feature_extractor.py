#!/usr/bin/env python3
"""
Feature Extractor for Behavioral Biometrics

This module extracts comprehensive behavioral features from keystroke and mouse
data for continuous authentication analysis. Implements 40 distinct metrics
covering keystroke dynamics and mouse behavior patterns.
"""

import numpy as np
import math
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts behavioral biometric features from user interaction data."""
    
    def __init__(self):
        """Initialize the feature extractor with configuration parameters."""
        self.keystroke_features = [
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
        ]
        
        self.mouse_features = [
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
        
        # Special keys for analysis
        self.special_keys = {
            'backspace': ['Backspace', 'Delete'],
            'shift': ['Shift', 'ShiftLeft', 'ShiftRight'],
            'enter': ['Enter', 'Return'],
            'arrow': ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'],
            'numeric': ['Numpad0', 'Numpad1', 'Numpad2', 'Numpad3', 'Numpad4',
                       'Numpad5', 'Numpad6', 'Numpad7', 'Numpad8', 'Numpad9'],
            'punctuation': ['.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '-', '_']
        }
        
        logger.info("FeatureExtractor initialized with 40 behavioral metrics")
    
    def extract_features(self, behavioral_data: Dict[str, Any]) -> List[float]:
        """
        Extract all 40 behavioral features from input data.
        
        Args:
            behavioral_data: Dictionary containing keystroke and mouse events
            
        Returns:
            List of 40 feature values
        """
        try:
            # Separate keystroke and mouse data
            keystroke_events = behavioral_data.get('keystrokes', [])
            mouse_events = behavioral_data.get('mouse_events', [])
            
            # Extract keystroke features (20 features)
            keystroke_features = self._extract_keystroke_features(keystroke_events)
            
            # Extract mouse features (20 features)
            mouse_features = self._extract_mouse_features(mouse_events)
            
            # Combine all features
            all_features = keystroke_features + mouse_features
            
            # Ensure exactly 40 features
            if len(all_features) != 40:
                logger.warning(f"Expected 40 features, got {len(all_features)}. Padding/truncating.")
                all_features = (all_features + [0.0] * 40)[:40]
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return [0.0] * 40  # Return zero features on error
    
    def _extract_keystroke_features(self, keystroke_events: List[Dict]) -> List[float]:
        """Extract 20 keystroke-related behavioral features."""
        
        if not keystroke_events:
            return [0.0] * 20
        
        features = []
        
        try:
            # Process keystroke events
            hold_times = []
            flight_times = []
            digraph_latencies = []
            trigraph_latencies = []
            
            key_sequence = []
            timestamps = []
            
            for event in keystroke_events:
                if event.get('type') == 'keydown':
                    key_sequence.append(event.get('key', ''))
                    timestamps.append(event.get('timestamp', 0))
                    
                    # Calculate hold time if keyup exists
                    keyup_event = self._find_matching_keyup(event, keystroke_events)
                    if keyup_event:
                        hold_time = keyup_event.get('timestamp', 0) - event.get('timestamp', 0)
                        if hold_time > 0:
                            hold_times.append(hold_time)
            
            # Calculate flight times (time between key releases and next key press)
            for i in range(len(timestamps) - 1):
                flight_time = timestamps[i + 1] - timestamps[i]
                if flight_time > 0:
                    flight_times.append(flight_time)
            
            # Calculate digraph and trigraph latencies
            for i in range(len(key_sequence) - 1):
                if i < len(timestamps) - 1:
                    latency = timestamps[i + 1] - timestamps[i]
                    if latency > 0:
                        digraph_latencies.append(latency)
            
            for i in range(len(key_sequence) - 2):
                if i < len(timestamps) - 2:
                    latency = timestamps[i + 2] - timestamps[i]
                    if latency > 0:
                        trigraph_latencies.append(latency)
            
            # Feature 1-2: Key Hold Time Statistics
            features.append(np.mean(hold_times) if hold_times else 0.0)
            features.append(np.std(hold_times) if len(hold_times) > 1 else 0.0)
            
            # Feature 3-4: Flight Time Statistics
            features.append(np.mean(flight_times) if flight_times else 0.0)
            features.append(np.std(flight_times) if len(flight_times) > 1 else 0.0)
            
            # Feature 5-6: Typing Speed Statistics
            typing_speeds = self._calculate_typing_speeds(timestamps, key_sequence)
            features.append(np.mean(typing_speeds) if typing_speeds else 0.0)
            features.append(np.std(typing_speeds) if len(typing_speeds) > 1 else 0.0)
            
            # Feature 7-8: Backspace and Delete Frequency
            features.append(self._calculate_key_frequency(key_sequence, self.special_keys['backspace']))
            features.append(self._calculate_key_frequency(key_sequence, ['Delete']))
            
            # Feature 9-11: Special Key Usage
            features.append(self._calculate_key_frequency(key_sequence, self.special_keys['shift']))
            features.append(self._calculate_key_frequency(key_sequence, self.special_keys['enter']))
            features.append(self._calculate_key_frequency(key_sequence, self.special_keys['arrow']))
            
            # Feature 12-13: Digraph and Trigraph Latency
            features.append(np.mean(digraph_latencies) if digraph_latencies else 0.0)
            features.append(np.mean(trigraph_latencies) if trigraph_latencies else 0.0)
            
            # Feature 14: Error Correction Rate
            features.append(self._calculate_error_correction_rate(key_sequence))
            
            # Feature 15: Capitalization Method
            features.append(self._analyze_capitalization_method(key_sequence))
            
            # Feature 16: Punctuation Frequency
            features.append(self._calculate_key_frequency(key_sequence, self.special_keys['punctuation']))
            
            # Feature 17: Word Count per Window
            features.append(self._estimate_word_count(key_sequence))
            
            # Feature 18: Special Character Frequency
            features.append(self._calculate_special_char_frequency(key_sequence))
            
            # Feature 19: Session Uptime Typing (ratio of typing time to total time)
            features.append(self._calculate_session_uptime_ratio(timestamps))
            
            # Feature 20: Numeric Keypad Usage
            features.append(self._calculate_key_frequency(key_sequence, self.special_keys['numeric']))
            
        except Exception as e:
            logger.error(f"Error extracting keystroke features: {e}")
            features = [0.0] * 20
        
        # Ensure exactly 20 features
        return (features + [0.0] * 20)[:20]
    
    def _extract_mouse_features(self, mouse_events: List[Dict]) -> List[float]:
        """Extract 20 mouse-related behavioral features."""
        
        if not mouse_events:
            return [0.0] * 20
        
        features = []
        
        try:
            # Process mouse events
            positions = []
            timestamps = []
            click_events = []
            scroll_events = []
            
            for event in mouse_events:
                if event.get('type') == 'mousemove':
                    positions.append((event.get('x', 0), event.get('y', 0)))
                    timestamps.append(event.get('timestamp', 0))
                elif event.get('type') in ['mousedown', 'mouseup', 'click']:
                    click_events.append(event)
                elif event.get('type') == 'wheel':
                    scroll_events.append(event)
            
            if not positions or len(positions) < 2:
                return [0.0] * 20
            
            # Calculate movement metrics
            speeds = self._calculate_mouse_speeds(positions, timestamps)
            accelerations = self._calculate_mouse_accelerations(speeds, timestamps)
            angles = self._calculate_movement_angles(positions)
            
            # Feature 1-2: Mouse Speed Statistics
            features.append(np.mean(speeds) if speeds else 0.0)
            features.append(np.max(speeds) if speeds else 0.0)
            
            # Feature 3: Average Mouse Acceleration
            features.append(np.mean(accelerations) if accelerations else 0.0)
            
            # Feature 4: Movement Trajectory Curvature
            features.append(self._calculate_trajectory_curvature(positions))
            
            # Feature 5: Movement Jitter
            features.append(self._calculate_movement_jitter(positions))
            
            # Feature 6: Path Straightness
            features.append(self._calculate_path_straightness(positions))
            
            # Feature 7-8: Pause Analysis
            pauses = self._detect_movement_pauses(speeds, timestamps)
            features.append(len(pauses))  # Pause count
            features.append(np.mean([p['duration'] for p in pauses]) if pauses else 0.0)
            
            # Feature 9: Click Rate
            features.append(self._calculate_click_rate(click_events, timestamps))
            
            # Feature 10-11: Click Type Frequencies
            features.append(self._calculate_double_click_frequency(click_events))
            features.append(self._calculate_right_click_frequency(click_events))
            
            # Feature 12: Click Duration
            features.append(self._calculate_click_duration(click_events))
            
            # Feature 13-14: Scroll Analysis
            features.append(self._calculate_scroll_speed(scroll_events))
            features.append(self._calculate_scroll_direction_ratio(scroll_events))
            
            # Feature 15-16: Drag and Drop Analysis
            drag_actions = self._detect_drag_actions(mouse_events)
            features.append(len(drag_actions))  # Drag count
            features.append(np.mean([a['distance'] for a in drag_actions]) if drag_actions else 0.0)
            
            # Feature 17: Hover Duration
            features.append(self._calculate_hover_duration(positions, timestamps))
            
            # Feature 18: Idle Time
            features.append(self._calculate_mouse_idle_time(timestamps))
            
            # Feature 19: Movement Angle Variance
            features.append(np.var(angles) if len(angles) > 1 else 0.0)
            
            # Feature 20: Wheel Click Frequency
            features.append(self._calculate_wheel_click_frequency(mouse_events))
            
        except Exception as e:
            logger.error(f"Error extracting mouse features: {e}")
            features = [0.0] * 20
        
        # Ensure exactly 20 features
        return (features + [0.0] * 20)[:20]
    
    # Keystroke Analysis Helper Methods
    
    def _find_matching_keyup(self, keydown_event: Dict, all_events: List[Dict]) -> Optional[Dict]:
        """Find matching keyup event for a keydown event."""
        key = keydown_event.get('key')
        timestamp = keydown_event.get('timestamp')
        
        for event in all_events:
            if (event.get('type') == 'keyup' and 
                event.get('key') == key and 
                event.get('timestamp', 0) > timestamp):
                return event
        return None
    
    def _calculate_typing_speeds(self, timestamps: List[float], keys: List[str]) -> List[float]:
        """Calculate typing speed in characters per minute for sliding windows."""
        speeds = []
        window_size = 10  # 10 key window
        
        for i in range(len(timestamps) - window_size + 1):
            window_keys = keys[i:i + window_size]
            window_time = timestamps[i + window_size - 1] - timestamps[i]
            
            if window_time > 0:
                char_count = sum(1 for key in window_keys if len(key) == 1)  # Only count character keys
                speed = (char_count / window_time) * 60000  # Convert to CPM
                speeds.append(speed)
        
        return speeds
    
    def _calculate_key_frequency(self, key_sequence: List[str], target_keys: List[str]) -> float:
        """Calculate frequency of specific keys in the sequence."""
        if not key_sequence:
            return 0.0
        
        count = sum(1 for key in key_sequence if key in target_keys)
        return count / len(key_sequence)
    
    def _calculate_error_correction_rate(self, key_sequence: List[str]) -> float:
        """Calculate error correction rate based on backspace/delete usage."""
        if not key_sequence:
            return 0.0
        
        correction_keys = self.special_keys['backspace'] + ['Delete']
        corrections = sum(1 for key in key_sequence if key in correction_keys)
        total_keys = len(key_sequence)
        
        return corrections / total_keys if total_keys > 0 else 0.0
    
    def _analyze_capitalization_method(self, key_sequence: List[str]) -> float:
        """Analyze capitalization method (0 = caps lock, 1 = shift)."""
        shift_count = sum(1 for key in key_sequence if key in self.special_keys['shift'])
        caps_lock_count = sum(1 for key in key_sequence if key == 'CapsLock')
        
        total_caps_events = shift_count + caps_lock_count
        if total_caps_events == 0:
            return 0.5  # Neutral
        
        return shift_count / total_caps_events
    
    def _estimate_word_count(self, key_sequence: List[str]) -> float:
        """Estimate number of words typed based on space characters."""
        space_count = sum(1 for key in key_sequence if key == ' ')
        return space_count + 1 if space_count > 0 else 0
    
    def _calculate_special_char_frequency(self, key_sequence: List[str]) -> float:
        """Calculate frequency of special characters (non-alphanumeric)."""
        if not key_sequence:
            return 0.0
        
        special_chars = sum(1 for key in key_sequence 
                          if len(key) == 1 and not key.isalnum() and key != ' ')
        return special_chars / len(key_sequence)
    
    def _calculate_session_uptime_ratio(self, timestamps: List[float]) -> float:
        """Calculate ratio of active typing time to total session time."""
        if len(timestamps) < 2:
            return 0.0
        
        total_session_time = timestamps[-1] - timestamps[0]
        if total_session_time <= 0:
            return 0.0
        
        # Calculate active typing time (time between consecutive keystrokes < 2 seconds)
        active_time = 0
        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap < 2000:  # 2 seconds threshold
                active_time += gap
        
        return active_time / total_session_time
    
    # Mouse Analysis Helper Methods
    
    def _calculate_mouse_speeds(self, positions: List[Tuple[float, float]], 
                               timestamps: List[float]) -> List[float]:
        """Calculate mouse movement speeds."""
        speeds = []
        
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            dt = timestamps[i + 1] - timestamps[i]
            
            if dt > 0:
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                speed = distance / dt * 1000  # pixels per second
                speeds.append(speed)
        
        return speeds
    
    def _calculate_mouse_accelerations(self, speeds: List[float], 
                                     timestamps: List[float]) -> List[float]:
        """Calculate mouse movement accelerations."""
        accelerations = []
        
        for i in range(len(speeds) - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt > 0:
                acceleration = (speeds[i + 1] - speeds[i]) / dt * 1000
                accelerations.append(abs(acceleration))
        
        return accelerations
    
    def _calculate_movement_angles(self, positions: List[Tuple[float, float]]) -> List[float]:
        """Calculate movement direction angles."""
        angles = []
        
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx != 0 or dy != 0:
                angle = math.atan2(dy, dx)
                angles.append(angle)
        
        return angles
    
    def _calculate_trajectory_curvature(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate trajectory curvature using path deviation."""
        if len(positions) < 3:
            return 0.0
        
        # Calculate total path length
        path_length = 0
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            path_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calculate straight-line distance
        x1, y1 = positions[0]
        x2, y2 = positions[-1]
        straight_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if straight_distance == 0:
            return 0.0
        
        return path_length / straight_distance - 1.0
    
    def _calculate_movement_jitter(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate movement jitter based on direction changes."""
        if len(positions) < 3:
            return 0.0
        
        angles = self._calculate_movement_angles(positions)
        
        direction_changes = 0
        for i in range(len(angles) - 1):
            angle_diff = abs(angles[i + 1] - angles[i])
            # Normalize to [0, π]
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            
            if angle_diff > math.pi / 4:  # 45 degrees threshold
                direction_changes += 1
        
        return direction_changes / len(angles) if angles else 0.0
    
    def _calculate_path_straightness(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate path straightness ratio."""
        if len(positions) < 2:
            return 1.0
        
        # Calculate straight-line distance
        x1, y1 = positions[0]
        x2, y2 = positions[-1]
        straight_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calculate actual path length
        path_length = 0
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            path_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if path_length == 0:
            return 1.0
        
        return straight_distance / path_length
    
    def _detect_movement_pauses(self, speeds: List[float], 
                               timestamps: List[float]) -> List[Dict]:
        """Detect movement pauses in mouse data."""
        pauses = []
        pause_threshold = 50  # pixels per second
        
        in_pause = False
        pause_start = 0
        
        for i, speed in enumerate(speeds):
            if speed < pause_threshold and not in_pause:
                in_pause = True
                pause_start = timestamps[i]
            elif speed >= pause_threshold and in_pause:
                in_pause = False
                pause_duration = timestamps[i] - pause_start
                pauses.append({
                    'start': pause_start,
                    'duration': pause_duration
                })
        
        return pauses
    
    def _calculate_click_rate(self, click_events: List[Dict], 
                             timestamps: List[float]) -> float:
        """Calculate mouse click rate."""
        if not timestamps or not click_events:
            return 0.0
        
        total_time = timestamps[-1] - timestamps[0]
        if total_time <= 0:
            return 0.0
        
        click_count = len([e for e in click_events if e.get('type') == 'click'])
        return click_count / total_time * 60000  # clicks per minute
    
    def _calculate_double_click_frequency(self, click_events: List[Dict]) -> float:
        """Calculate double-click frequency."""
        if len(click_events) < 2:
            return 0.0
        
        double_clicks = 0
        click_timestamps = [e.get('timestamp', 0) for e in click_events 
                          if e.get('type') == 'click']
        
        for i in range(len(click_timestamps) - 1):
            time_diff = click_timestamps[i + 1] - click_timestamps[i]
            if 0 < time_diff < 500:  # 500ms double-click threshold
                double_clicks += 1
        
        return double_clicks / len(click_events)
    
    def _calculate_right_click_frequency(self, click_events: List[Dict]) -> float:
        """Calculate right-click frequency."""
        if not click_events:
            return 0.0
        
        right_clicks = sum(1 for e in click_events 
                          if e.get('button') == 2 or e.get('which') == 3)
        return right_clicks / len(click_events)
    
    def _calculate_click_duration(self, click_events: List[Dict]) -> float:
        """Calculate average click duration."""
        durations = []
        
        # Match mousedown and mouseup events
        mousedown_events = {e.get('timestamp', 0): e for e in click_events 
                           if e.get('type') == 'mousedown'}
        mouseup_events = {e.get('timestamp', 0): e for e in click_events 
                         if e.get('type') == 'mouseup'}
        
        for down_time, down_event in mousedown_events.items():
            # Find corresponding mouseup
            for up_time, up_event in mouseup_events.items():
                if (up_time > down_time and 
                    up_event.get('button') == down_event.get('button')):
                    duration = up_time - down_time
                    if duration < 1000:  # Reasonable click duration
                        durations.append(duration)
                    break
        
        return np.mean(durations) if durations else 0.0
    
    def _calculate_scroll_speed(self, scroll_events: List[Dict]) -> float:
        """Calculate scroll speed."""
        if len(scroll_events) < 2:
            return 0.0
        
        scroll_amounts = []
        timestamps = []
        
        for event in scroll_events:
            scroll_amounts.append(abs(event.get('deltaY', 0)))
            timestamps.append(event.get('timestamp', 0))
        
        speeds = []
        for i in range(len(scroll_amounts) - 1):
            dt = timestamps[i + 1] - timestamps[i]
            if dt > 0:
                speed = scroll_amounts[i] / dt * 1000  # units per second
                speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0.0
    
    def _calculate_scroll_direction_ratio(self, scroll_events: List[Dict]) -> float:
        """Calculate ratio of upward to downward scrolling."""
        if not scroll_events:
            return 0.5
        
        up_scrolls = sum(1 for e in scroll_events if e.get('deltaY', 0) < 0)
        down_scrolls = sum(1 for e in scroll_events if e.get('deltaY', 0) > 0)
        
        total_scrolls = up_scrolls + down_scrolls
        return up_scrolls / total_scrolls if total_scrolls > 0 else 0.5
    
    def _detect_drag_actions(self, mouse_events: List[Dict]) -> List[Dict]:
        """Detect drag-and-drop actions."""
        drags = []
        
        drag_start = None
        for event in mouse_events:
            if event.get('type') == 'mousedown':
                drag_start = event
            elif event.get('type') == 'mouseup' and drag_start:
                # Calculate drag distance
                start_x = drag_start.get('x', 0)
                start_y = drag_start.get('y', 0)
                end_x = event.get('x', 0)
                end_y = event.get('y', 0)
                
                distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                duration = event.get('timestamp', 0) - drag_start.get('timestamp', 0)
                
                if distance > 10 and duration > 100:  # Minimum drag criteria
                    drags.append({
                        'distance': distance,
                        'duration': duration,
                        'start': (start_x, start_y),
                        'end': (end_x, end_y)
                    })
                
                drag_start = None
        
        return drags
    
    def _calculate_hover_duration(self, positions: List[Tuple[float, float]], 
                                 timestamps: List[float]) -> float:
        """Calculate average hover duration."""
        if len(positions) < 2:
            return 0.0
        
        hover_durations = []
        hover_threshold = 5  # pixels
        
        current_hover_start = None
        last_position = positions[0]
        
        for i, position in enumerate(positions[1:], 1):
            distance = math.sqrt((position[0] - last_position[0])**2 + 
                               (position[1] - last_position[1])**2)
            
            if distance < hover_threshold:
                if current_hover_start is None:
                    current_hover_start = timestamps[i - 1]
            else:
                if current_hover_start is not None:
                    hover_duration = timestamps[i] - current_hover_start
                    hover_durations.append(hover_duration)
                    current_hover_start = None
                last_position = position
        
        return np.mean(hover_durations) if hover_durations else 0.0
    
    def _calculate_mouse_idle_time(self, timestamps: List[float]) -> float:
        """Calculate total idle time in mouse movement."""
        if len(timestamps) < 2:
            return 0.0
        
        idle_threshold = 1000  # 1 second
        total_idle = 0
        
        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap > idle_threshold:
                total_idle += gap
        
        total_time = timestamps[-1] - timestamps[0]
        return total_idle / total_time if total_time > 0 else 0.0
    
    def _calculate_wheel_click_frequency(self, mouse_events: List[Dict]) -> float:
        """Calculate frequency of mouse wheel clicks."""
        if not mouse_events:
            return 0.0
        
        wheel_clicks = sum(1 for e in mouse_events 
                          if e.get('type') == 'click' and e.get('button') == 1)
        total_events = len(mouse_events)
        
        return wheel_clicks / total_events if total_events > 0 else 0.0