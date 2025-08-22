#!/usr/bin/env python3
"""
Behavioral Models Module for Continuous Authentication

This module implements the ensemble of machine learning models used for
behavioral biometrics analysis including neural networks and classical ML models.
"""

import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import GRU, Dense, Dropout, Input, RepeatVector, TimeDistributed
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural network models will be disabled.")

# Scikit-learn imports
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import joblib

from config import Config

logger = logging.getLogger(__name__)

class BehavioralModels:
    """Ensemble of behavioral biometric models for continuous authentication."""
    
    def __init__(self):
        """Initialize the behavioral models ensemble."""
        self.config = Config.MODEL_CONFIGS
        self.feature_scalers = {}
        self.trained_models = {}
        self.model_performance = {}
        self.is_tensorflow_available = TF_AVAILABLE
        
        # Initialize scalers
        self.feature_scalers['standard'] = StandardScaler()
        self.feature_scalers['minmax'] = MinMaxScaler()
        
        logger.info(f"BehavioralModels initialized. TensorFlow available: {self.is_tensorflow_available}")
    
    def prepare_features(self, features: List[List[float]], user_id: int) -> np.ndarray:
        """Prepare and scale features for model training/inference."""
        features_array = np.array(features)
        
        # Handle single sample case
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
        
        # Fit scaler if not already fitted for this user
        scaler_key = f"user_{user_id}"
        if scaler_key not in self.feature_scalers:
            self.feature_scalers[scaler_key] = StandardScaler()
            scaled_features = self.feature_scalers[scaler_key].fit_transform(features_array)
        else:
            scaled_features = self.feature_scalers[scaler_key].transform(features_array)
        
        return scaled_features
    
    def create_gru_model(self, input_shape: Tuple[int, int]) -> Optional[Any]:
        """Create GRU model for sequential behavioral analysis."""
        if not self.is_tensorflow_available:
            logger.warning("TensorFlow not available. Skipping GRU model creation.")
            return None
        
        try:
            config = self.config['gru']
            
            model = Sequential([
                GRU(config['hidden_units'], 
                    return_sequences=True,
                    input_shape=input_shape,
                    dropout=config['dropout_rate']),
                GRU(config['hidden_units'] // 2, 
                    dropout=config['dropout_rate']),
                Dense(32, activation='relu'),
                Dropout(config['dropout_rate']),
                Dense(1, activation='sigmoid')
            ])
            
            optimizer = Adam(learning_rate=config['learning_rate'])
            model.compile(optimizer=optimizer, 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating GRU model: {e}")
            return None
    
    def create_autoencoder_model(self, input_dim: int) -> Optional[Any]:
        """Create autoencoder model for anomaly detection."""
        if not self.is_tensorflow_available:
            logger.warning("TensorFlow not available. Skipping autoencoder model creation.")
            return None
        
        try:
            config = self.config['autoencoder']
            encoding_dim = config['encoding_dim']
            
            # Encoder
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(config['hidden_layers'][0], activation='relu')(input_layer)
            encoded = Dense(config['hidden_layers'][1], activation='relu')(encoded)
            encoded = Dense(encoding_dim, activation='relu')(encoded)
            
            # Decoder
            decoded = Dense(config['hidden_layers'][1], activation='relu')(encoded)
            decoded = Dense(config['hidden_layers'][0], activation='relu')(decoded)
            decoded = Dense(input_dim, activation='sigmoid')(decoded)
            
            # Autoencoder model
            autoencoder = Model(input_layer, decoded)
            encoder = Model(input_layer, encoded)
            
            optimizer = Adam(learning_rate=config['learning_rate'])
            autoencoder.compile(optimizer=optimizer, loss='mse')
            
            return {'autoencoder': autoencoder, 'encoder': encoder}
            
        except Exception as e:
            logger.error(f"Error creating autoencoder model: {e}")
            return None
    
    def create_isolation_forest(self) -> IsolationForest:
        """Create Isolation Forest model for outlier detection."""
        config = self.config['isolation_forest']
        
        model = IsolationForest(
            n_estimators=config['n_estimators'],
            contamination=config['contamination'],
            random_state=config['random_state'],
            n_jobs=-1
        )
        
        return model
    
    def create_one_class_svm(self) -> OneClassSVM:
        """Create One-Class SVM model for novelty detection."""
        config = self.config['one_class_svm']
        
        model = OneClassSVM(
            kernel=config['kernel'],
            gamma=config['gamma'],
            nu=config['nu']
        )
        
        return model
    
    def create_passive_aggressive_classifier(self) -> PassiveAggressiveClassifier:
        """Create Passive-Aggressive classifier for adaptive learning."""
        config = self.config['passive_aggressive']
        
        model = PassiveAggressiveClassifier(
            C=config['C'],
            random_state=config['random_state'],
            max_iter=config['max_iter']
        )
        
        return model
    
    def create_incremental_knn(self) -> LocalOutlierFactor:
        """Create incremental k-NN model using Local Outlier Factor."""
        config = self.config['incremental_knn']
        
        model = LocalOutlierFactor(
            n_neighbors=config['n_neighbors'],
            contamination=config['contamination'],
            novelty=True
        )
        
        return model
    
    def train_user_models(self, user_id: int, features: List[List[float]]):
        """Train all models for a specific user with calibration data."""
        try:
            logger.info(f"Starting model training for user {user_id}")
            
            # Prepare features
            scaled_features = self.prepare_features(features, user_id)
            
            if len(scaled_features) < 20:
                raise ValueError("Insufficient data for training (minimum 20 samples required)")
            
            # Initialize user model storage
            user_key = f"user_{user_id}"
            self.trained_models[user_key] = {}
            self.model_performance[user_key] = {}
            
            # Train Isolation Forest
            logger.info("Training Isolation Forest...")
            iso_forest = self.create_isolation_forest()
            iso_forest.fit(scaled_features)
            self.trained_models[user_key]['isolation_forest'] = iso_forest
            
            # Evaluate Isolation Forest
            anomaly_scores = iso_forest.decision_function(scaled_features)
            self.model_performance[user_key]['isolation_forest'] = {
                'anomaly_scores_mean': float(np.mean(anomaly_scores)),
                'anomaly_scores_std': float(np.std(anomaly_scores)),
                'training_samples': len(scaled_features)
            }
            
            # Train One-Class SVM
            logger.info("Training One-Class SVM...")
            oc_svm = self.create_one_class_svm()
            oc_svm.fit(scaled_features)
            self.trained_models[user_key]['one_class_svm'] = oc_svm
            
            # Evaluate One-Class SVM
            svm_scores = oc_svm.decision_function(scaled_features)
            self.model_performance[user_key]['one_class_svm'] = {
                'decision_scores_mean': float(np.mean(svm_scores)),
                'decision_scores_std': float(np.std(svm_scores)),
                'training_samples': len(scaled_features)
            }
            
            # Train incremental k-NN (LOF)
            logger.info("Training incremental k-NN...")
            knn_model = self.create_incremental_knn()
            knn_model.fit(scaled_features)
            self.trained_models[user_key]['incremental_knn'] = knn_model
            
            # Train neural networks if TensorFlow is available
            if self.is_tensorflow_available and len(scaled_features) >= 50:
                self._train_neural_networks(user_id, scaled_features, user_key)
            
            # Train Passive-Aggressive with synthetic labels (normal behavior = 1)
            logger.info("Training Passive-Aggressive Classifier...")
            pa_classifier = self.create_passive_aggressive_classifier()
            
            # Create synthetic training data with some noise as anomalies
            normal_labels = np.ones(len(scaled_features))
            noise_data = scaled_features + np.random.normal(0, 0.1, scaled_features.shape)
            noise_labels = np.zeros(len(noise_data))
            
            combined_features = np.vstack([scaled_features, noise_data])
            combined_labels = np.hstack([normal_labels, noise_labels])
            
            pa_classifier.fit(combined_features, combined_labels)
            self.trained_models[user_key]['passive_aggressive'] = pa_classifier
            
            # Performance evaluation
            pa_predictions = pa_classifier.predict(scaled_features)
            self.model_performance[user_key]['passive_aggressive'] = {
                'accuracy_on_normal': float(np.mean(pa_predictions)),
                'training_samples': len(combined_features)
            }
            
            logger.info(f"Model training completed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error training models for user {user_id}: {e}")
            raise
    
    def _train_neural_networks(self, user_id: int, scaled_features: np.ndarray, user_key: str):
        """Train neural network models (GRU and Autoencoder)."""
        try:
            # Train Autoencoder
            logger.info("Training Autoencoder...")
            autoencoder_models = self.create_autoencoder_model(scaled_features.shape[1])
            
            if autoencoder_models:
                autoencoder = autoencoder_models['autoencoder']
                
                # Training callbacks
                early_stopping = EarlyStopping(
                    monitor='loss', 
                    patience=10, 
                    restore_best_weights=True
                )
                
                # Train autoencoder
                history = autoencoder.fit(
                    scaled_features, scaled_features,
                    epochs=self.config['autoencoder']['epochs'],
                    batch_size=self.config['autoencoder']['batch_size'],
                    shuffle=True,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                self.trained_models[user_key]['autoencoder'] = autoencoder_models
                
                # Calculate reconstruction threshold
                reconstructed = autoencoder.predict(scaled_features, verbose=0)
                reconstruction_errors = np.mean(np.square(scaled_features - reconstructed), axis=1)
                threshold = np.percentile(reconstruction_errors, 95)
                
                self.model_performance[user_key]['autoencoder'] = {
                    'reconstruction_threshold': float(threshold),
                    'mean_reconstruction_error': float(np.mean(reconstruction_errors)),
                    'training_loss': float(min(history.history['loss']))
                }
            
            # Train GRU for sequential data
            if len(scaled_features) >= 100:  # Need more data for sequence modeling
                logger.info("Training GRU...")
                gru_model = self._train_gru_model(scaled_features, user_key)
                
                if gru_model:
                    self.trained_models[user_key]['gru'] = gru_model
            
        except Exception as e:
            logger.error(f"Error training neural networks: {e}")
    
    def _train_gru_model(self, scaled_features: np.ndarray, user_key: str) -> Optional[Any]:
        """Train GRU model for sequential behavioral analysis."""
        try:
            sequence_length = self.config['gru']['sequence_length']
            
            # Create sequences for GRU training
            sequences, labels = self._create_sequences(scaled_features, sequence_length)
            
            if len(sequences) < 20:
                logger.warning("Insufficient sequences for GRU training")
                return None
            
            # Create and train GRU model
            gru_model = self.create_gru_model((sequence_length, scaled_features.shape[1]))
            
            if gru_model:
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    sequences, labels, test_size=0.2, random_state=42
                )
                
                # Training callbacks
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                )
                
                # Train model
                history = gru_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.config['gru']['epochs'],
                    batch_size=self.config['gru']['batch_size'],
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Store performance metrics
                self.model_performance[user_key]['gru'] = {
                    'final_train_loss': float(history.history['loss'][-1]),
                    'final_val_loss': float(history.history['val_loss'][-1]),
                    'final_train_acc': float(history.history['accuracy'][-1]),
                    'final_val_acc': float(history.history['val_accuracy'][-1])
                }
                
                return gru_model
            
        except Exception as e:
            logger.error(f"Error training GRU model: {e}")
            return None
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for GRU training with synthetic anomalies."""
        sequences = []
        labels = []
        
        # Create normal sequences
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
            labels.append(1)  # Normal behavior
        
        # Create some anomalous sequences by adding noise
        num_anomalies = len(sequences) // 10  # 10% anomalies
        for _ in range(num_anomalies):
            # Select random normal sequence and add noise
            idx = np.random.randint(0, len(sequences))
            anomalous_seq = sequences[idx].copy()
            
            # Add random noise to simulate anomalous behavior
            noise_strength = 0.3
            anomalous_seq += np.random.normal(0, noise_strength, anomalous_seq.shape)
            
            sequences.append(anomalous_seq)
            labels.append(0)  # Anomalous behavior
        
        return np.array(sequences), np.array(labels)
    
    def predict_anomaly(self, user_id: int, features: List[float]) -> Dict[str, Any]:
        """Predict if current behavior is anomalous using ensemble of models."""
        try:
            user_key = f"user_{user_id}"
            
            if user_key not in self.trained_models:
                raise ValueError(f"No trained models found for user {user_id}")
            
            # Prepare features
            feature_array = np.array(features).reshape(1, -1)
            
            # Scale features using user-specific scaler
            if user_key in self.feature_scalers:
                scaled_features = self.feature_scalers[user_key].transform(feature_array)
            else:
                scaled_features = feature_array
            
            models = self.trained_models[user_key]
            predictions = {}
            scores = {}
            
            # Isolation Forest prediction
            if 'isolation_forest' in models:
                iso_pred = models['isolation_forest'].predict(scaled_features)[0]
                iso_score = models['isolation_forest'].decision_function(scaled_features)[0]
                predictions['isolation_forest'] = iso_pred
                scores['isolation_forest'] = float(iso_score)
            
            # One-Class SVM prediction
            if 'one_class_svm' in models:
                svm_pred = models['one_class_svm'].predict(scaled_features)[0]
                svm_score = models['one_class_svm'].decision_function(scaled_features)[0]
                predictions['one_class_svm'] = svm_pred
                scores['one_class_svm'] = float(svm_score)
            
            # Incremental k-NN prediction
            if 'incremental_knn' in models:
                knn_pred = models['incremental_knn'].predict(scaled_features)[0]
                knn_score = models['incremental_knn'].decision_function(scaled_features)[0]
                predictions['incremental_knn'] = knn_pred
                scores['incremental_knn'] = float(knn_score)
            
            # Passive-Aggressive prediction
            if 'passive_aggressive' in models:
                pa_pred = models['passive_aggressive'].predict(scaled_features)[0]
                pa_score = models['passive_aggressive'].decision_function(scaled_features)[0]
                predictions['passive_aggressive'] = pa_pred
                scores['passive_aggressive'] = float(pa_score)
            
            # Neural network predictions
            if self.is_tensorflow_available:
                # Autoencoder prediction
                if 'autoencoder' in models:
                    autoencoder = models['autoencoder']['autoencoder']
                    reconstructed = autoencoder.predict(scaled_features, verbose=0)
                    reconstruction_error = np.mean(np.square(scaled_features - reconstructed))
                    
                    threshold = self.model_performance[user_key]['autoencoder']['reconstruction_threshold']
                    ae_pred = -1 if reconstruction_error > threshold else 1
                    
                    predictions['autoencoder'] = ae_pred
                    scores['autoencoder'] = float(reconstruction_error)
            
            # Ensemble decision
            ensemble_result = self._make_ensemble_decision(predictions, scores, user_key)
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error predicting anomaly for user {user_id}: {e}")
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'severity': 'unknown',
                'individual_predictions': {},
                'error': str(e)
            }
    
    def _make_ensemble_decision(self, predictions: Dict[str, int], 
                               scores: Dict[str, float], user_key: str) -> Dict[str, Any]:
        """Make final ensemble decision based on individual model predictions."""
        
        # Convert predictions to anomaly indicators (1 = normal, -1 = anomaly)
        anomaly_votes = 0
        total_votes = 0
        
        weighted_score = 0.0
        total_weight = 0.0
        
        # Model weights (can be adjusted based on performance)
        model_weights = {
            'isolation_forest': 0.25,
            'one_class_svm': 0.20,
            'incremental_knn': 0.15,
            'passive_aggressive': 0.20,
            'autoencoder': 0.15,
            'gru': 0.05
        }
        
        for model_name, prediction in predictions.items():
            if prediction == -1:  # Anomaly
                anomaly_votes += 1
            total_votes += 1
            
            # Weighted scoring
            if model_name in model_weights and model_name in scores:
                weight = model_weights[model_name]
                # Normalize scores to [0, 1] range where higher = more anomalous
                normalized_score = self._normalize_score(model_name, scores[model_name], user_key)
                weighted_score += weight * normalized_score
                total_weight += weight
        
        # Calculate ensemble metrics
        if total_votes > 0:
            anomaly_ratio = anomaly_votes / total_votes
            confidence = weighted_score / total_weight if total_weight > 0 else 0.0
        else:
            anomaly_ratio = 0.0
            confidence = 0.0
        
        # Determine anomaly status and severity
        is_anomaly = anomaly_ratio >= 0.3  # At least 30% of models agree
        
        if confidence >= 0.8:
            severity = 'high'
        elif confidence >= 0.6:
            severity = 'medium'
        elif confidence >= 0.3:
            severity = 'low'
        else:
            severity = 'normal'
            is_anomaly = False
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': float(confidence),
            'severity': severity,
            'anomaly_ratio': float(anomaly_ratio),
            'individual_predictions': predictions,
            'individual_scores': scores,
            'total_votes': total_votes,
            'anomaly_votes': anomaly_votes
        }
    
    def _normalize_score(self, model_name: str, score: float, user_key: str) -> float:
        """Normalize model scores to [0, 1] range where 1 = highly anomalous."""
        
        try:
            if user_key not in self.model_performance:
                return 0.5  # Default neutral score
            
            performance = self.model_performance[user_key]
            
            if model_name == 'isolation_forest':
                # Isolation Forest: negative scores = anomalies
                if model_name in performance:
                    mean_score = performance[model_name].get('anomaly_scores_mean', 0)
                    std_score = performance[model_name].get('anomaly_scores_std', 1)
                    normalized = max(0, min(1, (mean_score - score) / (2 * std_score) + 0.5))
                    return normalized
                
            elif model_name == 'one_class_svm':
                # One-Class SVM: negative scores = anomalies
                if model_name in performance:
                    mean_score = performance[model_name].get('decision_scores_mean', 0)
                    std_score = performance[model_name].get('decision_scores_std', 1)
                    normalized = max(0, min(1, (mean_score - score) / (2 * std_score) + 0.5))
                    return normalized
                
            elif model_name == 'autoencoder':
                # Autoencoder: higher reconstruction error = anomaly
                if model_name in performance:
                    threshold = performance[model_name].get('reconstruction_threshold', 0.1)
                    normalized = min(1, score / threshold)
                    return normalized
            
            # Default normalization for other models
            return max(0, min(1, abs(score)))
            
        except Exception as e:
            logger.warning(f"Error normalizing score for {model_name}: {e}")
            return 0.5
    
    def update_user_models(self, user_id: int, new_features: List[List[float]]):
        """Update existing models with new behavioral data."""
        try:
            user_key = f"user_{user_id}"
            
            if user_key not in self.trained_models:
                logger.warning(f"No existing models for user {user_id}. Training new models.")
                self.train_user_models(user_id, new_features)
                return
            
            # Prepare new features
            scaled_features = self.prepare_features(new_features, user_id)
            
            models = self.trained_models[user_key]
            
            # Update Passive-Aggressive classifier (incremental learning)
            if 'passive_aggressive' in models:
                # Assume new data is normal behavior
                labels = np.ones(len(scaled_features))
                models['passive_aggressive'].partial_fit(scaled_features, labels)
            
            # For other models, retrain with combined data
            # In production, you might want to implement true incremental learning
            logger.info(f"Models updated for user {user_id} with {len(new_features)} new samples")
            
        except Exception as e:
            logger.error(f"Error updating models for user {user_id}: {e}")
    
    def save_user_models(self, user_id: int, filepath: str):
        """Save trained models to disk."""
        try:
            user_key = f"user_{user_id}"
            
            if user_key not in self.trained_models:
                raise ValueError(f"No trained models found for user {user_id}")
            
            # Prepare data for saving
            save_data = {
                'models': {},
                'scaler': self.feature_scalers.get(user_key),
                'performance': self.model_performance.get(user_key, {}),
                'user_id': user_id
            }
            
            # Save non-neural network models
            for model_name, model in self.trained_models[user_key].items():
                if model_name not in ['autoencoder', 'gru']:
                    save_data['models'][model_name] = model
            
            # Save to pickle
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Save neural network models separately if TensorFlow is available
            if self.is_tensorflow_available:
                tf_models_path = filepath.replace('.pkl', '_tf_models')
                self._save_tensorflow_models(user_key, tf_models_path)
            
            logger.info(f"Models saved for user {user_id} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models for user {user_id}: {e}")
            raise
    
    def load_user_models(self, user_id: int, filepath: str):
        """Load trained models from disk."""
        try:
            user_key = f"user_{user_id}"
            
            # Load main models
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.trained_models[user_key] = save_data['models']
            self.feature_scalers[user_key] = save_data['scaler']
            self.model_performance[user_key] = save_data['performance']
            
            # Load TensorFlow models if available
            if self.is_tensorflow_available:
                tf_models_path = filepath.replace('.pkl', '_tf_models')
                self._load_tensorflow_models(user_key, tf_models_path)
            
            logger.info(f"Models loaded for user {user_id} from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models for user {user_id}: {e}")
            raise
    
    def _save_tensorflow_models(self, user_key: str, base_path: str):
        """Save TensorFlow models separately."""
        try:
            if user_key in self.trained_models:
                models = self.trained_models[user_key]
                
                if 'autoencoder' in models:
                    autoencoder_path = f"{base_path}_autoencoder"
                    models['autoencoder']['autoencoder'].save(autoencoder_path)
                
                if 'gru' in models:
                    gru_path = f"{base_path}_gru"
                    models['gru'].save(gru_path)
                    
        except Exception as e:
            logger.error(f"Error saving TensorFlow models: {e}")
    
    def _load_tensorflow_models(self, user_key: str, base_path: str):
        """Load TensorFlow models separately."""
        try:
            # Load autoencoder
            try:
                autoencoder_path = f"{base_path}_autoencoder"
                autoencoder = tf.keras.models.load_model(autoencoder_path)
                
                # Recreate encoder from autoencoder
                encoder_input = autoencoder.input
                encoder_output = autoencoder.layers[-3].output  # Get encoding layer
                encoder = Model(encoder_input, encoder_output)
                
                self.trained_models[user_key]['autoencoder'] = {
                    'autoencoder': autoencoder,
                    'encoder': encoder
                }
                
            except:
                pass  # Model file doesn't exist
            
            # Load GRU
            try:
                gru_path = f"{base_path}_gru"
                gru_model = tf.keras.models.load_model(gru_path)
                self.trained_models[user_key]['gru'] = gru_model
            except:
                pass  # Model file doesn't exist
                
        except Exception as e:
            logger.error(f"Error loading TensorFlow models: {e}")
    
    def get_model_info(self, user_id: int) -> Dict[str, Any]:
        """Get information about trained models for a user."""
        user_key = f"user_{user_id}"
        
        if user_key not in self.trained_models:
            return {'error': f'No models found for user {user_id}'}
        
        info = {
            'user_id': user_id,
            'models_available': list(self.trained_models[user_key].keys()),
            'performance_metrics': self.model_performance.get(user_key, {}),
            'scaler_fitted': user_key in self.feature_scalers,
            'tensorflow_available': self.is_tensorflow_available
        }
        
        return info