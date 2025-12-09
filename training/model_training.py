#model Training for CSP Solver

import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

#add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.trace_generator import FeatureVector



@dataclass
class TrainingConfig:
    #configuration for model training
    #data parameters
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    
    #model parameters
    hidden_layers: List[int] = None
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    #regularization
    dropout_rate: float = 0.2
    l2_regularization: float = 0.001
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 32, 16]


class SimpleNeuralNetwork:
    #simple neural network using only numpy
    #for use when deep learning libraries are unavailable
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 learning_rate: float = 0.001):
        self.layers = []
        self.learning_rate = learning_rate
        
        #initialize weights
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            #xavier initialization
            limit = np.sqrt(6 / (sizes[i] + sizes[i+1]))
            W = np.random.uniform(-limit, limit, (sizes[i], sizes[i+1]))
            b = np.zeros((1, sizes[i+1]))
            self.layers.append({'W': W, 'b': b})
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)  #prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        #forward pass through the network
        activations = [X]
        current = X
        
        for i, layer in enumerate(self.layers[:-1]):
            z = current @ layer['W'] + layer['b']
            current = self.relu(z)
            activations.append(current)
        
        #output layer with sigmoid
        z = current @ self.layers[-1]['W'] + self.layers[-1]['b']
        output = self.sigmoid(z)
        activations.append(output)
        
        return output, activations
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray]):
        #backward pass with gradient computation
        m = X.shape[0]
        gradients = []
        
        #output layer gradient
        output = activations[-1]
        delta = output - y.reshape(-1, 1)
        
        for i in range(len(self.layers) - 1, -1, -1):
            dW = activations[i].T @ delta / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients.insert(0, {'dW': dW, 'db': db})
            
            if i > 0:
                delta = (delta @ self.layers[i]['W'].T) * self.relu_derivative(activations[i])
        
        #update weights
        for i, (layer, grad) in enumerate(zip(self.layers, gradients)):
            layer['W'] -= self.learning_rate * grad['dW']
            layer['b'] -= self.learning_rate * grad['db']
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        #single training step, returns loss
        output, activations = self.forward(X)
        
        #binary cross-entropy loss
        eps = 1e-7
        loss = -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))
        
        self.backward(X, y, activations)
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        #make predictions
        output, _ = self.forward(X)
        return output.flatten()
    
    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        #make binary predictions
        return (self.predict(X) >= threshold).astype(int)


class GradientBoostingSimple:
    #simple gradient boosting implementation using decision stumps
    #for use when scikit-learn is unavailable
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = 0.5
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        #fit the model
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.initial_prediction)
        
        for i in range(self.n_estimators):
            #compute residuals (gradient)
            residuals = y - self._sigmoid(predictions)
            
            #fit a simple decision stump
            tree = self._fit_stump(X, residuals)
            self.trees.append(tree)
            
            #update predictions
            predictions += self.learning_rate * self._predict_stump(X, tree)
        
        return self
    
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def _fit_stump(self, X: np.ndarray, residuals: np.ndarray) -> Dict:
        #fit a decision stump (single split)
        n_samples, n_features = X.shape
        best_split = None
        best_mse = float('inf')
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds[::max(1, len(thresholds) // 10)]:  #sample thresholds
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() < 1 or right_mask.sum() < 1:
                    continue
                
                left_pred = residuals[left_mask].mean()
                right_pred = residuals[right_mask].mean()
                
                predictions = np.where(left_mask, left_pred, right_pred)
                mse = np.mean((residuals - predictions) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'left_value': left_pred,
                        'right_value': right_pred,
                    }
        
        return best_split or {'feature': 0, 'threshold': 0, 'left_value': 0, 'right_value': 0}
    
    def _predict_stump(self, X: np.ndarray, tree: Dict) -> np.ndarray:
        #predict using a decision stump
        left_mask = X[:, tree['feature']] <= tree['threshold']
        return np.where(left_mask, tree['left_value'], tree['right_value'])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        #predict probabilities
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.trees:
            predictions += self.learning_rate * self._predict_stump(X, tree)
        
        return self._sigmoid(predictions)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        #make binary predictions
        return (self.predict_proba(X) >= threshold).astype(int)


class VariableSelectionModel:
    #model to predict the best variable to select next
    #learns from traces which variable selections led to solutions
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model = None
        self.feature_means = None
        self.feature_stds = None
        
    def prepare_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        #porepare and normalize training data
        #normalize features
        self.feature_means = features.mean(axis=0)
        self.feature_stds = features.std(axis=0) + 1e-8
        features_norm = (features - self.feature_means) / self.feature_stds
        
        #split data
        n = len(features)
        indices = np.random.permutation(n)
        
        train_end = int(n * self.config.train_split)
        val_end = int(n * (self.config.train_split + self.config.validation_split))
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        
        X_train = features_norm[train_idx]
        y_train = labels[train_idx]
        X_val = features_norm[val_idx]
        y_val = labels[val_idx]
        
        return X_train, y_train, X_val, y_val
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, List[float]]:
        #train the variable selection model
        
        #args:
            #features: Feature matrix (num_samples, num_features)
            #labels: Binary labels (1 if decision led to solution)
            
        #returns:
            #raining history with loss and accuracy per epoch
        X_train, y_train, X_val, y_val = self.prepare_data(features, labels)
        
        #initialize model
        input_size = X_train.shape[1]
        self.model = SimpleNeuralNetwork(
            input_size=input_size,
            hidden_sizes=self.config.hidden_layers,
            output_size=1,
            learning_rate=self.config.learning_rate
        )
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            #training
            train_loss = self.model.train_step(X_train, y_train)
            train_pred = self.model.predict_binary(X_train)
            train_acc = (train_pred == y_train).mean()
            
            #validation
            val_pred_proba = self.model.predict(X_val)
            eps = 1e-7
            val_loss = -np.mean(y_val * np.log(val_pred_proba + eps) + 
                               (1 - y_val) * np.log(1 - val_pred_proba + eps))
            val_pred = (val_pred_proba >= 0.5).astype(int)
            val_acc = (val_pred == y_val).mean()
            
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            #early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        return history
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        #predict probability that each decision leads to solution
        if self.feature_means is not None:
            features = (features - self.feature_means) / self.feature_stds
        return self.model.predict(features)
    
    def score_variable(self, state_features: np.ndarray) -> float:
        #score a variable selection decision
        return self.predict(state_features.reshape(1, -1))[0]
    
    def save(self, filepath: str):
        #save model to file
        data = {
            'model_layers': [(l['W'].tolist(), l['b'].tolist()) for l in self.model.layers],
            'feature_means': self.feature_means.tolist() if self.feature_means is not None else None,
            'feature_stds': self.feature_stds.tolist() if self.feature_stds is not None else None,
            'config': self.config.__dict__,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        #load model from file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.feature_means = np.array(data['feature_means']) if data['feature_means'] else None
        self.feature_stds = np.array(data['feature_stds']) if data['feature_stds'] else None
        
        #reconstruct model
        if data['model_layers']:
            first_layer = data['model_layers'][0]
            input_size = len(first_layer[0])
            output_size = len(data['model_layers'][-1][0][0])
            hidden_sizes = [len(l[0][0]) for l in data['model_layers'][:-1]]
            
            self.model = SimpleNeuralNetwork(input_size, hidden_sizes, output_size)
            for i, (W, b) in enumerate(data['model_layers']):
                self.model.layers[i]['W'] = np.array(W)
                self.model.layers[i]['b'] = np.array(b)


class ValueOrderingModel:
    #model to predict the best value ordering for a variable
    #uses gradient boosting for interpretability
    
    def __init__(self, n_estimators: int = 50):
        self.model = GradientBoostingSimple(n_estimators=n_estimators)
        self.feature_means = None
        self.feature_stds = None
        
    def train(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        #train the value ordering model
        #normalize
        self.feature_means = features.mean(axis=0)
        self.feature_stds = features.std(axis=0) + 1e-8
        features_norm = (features - self.feature_means) / self.feature_stds
        
        #train
        self.model.fit(features_norm, labels)
        
        #evaluate
        predictions = self.model.predict(features_norm)
        accuracy = (predictions == labels).mean()
        
        return {'accuracy': accuracy}
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        #predict value quality scores
        if self.feature_means is not None:
            features = (features - self.feature_means) / self.feature_stds
        return self.model.predict_proba(features)


class FailurePredictionModel:
    #model to predict if current search state will lead to failure
    #can be used for early pruning of unpromising branches
    
    def __init__(self, threshold: float = 0.7):
        self.model = None
        self.threshold = threshold
        self.feature_means = None
        self.feature_stds = None
        
    def train(self, features: np.ndarray, labels: np.ndarray, 
              steps_to_backtrack: np.ndarray) -> Dict:
        #train failure prediction model
        
        #args:
            #features: State features
            #labels: Whether decision led to solution
            #steps_to_backtrack: Steps until backtracking (0 = no backtrack)
        #create failure labels (1 = will fail soon)
        failure_labels = ((labels == 0) & (steps_to_backtrack < 5)).astype(int)
        
        #normalize
        self.feature_means = features.mean(axis=0)
        self.feature_stds = features.std(axis=0) + 1e-8
        features_norm = (features - self.feature_means) / self.feature_stds
        
        #train simple model
        self.model = GradientBoostingSimple(n_estimators=30, learning_rate=0.1)
        self.model.fit(features_norm, failure_labels)
        
        #evaluate
        predictions = self.model.predict(features_norm)
        accuracy = (predictions == failure_labels).mean()
        
        return {'accuracy': accuracy}
    
    def should_prune(self, features: np.ndarray) -> bool:
        #determine if current branch should be pruned
        if self.model is None:
            return False
        
        if self.feature_means is not None:
            features = (features - self.feature_means) / self.feature_stds
        
        failure_prob = self.model.predict_proba(features.reshape(1, -1))[0]
        return failure_prob >= self.threshold


class ModelTrainer:
    #main training pipeline for all models
    
    def __init__(self, trace_dir: str, model_dir: str = './models'):
        self.trace_dir = trace_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        #load pre-generated feature matrices
        features_path = os.path.join(self.trace_dir, 'features.npy')
        labels_path = os.path.join(self.trace_dir, 'labels.npy')
        
        if os.path.exists(features_path) and os.path.exists(labels_path):
            X = np.load(features_path)
            y = np.load(labels_path)
            
            #generate synthetic backtrack data if not available
            backtrack_steps = np.zeros(len(y))
            return X, y, backtrack_steps
        
        #load from individual trace files
        return self._load_from_traces()
    
    def _load_from_traces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        #load training data from individual trace files
        import glob
        
        trace_files = glob.glob(os.path.join(self.trace_dir, '*_trace.json'))
        
        all_features = []
        all_labels = []
        all_backtrack = []
        
        for trace_file in trace_files:
            with open(trace_file, 'r') as f:
                data = json.load(f)
            
            for entry in data.get('trace', []):
                features = entry.get('features', {})
                
                #convert to feature vector
                fv = FeatureVector(
                    depth=features.get('depth', 0),
                    num_assigned=features.get('num_assigned', 0),
                    num_unassigned=features.get('num_unassigned', 0),
                    assignment_ratio=features.get('assignment_ratio', 0),
                    min_domain_size=features.get('min_domain_size', 0),
                    max_domain_size=features.get('max_domain_size', 0),
                    avg_domain_size=features.get('avg_domain_size', 0),
                    std_domain_size=features.get('std_domain_size', 0),
                    num_singleton_domains=features.get('num_singleton_domains', 0),
                    chosen_var_domain_size=features.get('chosen_var_domain_size', 0),
                    chosen_var_constraint_count=features.get('chosen_var_constraint_count', 0),
                    chosen_var_category_assigned_count=features.get('chosen_var_category_assigned_count', 0),
                    total_constraints=features.get('total_constraints', 0),
                    satisfied_constraints=features.get('satisfied_constraints', 0),
                    constraint_satisfaction_ratio=features.get('constraint_satisfaction_ratio', 0),
                    chosen_value=features.get('chosen_value', 0),
                    chosen_value_eliminates=features.get('chosen_value_eliminates', 0),
                )
                
                all_features.append(fv.to_array())
                all_labels.append(1 if features.get('led_to_solution', False) else 0)
                all_backtrack.append(features.get('backtrack_after', 0))
        
        return (
            np.array(all_features) if all_features else np.zeros((0, 17)),
            np.array(all_labels) if all_labels else np.zeros(0),
            np.array(all_backtrack) if all_backtrack else np.zeros(0)
        )
    
    def train_all_models(self) -> Dict[str, Any]:
        #train all models and save them
        print("Loading training data...")
        X, y, backtrack = self.load_training_data()
        
        if len(X) == 0:
            print("No training data found!")
            return {}
        
        print(f"Training data shape: {X.shape}")
        print(f"Positive samples: {y.sum()} / {len(y)}")
        
        results = {}
        
        #train variable selection model
        print("\n" + "="*50)
        print("Training Variable Selection Model")
        print("="*50)
        var_model = VariableSelectionModel()
        var_history = var_model.train(X, y)
        var_model.save(os.path.join(self.model_dir, 'variable_selection.json'))
        results['variable_selection'] = {
            'final_train_accuracy': var_history['train_accuracy'][-1],
            'final_val_accuracy': var_history['val_accuracy'][-1],
        }
        
        #train value ordering model
        print("\n" + "="*50)
        print("Training Value Ordering Model")
        print("="*50)
        val_model = ValueOrderingModel()
        val_results = val_model.train(X, y)
        results['value_ordering'] = val_results
        
        #train failure prediction model
        print("\n" + "="*50)
        print("Training Failure Prediction Model")
        print("="*50)
        fail_model = FailurePredictionModel()
        fail_results = fail_model.train(X, y, backtrack)
        results['failure_prediction'] = fail_results
        
        #save results
        with open(os.path.join(self.model_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        print(f"Results saved to {self.model_dir}")
        
        return results


def create_synthetic_training_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    #create synthetic training data for testing
    #it is good when real traces are not available
    np.random.seed(42)
    
    X = np.random.randn(num_samples, 17)
    
    #create synthetic labels based on feature patterns
    #good decisions tend to have:
    # + higher assignment ratio (feature 3)
    # + lower min domain size (feature 4)
    # + higher constraint satisfaction (feature 14)
    
    score = (
        0.3 * X[:, 3] +  #assignment_ratio
        -0.2 * X[:, 4] +  #min_domain_size (lower is better for MRV)
        0.4 * X[:, 14] +  #constraint_satisfaction_ratio
        0.1 * np.random.randn(num_samples)  #noise
    )
    
    y = (score > np.median(score)).astype(int)
    
    return X, y


if __name__ == "__main__":
    print("Testing Model Training")
    print("=" * 50)
    
    #create synthetic data for testing
    print("\nCreating synthetic training data...")
    X, y = create_synthetic_training_data(1000)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {y.sum()} positive / {len(y)} total")
    
    #test variable selection model
    print("\n" + "-"*40)
    print("Training Variable Selection Model")
    print("-"*40)
    
    var_model = VariableSelectionModel(TrainingConfig(num_epochs=50))
    history = var_model.train(X, y)
    
    print(f"\nFinal train accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    
    #test value ordering model
    print("\n" + "-"*40)
    print("Training Value Ordering Model")
    print("-"*40)
    
    val_model = ValueOrderingModel(n_estimators=30)
    val_results = val_model.train(X, y)
    print(f"Accuracy: {val_results['accuracy']:.4f}")
    
    #save models
    os.makedirs('./models', exist_ok=True)
    var_model.save('./models/variable_selection.json')
    
    print("\nModels saved to models/")
