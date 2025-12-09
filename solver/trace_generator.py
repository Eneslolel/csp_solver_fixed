#trace Generator for CSP Solver

import json
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np

#add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_parser import CSPPuzzle, Constraint, ConstraintType
from solver.csp_solver import CSPSolver, SolverStats, SearchState

@dataclass
class FeatureVector:
    #feature vector representing a decision point in CSP search
    #used for training ML models to predict good variable/value choices

    #state features
    depth: int = 0
    num_assigned: int = 0
    num_unassigned: int = 0
    assignment_ratio: float = 0.0
    
    #domain features
    min_domain_size: int = 0
    max_domain_size: int = 0
    avg_domain_size: float = 0.0
    std_domain_size: float = 0.0
    num_singleton_domains: int = 0
    
    #variable features (for the chosen variable)
    chosen_var_domain_size: int = 0
    chosen_var_constraint_count: int = 0
    chosen_var_category_assigned_count: int = 0
    
    #constraint features
    total_constraints: int = 0
    satisfied_constraints: int = 0
    constraint_satisfaction_ratio: float = 0.0
    
    #value features (for the chosen value)
    chosen_value: int = 0
    chosen_value_eliminates: int = 0
    
    #outcome (for training)
    led_to_solution: bool = False
    backtrack_after: int = 0  #steps until backtrack from this decision
    
    def to_array(self) -> np.ndarray:
        #convert to numpy array for ML models
        return np.array([
            self.depth,
            self.num_assigned,
            self.num_unassigned,
            self.assignment_ratio,
            self.min_domain_size,
            self.max_domain_size,
            self.avg_domain_size,
            self.std_domain_size,
            self.num_singleton_domains,
            self.chosen_var_domain_size,
            self.chosen_var_constraint_count,
            self.chosen_var_category_assigned_count,
            self.total_constraints,
            self.satisfied_constraints,
            self.constraint_satisfaction_ratio,
            self.chosen_value,
            self.chosen_value_eliminates,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        #get names of all features
        return [
            'depth', 'num_assigned', 'num_unassigned', 'assignment_ratio',
            'min_domain_size', 'max_domain_size', 'avg_domain_size', 'std_domain_size',
            'num_singleton_domains', 'chosen_var_domain_size', 'chosen_var_constraint_count',
            'chosen_var_category_assigned_count', 'total_constraints', 'satisfied_constraints',
            'constraint_satisfaction_ratio', 'chosen_value', 'chosen_value_eliminates',
        ]


@dataclass
class TraceEntry:
    #a single entry in the search trace
    step_number: int
    depth: int
    chosen_variable: Optional[Tuple[str, str]]
    chosen_value: Optional[int]
    features: FeatureVector
    domain_snapshot: Dict[str, int]  # var -> domain size
    action_type: str  # 'assign', 'backtrack', 'solution', 'failure'
    
    def to_dict(self) -> Dict:
        #convert to dictionary for JSON serialization
        return {
            'step_number': self.step_number,
            'depth': self.depth,
            'chosen_variable': list(self.chosen_variable) if self.chosen_variable else None,
            'chosen_value': self.chosen_value,
            'features': asdict(self.features),
            'domain_snapshot': self.domain_snapshot,
            'action_type': self.action_type,
        }


class TraceGenerator:
    #generates detailed traces from CSP solving
    
    def __init__(self, puzzle: CSPPuzzle):
        self.puzzle = puzzle
        self.traces: List[TraceEntry] = []
        self.solver = None
        
    def generate(self, solver: CSPSolver) -> List[TraceEntry]:
        #generate trace from a solver run
        #must be called after solver.solve() completes
        self.solver = solver
        self.traces = []
        
        search_trace = solver.trace
        stats = solver.get_stats()
        
        for i, state in enumerate(search_trace):
            features = self._extract_features(state, stats)
            
            #determine action type
            action_type = 'assign'
            if i == len(search_trace) - 1:
                action_type = 'solution' if stats['solved'] else 'failure'
            elif i + 1 < len(search_trace) and search_trace[i + 1].depth < state.depth:
                action_type = 'backtrack'
            
            #calculate if this decision led to solution
            features.led_to_solution = stats['solved'] and self._leads_to_solution(i, search_trace)
            features.backtrack_after = self._steps_until_backtrack(i, search_trace)
            
            entry = TraceEntry(
                step_number=state.step_number,
                depth=state.depth,
                chosen_variable=state.chosen_variable,
                chosen_value=state.chosen_value,
                features=features,
                domain_snapshot={str(k): v for k, v in state.domain_sizes.items()},
                action_type=action_type,
            )
            
            self.traces.append(entry)
        
        return self.traces
    
    def _extract_features(self, state: SearchState, stats: Dict) -> FeatureVector:
        #extract feature vector from a search state
        features = FeatureVector()
        
        #state features
        features.depth = state.depth
        features.num_assigned = len(state.assignments)
        features.num_unassigned = len(state.domains) - len(state.assignments)
        features.assignment_ratio = features.num_assigned / len(state.domains) if state.domains else 0
        
        #domain features
        unassigned_domains = [
            len(d) for v, d in state.domains.items() 
            if v not in state.assignments
        ]
        
        if unassigned_domains:
            features.min_domain_size = min(unassigned_domains)
            features.max_domain_size = max(unassigned_domains)
            features.avg_domain_size = np.mean(unassigned_domains)
            features.std_domain_size = np.std(unassigned_domains)
            features.num_singleton_domains = sum(1 for d in unassigned_domains if d == 1)
        
        #variable features
        if state.chosen_variable:
            features.chosen_var_domain_size = state.domain_sizes.get(state.chosen_variable, 0)
            features.chosen_var_constraint_count = len(
                self.solver.checker.get_related_constraints(state.chosen_variable)
            )
            
            #count how many vars in same category are assigned
            cat = state.chosen_variable[0]
            features.chosen_var_category_assigned_count = sum(
                1 for v in state.assignments if v[0] == cat
            )
        
        #constraint features
        features.total_constraints = len(self.puzzle.constraints)
        features.satisfied_constraints = sum(
            1 for c in self.puzzle.constraints 
            if self.solver.checker.check_constraint(c, state.assignments)
        )
        features.constraint_satisfaction_ratio = (
            features.satisfied_constraints / features.total_constraints 
            if features.total_constraints > 0 else 1.0
        )
        
        #value features
        if state.chosen_value is not None:
            features.chosen_value = state.chosen_value
            features.chosen_value_eliminates = state.pruned_values
        
        return features
    
    def _leads_to_solution(self, index: int, trace: List[SearchState]) -> bool:
        #check if a decision at given index was on the path to solution
        if not trace:
            return False
        
        #a decision leads to solution if no backtracking below its depth occurs after it
        state = trace[index]
        for future_state in trace[index + 1:]:
            if future_state.depth < state.depth:
                return False
        return True
    
    def _steps_until_backtrack(self, index: int, trace: List[SearchState]) -> int:
        #count steps until backtracking from this decision point
        if index >= len(trace) - 1:
            return 0
        
        state = trace[index]
        for i, future_state in enumerate(trace[index + 1:]):
            if future_state.depth < state.depth:
                return i + 1
        return len(trace) - index - 1
    
    def save_trace(self, filepath: str):
        #save trace to JSON file
        data = {
            'puzzle_id': self.puzzle.puzzle_id,
            'puzzle_size': self.puzzle.size,
            'num_houses': self.puzzle.num_houses,
            'num_categories': len(self.puzzle.categories),
            'num_constraints': len(self.puzzle.constraints),
            'solver_stats': self.solver.get_stats() if self.solver else {},
            'trace': [entry.to_dict() for entry in self.traces],
            'generated_at': datetime.now().isoformat(),
        }
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        #get feature matrix and labels for ML training
        
        #returns:
           # X: Feature matrix (num_steps, num_features)
           # y: Labels (1 if led to solution, 0 otherwise)
        X = np.array([entry.features.to_array() for entry in self.traces])
        y = np.array([1 if entry.features.led_to_solution else 0 for entry in self.traces])
        return X, y


class BatchTraceGenerator:
    #generate traces for multiple puzzles
    
    def __init__(self, output_dir: str = './traces'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_batch(self, puzzles: List[CSPPuzzle], 
                      solver_factory=None) -> Dict[str, Any]:
        #generate traces for a batch of puzzles
        
        #args:
            #puzzles: List of CSPPuzzle objects
            #solver_factory: Factory function to create solvers
            
        #returns:
            #statistics about the batch generation
        from solver.csp_solver import SolverFactory
        
        if solver_factory is None:
            solver_factory = SolverFactory.create_full
        
        all_features = []
        all_labels = []
        stats = {
            'total_puzzles': len(puzzles),
            'solved': 0,
            'failed': 0,
            'total_steps': 0,
            'avg_steps': 0,
            'traces_generated': 0,
        }
        
        for i, puzzle in enumerate(puzzles):
            print(f"Processing puzzle {i+1}/{len(puzzles)}: {puzzle.puzzle_id}")
            
            try:
                solver = solver_factory(puzzle)
                solution = solver.solve()
                
                if solution is not None:
                    stats['solved'] += 1
                else:
                    stats['failed'] += 1
                
                #generate trace
                generator = TraceGenerator(puzzle)
                traces = generator.generate(solver)
                
                #save individual trace
                trace_path = os.path.join(self.output_dir, f"{puzzle.puzzle_id}_trace.json")
                generator.save_trace(trace_path)
                
                #collect features
                X, y = generator.get_feature_matrix()
                all_features.append(X)
                all_labels.append(y)
                
                stats['total_steps'] += solver.get_stats()['total_steps']
                stats['traces_generated'] += len(traces)
                
            except Exception as e:
                print(f"  Error: {e}")
                stats['failed'] += 1
        
        #combine all features
        if all_features:
            X_combined = np.vstack(all_features)
            y_combined = np.concatenate(all_labels)
            
            #save combined dataset
            np.save(os.path.join(self.output_dir, 'features.npy'), X_combined)
            np.save(os.path.join(self.output_dir, 'labels.npy'), y_combined)
        
        stats['avg_steps'] = stats['total_steps'] / max(stats['total_puzzles'], 1)
        
        #save batch statistics
        with open(os.path.join(self.output_dir, 'batch_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats


def analyze_traces(trace_dir: str) -> Dict[str, Any]:
    #analyze generated traces to extract insights.
    import glob
    
    trace_files = glob.glob(os.path.join(trace_dir, '*_trace.json'))
    
    analysis = {
        'num_traces': len(trace_files),
        'depth_distribution': {},
        'avg_branching_factor': 0,
        'backtrack_rates': [],
        'constraint_types': {},
    }
    
    total_branching = 0
    total_decisions = 0
    
    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            data = json.load(f)
        
        for entry in data.get('trace', []):
            depth = entry['depth']
            analysis['depth_distribution'][depth] = analysis['depth_distribution'].get(depth, 0) + 1
            
            if entry.get('features', {}).get('chosen_var_domain_size', 0) > 0:
                total_branching += entry['features']['chosen_var_domain_size']
                total_decisions += 1
    
    if total_decisions > 0:
        analysis['avg_branching_factor'] = total_branching / total_decisions
    
    return analysis


if __name__ == "__main__":
    #test trace generation with sample puzzle
    from data.data_parser import create_sample_puzzle
    from solver.csp_solver import SolverFactory
    
    print("Testing Trace Generation")
    print("=" * 50)
    
    puzzle = create_sample_puzzle()
    solver = SolverFactory.create_full(puzzle)
    
    print("Solving puzzle...")
    solution = solver.solve()
    
    print(f"Solution found: {solution is not None}")
    print(f"Total steps: {solver.get_stats()['total_steps']}")
    
    #generate trace
    generator = TraceGenerator(puzzle)
    traces = generator.generate(solver)
    
    print(f"\nTrace entries: {len(traces)}")
    
    #save trace
    generator.save_trace('./traces/sample_trace.json')
    print("\nTrace saved to traces/sample_trace.json")
    
    #get feature matrix
    X, y = generator.get_feature_matrix()
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Positive labels: {y.sum()}")
    
    #print feature names
    print(f"\nFeature names: {FeatureVector.feature_names()}")
