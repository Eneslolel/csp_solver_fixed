#evaluation and analysis for CSP Solver

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_parser import CSPPuzzle, PuzzleParser
from solver.csp_solver import CSPSolver, SolverFactory, solution_to_grid


@dataclass
class PuzzleResult:
    #result for a single puzzle
    puzzle_id: str
    size: str
    solved: bool
    correct: bool
    steps: int
    backtracks: int
    time_seconds: float
    solution: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class EvaluationMetrics:
    #comprehensive evaluation metrics
    #overall metrics
    total_puzzles: int = 0
    solved_count: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    
    #efficiency metrics
    total_steps: int = 0
    avg_steps: float = 0.0
    median_steps: float = 0.0
    max_steps: int = 0
    min_steps: int = 0
    
    #time metrics
    total_time: float = 0.0
    avg_time: float = 0.0
    
    #backtracking metrics
    total_backtracks: int = 0
    avg_backtracks: float = 0.0
    
    #by size metrics
    by_size: Dict[str, Dict] = field(default_factory=dict)
    
    #composite score
    composite_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SolutionVerifier:
    #verifies solver solutions against ground truth
    
    @staticmethod
    def verify(solution: Dict[Tuple[str, str], int], 
               ground_truth: Dict, 
               puzzle: CSPPuzzle) -> Tuple[bool, float]:
        
        #verify a solution against ground truth
        
        #args:
            #solution: Solver output {(category, value): house_position}
            #ground_truth: Ground truth from dataset
            #puzzle: Original puzzle object
            
        #returns:
            #(is_correct, cell_accuracy)
       
        if solution is None:
            return False, 0.0
        
        if not ground_truth:
            return True, 1.0  #no ground truth to compare
        
        #extract expected solution
        if 'header' in ground_truth and 'rows' in ground_truth:
            header = ground_truth['header']
            rows = ground_truth['rows']
            
            total_cells = 0
            correct_cells = 0
            
            for row_idx, row in enumerate(rows):
                house_pos = row_idx + 1
                for col_idx, value in enumerate(row):
                    if col_idx < len(header):
                        cat = header[col_idx].lower()
                        val = value.lower()
                        key = (cat, val)
                        
                        if key in solution:
                            total_cells += 1
                            if solution[key] == house_pos:
                                correct_cells += 1
            
            if total_cells == 0:
                return True, 1.0
            
            cell_accuracy = correct_cells / total_cells
            is_correct = cell_accuracy == 1.0
            
            return is_correct, cell_accuracy
        
        return True, 1.0  #default if format not recognized


class Evaluator:
    #main evaluator class for running comprehensive evaluation.
    
    def __init__(self, solver_factory=None, alpha: float = 10.0):
        #args:
            #solver_factory: Factory function to create solvers
            #alpha: Efficiency penalty weight for composite score
        self.solver_factory = solver_factory or SolverFactory.create_full
        self.alpha = alpha
        self.verifier = SolutionVerifier()
        
    def evaluate_puzzle(self, puzzle: CSPPuzzle) -> PuzzleResult:
        #evaluate solver on a single puzzle
        start_time = time.time()
        
        try:
            solver = self.solver_factory(puzzle)
            solution = solver.solve()
            stats = solver.get_stats()
            
            elapsed = time.time() - start_time
            
            #verify solution
            is_correct = False
            if solution is not None and puzzle.solution:
                is_correct, _ = self.verifier.verify(solution, puzzle.solution, puzzle)
            elif solution is not None:
                is_correct = stats['solved']  # Assume correct if no ground truth
            
            return PuzzleResult(
                puzzle_id=puzzle.puzzle_id,
                size=puzzle.size,
                solved=stats['solved'],
                correct=is_correct,
                steps=stats['total_steps'],
                backtracks=stats['backtracks'],
                time_seconds=elapsed,
                solution={str(k): v for k, v in solution.items()} if solution else None,
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return PuzzleResult(
                puzzle_id=puzzle.puzzle_id,
                size=puzzle.size,
                solved=False,
                correct=False,
                steps=0,
                backtracks=0,
                time_seconds=elapsed,
                error=str(e),
            )
    
    def evaluate_batch(self, puzzles: List[CSPPuzzle], 
                      progress_callback=None) -> Tuple[List[PuzzleResult], EvaluationMetrics]:
        #evaluate solver on a batch of puzzles.
        
        #args:
            #puzzles: List of puzzles to evaluate
            #progress_callback: Optional callback(current, total, result)
            
        #returns:
            #(results, metrics)
        results = []
        
        for i, puzzle in enumerate(puzzles):
            result = self.evaluate_puzzle(puzzle)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(puzzles), result)
        
        metrics = self.compute_metrics(results)
        return results, metrics
    
    def compute_metrics(self, results: List[PuzzleResult], 
                       max_avg_steps: Optional[float] = None) -> EvaluationMetrics:
        #compute comprehensive metrics from results
        metrics = EvaluationMetrics()
        
        metrics.total_puzzles = len(results)
        if not results:
            return metrics
        
        #aggregate metrics
        steps_list = []
        times_list = []
        by_size = defaultdict(lambda: {'total': 0, 'solved': 0, 'correct': 0, 
                                       'steps': [], 'times': []})
        
        for result in results:
            if result.solved:
                metrics.solved_count += 1
            if result.correct:
                metrics.correct_count += 1
            
            metrics.total_steps += result.steps
            metrics.total_backtracks += result.backtracks
            metrics.total_time += result.time_seconds
            
            steps_list.append(result.steps)
            times_list.append(result.time_seconds)
            
            #by size
            size = result.size or 'unknown'
            by_size[size]['total'] += 1
            by_size[size]['steps'].append(result.steps)
            by_size[size]['times'].append(result.time_seconds)
            if result.solved:
                by_size[size]['solved'] += 1
            if result.correct:
                by_size[size]['correct'] += 1
        
        #compute averages
        n = len(results)
        metrics.accuracy = metrics.correct_count / n
        metrics.avg_steps = metrics.total_steps / n
        metrics.median_steps = float(np.median(steps_list))
        metrics.max_steps = max(steps_list)
        metrics.min_steps = min(steps_list)
        metrics.avg_time = metrics.total_time / n
        metrics.avg_backtracks = metrics.total_backtracks / n
        
        #by size metrics
        for size, data in by_size.items():
            metrics.by_size[size] = {
                'total': data['total'],
                'solved': data['solved'],
                'correct': data['correct'],
                'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
                'avg_steps': np.mean(data['steps']) if data['steps'] else 0,
                'avg_time': np.mean(data['times']) if data['times'] else 0,
            }
        
        #compute composite score
        if max_avg_steps is None:
            max_avg_steps = metrics.avg_steps  #self-reference for single solver
        
        if max_avg_steps > 0:
            efficiency_penalty = self.alpha * (metrics.avg_steps / max_avg_steps)
        else:
            efficiency_penalty = 0
        
        metrics.composite_score = (metrics.accuracy * 100) - efficiency_penalty
        
        return metrics


class ComparativeEvaluator:
    #compare multiple solver configurations
    
    def __init__(self, puzzles: List[CSPPuzzle]):
        self.puzzles = puzzles
        
    def compare_solvers(self, solver_configs: Dict[str, callable]) -> Dict[str, EvaluationMetrics]:
        #compare multiple solver configurations
        
        #args:
            #solver_configs: {name: solver_factory}
            
        #returns:
            #{name: metrics}
        all_results = {}
        max_avg_steps = 0
        
        #run all solvers
        for name, factory in solver_configs.items():
            print(f"\nEvaluating: {name}")
            print("-" * 40)
            
            evaluator = Evaluator(solver_factory=factory)
            results, metrics = evaluator.evaluate_batch(
                self.puzzles,
                progress_callback=lambda i, n, r: print(f"  [{i}/{n}] {r.puzzle_id}: "
                                                        f"{'✓' if r.correct else '✗'} "
                                                        f"({r.steps} steps)")
            )
            
            all_results[name] = (results, metrics)
            max_avg_steps = max(max_avg_steps, metrics.avg_steps)
        
        #recompute composite scores with global max
        final_metrics = {}
        for name, (results, metrics) in all_results.items():
            evaluator = Evaluator(solver_factory=solver_configs[name])
            metrics = evaluator.compute_metrics(results, max_avg_steps)
            final_metrics[name] = metrics
        
        return final_metrics
    
    def generate_report(self, metrics: Dict[str, EvaluationMetrics]) -> str:
        #generate a comparison report
        lines = []
        lines.append("=" * 70)
        lines.append("CSP SOLVER EVALUATION REPORT")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("=" * 70)
        lines.append("")
        
        #summary table
        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(f"{'Solver':<25} {'Accuracy':>10} {'Avg Steps':>12} {'Composite':>12}")
        lines.append("-" * 70)
        
        for name, m in sorted(metrics.items(), key=lambda x: -x[1].composite_score):
            lines.append(f"{name:<25} {m.accuracy*100:>9.1f}% {m.avg_steps:>12.1f} {m.composite_score:>12.2f}")
        
        lines.append("-" * 70)
        lines.append("")
        
        #detailed metrics for each solver
        for name, m in metrics.items():
            lines.append(f"\n{name}")
            lines.append("=" * 40)
            lines.append(f"Total puzzles:     {m.total_puzzles}")
            lines.append(f"Solved:            {m.solved_count} ({m.solved_count/m.total_puzzles*100:.1f}%)")
            lines.append(f"Correct:           {m.correct_count} ({m.accuracy*100:.1f}%)")
            lines.append(f"Total steps:       {m.total_steps}")
            lines.append(f"Avg steps:         {m.avg_steps:.1f}")
            lines.append(f"Median steps:      {m.median_steps:.1f}")
            lines.append(f"Avg backtracks:    {m.avg_backtracks:.1f}")
            lines.append(f"Total time:        {m.total_time:.2f}s")
            lines.append(f"Avg time:          {m.avg_time*1000:.1f}ms")
            lines.append(f"Composite score:   {m.composite_score:.2f}")
            
            if m.by_size:
                lines.append("\nBy Size:")
                for size, data in sorted(m.by_size.items()):
                    lines.append(f"  {size}: {data['correct']}/{data['total']} correct, "
                               f"{data['avg_steps']:.1f} avg steps")
        
        return "\n".join(lines)


class GeneralizationAnalyzer:
    #analyze solver generalization across puzzle sizes.
    
    def __init__(self, results: List[PuzzleResult]):
        self.results = results
        
    def analyze_by_size(self) -> Dict[str, Dict]:
        #analyze performance breakdown by puzzle size
        by_size = defaultdict(lambda: {
            'puzzles': [],
            'accuracy': 0,
            'avg_steps': 0,
            'avg_time': 0,
        })
        
        for result in self.results:
            size = result.size or 'unknown'
            by_size[size]['puzzles'].append(result)
        
        for size, data in by_size.items():
            puzzles = data['puzzles']
            n = len(puzzles)
            data['count'] = n
            data['accuracy'] = sum(1 for p in puzzles if p.correct) / n if n > 0 else 0
            data['avg_steps'] = sum(p.steps for p in puzzles) / n if n > 0 else 0
            data['avg_time'] = sum(p.time_seconds for p in puzzles) / n if n > 0 else 0
            del data['puzzles']
        
        return dict(by_size)
    
    def compute_generalization_score(self, training_sizes: List[str]) -> float:
        #compute generalization score: performance on unseen sizes vs training sizes.
        by_size = self.analyze_by_size()
        
        training_acc = []
        test_acc = []
        
        for size, data in by_size.items():
            if size in training_sizes:
                training_acc.append(data['accuracy'])
            else:
                test_acc.append(data['accuracy'])
        
        if not training_acc or not test_acc:
            return 1.0  # No comparison possible
        
        #generalization score: ratio of test accuracy to training accuracy
        avg_train = np.mean(training_acc)
        avg_test = np.mean(test_acc)
        
        if avg_train == 0:
            return 0.0
        
        return min(1.0, avg_test / avg_train)


def run_evaluation_pipeline(puzzles: List[CSPPuzzle], output_dir: str = './evaluation'):
    #run the full evaluation pipeline.
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("CSP SOLVER EVALUATION PIPELINE")
    print("=" * 70)
    print(f"\nTotal puzzles: {len(puzzles)}")
    
    #define solver configurations to compare
    solver_configs = {
        'Baseline (no heuristics)': SolverFactory.create_baseline,
        'MRV only': SolverFactory.create_mrv_only,
        'MRV + Forward Checking': SolverFactory.create_forward_checking,
        'Full (MRV+LCV+FC+AC3)': SolverFactory.create_full,
    }
    
    #run comparative evaluation
    comparator = ComparativeEvaluator(puzzles)
    all_metrics = comparator.compare_solvers(solver_configs)
    
    #generate report
    report = comparator.generate_report(all_metrics)
    print("\n" + report)
    
    #save report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    #save metrics as JSON
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    metrics_dict = {name: m.to_dict() for name, m in all_metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    #return best solver
    best_solver = max(all_metrics.items(), key=lambda x: x[1].composite_score)
    print(f"\nBest solver: {best_solver[0]} (composite score: {best_solver[1].composite_score:.2f})")
    
    return all_metrics


def create_sample_puzzles_for_testing() -> List[CSPPuzzle]:
    #create sample puzzles for testing the evaluation system
    puzzles = []
    
    #sample puzzle 1: Simple 3-house puzzle
    puzzle1_text = """
    There are 3 houses in a row.
    Names: Alice, Bob, Carol
    Colors: red, blue, green
    Pets: dog, cat, bird
    
    1. Alice lives in the red house.
    2. Bob has a cat.
    3. Carol lives in house 3.
    4. The person with the dog lives next to the green house.
    """
    
    parser = PuzzleParser()
    puzzle1 = parser.parse(puzzle1_text, puzzle_id="test_001", size="3*3")
    puzzle1.solution = {
        'header': ['Name', 'Color', 'Pet'],
        'rows': [
            ['Alice', 'red', 'bird'],
            ['Bob', 'blue', 'cat'],
            ['Carol', 'green', 'dog'],
        ]
    }
    puzzles.append(puzzle1)
    
    #sample puzzle 2: Medium 4-house puzzle
    puzzle2_text = """
    There are 4 houses.
    Names: Alice, Bob, Carol, David
    Colors: red, blue, green, yellow
    
    1. Alice lives in house 1.
    2. The green house is immediately to the right of the red house.
    3. David does not live in the yellow house.
    4. Bob lives next to Carol.
    """
    
    puzzle2 = parser.parse(puzzle2_text, puzzle_id="test_002", size="4*2")
    puzzles.append(puzzle2)
    
    #sample puzzle 3: Another 3-house puzzle
    puzzle3_text = """
    There are 3 houses.
    Names: Eve, Frank, Grace
    Drinks: coffee, tea, juice
    
    1. Eve drinks coffee.
    2. Frank lives in house 2.
    3. The juice drinker lives in house 1.
    """
    
    puzzle3 = parser.parse(puzzle3_text, puzzle_id="test_003", size="3*2")
    puzzles.append(puzzle3)
    
    return puzzles


if __name__ == "__main__":
    print("Testing Evaluation System")
    print("=" * 50)
    
    #create sample puzzles
    puzzles = create_sample_puzzles_for_testing()
    print(f"Created {len(puzzles)} sample puzzles")
    
    #run evaluation
    output_dir = './evaluation'
    metrics = run_evaluation_pipeline(puzzles, output_dir)
    
    #test generalization analyzer
    print("\n" + "=" * 50)
    print("Testing Generalization Analyzer")
    print("=" * 50)
    
    evaluator = Evaluator()
    results, _ = evaluator.evaluate_batch(puzzles)
    
    analyzer = GeneralizationAnalyzer(results)
    by_size = analyzer.analyze_by_size()
    
    print("\nPerformance by size:")
    for size, data in by_size.items():
        print(f"  {size}: {data['accuracy']*100:.1f}% accuracy, {data['avg_steps']:.1f} avg steps")
