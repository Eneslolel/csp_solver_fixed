#CSP Solver for ZebraLogicBench

#main entry point for the CSP solver system

#usage:
    #python main.py --mode demo
    #python main.py --mode solve --input puzzles.json
    #python main.py --mode evaluate

import argparse
import os
import sys
import json
from typing import List, Dict, Any

#add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data.data_parser import (
    CSPPuzzle, PuzzleParser, DataLoader, 
    Constraint, ConstraintType, create_sample_puzzle
)
from solver.csp_solver import (
    CSPSolver, SolverFactory, SolverStats,
    ConstraintChecker, solution_to_grid
)
from solver.trace_generator import (
    TraceGenerator, BatchTraceGenerator, 
    FeatureVector, TraceEntry
)
from training.model_training import (
    ModelTrainer, VariableSelectionModel, 
    TrainingConfig, create_synthetic_training_data
)
from evaluation.evaluator import (
    Evaluator, ComparativeEvaluator, EvaluationMetrics,
    GeneralizationAnalyzer, run_evaluation_pipeline,
    create_sample_puzzles_for_testing
)


def demo_mode():
    #run a demonstration of the CSP solver system
    print("=" * 70)
    print("CSP SOLVER DEMONSTRATION")
    print("=" * 70)
    
    #create a sample puzzle
    print("\n1. Creating sample puzzle...")
    print("-" * 40)
    
    puzzle_text = """
    There are 4 houses in a row.
    In each house lives a person with a unique name, favorite color, and pet.
    
    Names: Alice, Bob, Carol, David
    Colors: red, blue, green, yellow
    Pets: dog, cat, bird, fish
    
    Clues:
    1. Alice lives in the red house.
    2. Bob has a dog.
    3. The person in the green house has a cat.
    4. Carol lives immediately to the right of Alice.
    5. David does not live in the yellow house.
    6. The bird owner lives in house 1.
    7. The fish owner lives next to the person in the blue house.
    """
    
    parser = PuzzleParser()
    puzzle = parser.parse(puzzle_text, puzzle_id="demo_001", size="4*3")
    
    print(f"Puzzle ID: {puzzle.puzzle_id}")
    print(f"Houses: {puzzle.num_houses}")
    print(f"Categories: {list(puzzle.categories.keys())}")
    print(f"Constraints parsed: {len(puzzle.constraints)}")
    
    #solve with different configurations
    print("\n2. Solving with different configurations...")
    print("-" * 40)
    
    configs = [
        ("Baseline (no heuristics)", SolverFactory.create_baseline),
        ("MRV heuristic only", SolverFactory.create_mrv_only),
        ("MRV + Forward Checking", SolverFactory.create_forward_checking),
        ("Full (MRV+LCV+FC+AC3)", SolverFactory.create_full),
    ]
    
    results = []
    for name, factory in configs:
        solver = factory(puzzle)
        solution = solver.solve()
        stats = solver.get_stats()
        results.append((name, solution, stats))
        
        status = "✓ Solved" if stats['solved'] else "X Failed"
        print(f"{name}:")
        print(f"  {status} in {stats['total_steps']} steps, "
              f"{stats['backtracks']} backtracks, {stats['time_elapsed']*1000:.1f}ms")
    
    #show solution
    print("\n3. Solution...")
    print("-" * 40)
    
    #find first solved result
    best_solution = None
    for name, sol, stats in results:
        if sol:
            best_solution = sol
            break
    
    if best_solution:
        by_house = {}
        for (cat, val), house in best_solution.items():
            if house not in by_house:
                by_house[house] = {}
            by_house[house][cat] = val
        
        for house in sorted(by_house.keys()):
            print(f"House {house}: {by_house[house]}")
    else:
        print("No solution found!")
    
    #generate trace
    print("\n4. Generating search trace...")
    print("-" * 40)
    
    solver = SolverFactory.create_full(puzzle)
    solver.solve()
    
    trace_gen = TraceGenerator(puzzle)
    traces = trace_gen.generate(solver)
    
    print(f"Trace entries: {len(traces)}")
    
    #save trace
    trace_dir = os.path.join(SCRIPT_DIR, 'traces')
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, 'demo_trace.json')
    trace_gen.save_trace(trace_path)
    print(f"Trace saved to: {trace_path}")
    
    #training demo
    print("\n5. Training model on synthetic data...")
    print("-" * 40)
    
    X, y = create_synthetic_training_data(500)
    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    
    model = VariableSelectionModel(TrainingConfig(num_epochs=30))
    history = model.train(X, y)
    
    print(f"Final validation accuracy: {history['val_accuracy'][-1]*100:.1f}%")
    
    #save model
    model_dir = os.path.join(SCRIPT_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'demo_model.json')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


def solve_mode(args):
    #solve puzzles from input file
    print("Loading puzzles...")
    
    loader = DataLoader()
    
    if args.input.endswith('.json'):
        puzzles = loader.load_from_json(args.input)
    else:
        print(f"Unsupported file format: {args.input}")
        return
    
    print(f"Loaded {len(puzzles)} puzzles")
    
    solver_map = {
        'baseline': SolverFactory.create_baseline,
        'mrv': SolverFactory.create_mrv_only,
        'forward': SolverFactory.create_forward_checking,
        'full': SolverFactory.create_full,
    }
    
    factory = solver_map.get(args.solver, SolverFactory.create_full)
    
    results = []
    solved_count = 0
    total_steps = 0
    
    for i, puzzle in enumerate(puzzles):
        solver = factory(puzzle)
        solution = solver.solve()
        stats = solver.get_stats()
        
        if stats['solved']:
            solved_count += 1
        total_steps += stats['total_steps']
        
        results.append({
            'id': puzzle.puzzle_id,
            'solved': stats['solved'],
            'steps': stats['total_steps'],
            'time': stats['time_elapsed'],
        })
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(puzzles)}")
    
    print(f"\nResults:")
    print(f"  Solved: {solved_count}/{len(puzzles)} ({solved_count/len(puzzles)*100:.1f}%)")
    print(f"  Avg steps: {total_steps/len(puzzles):.1f}")


def evaluate_mode(args):
    #evaluate solver on puzzles
    print("Running evaluation...")
    
    if args.input:
        loader = DataLoader()
        if args.input.endswith('.json'):
            puzzles = loader.load_from_json(args.input)
        else:
            print(f"Unsupported format: {args.input}")
            return
    else:
        puzzles = create_sample_puzzles_for_testing()
    
    output_dir = os.path.join(SCRIPT_DIR, args.output or 'evaluation')
    metrics = run_evaluation_pipeline(puzzles, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='CSP Solver for ZebraLogicBench',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', choices=['demo', 'solve', 'evaluate'],
                       default='demo', help='Operation mode')
    parser.add_argument('--input', type=str, help='Input file path')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--solver', choices=['baseline', 'mrv', 'forward', 'full'],
                       default='full', help='Solver configuration')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_mode()
    elif args.mode == 'solve':
        if not args.input:
            print("Error: --input required for solve mode")
            return
        solve_mode(args)
    elif args.mode == 'evaluate':
        evaluate_mode(args)


if __name__ == "__main__":
    main()
