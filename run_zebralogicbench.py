#ZebraLogicBench Pipeline Script

#run this script only from the folder containing the parquet files

#usage:
    #python run_zebralogicbench.py
    #python run_zebralogicbench.py --max_puzzles 50

import argparse
import os
import sys
import json
import time
import glob
from typing import List, Dict, Any
from collections import defaultdict

#add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data.data_parser import CSPPuzzle, PuzzleParser
from solver.csp_solver import SolverFactory
from solver.trace_generator import BatchTraceGenerator
from training.model_training import ModelTrainer
from evaluation.evaluator import ComparativeEvaluator, Evaluator


def find_parquet_file(search_dir: str = None) -> str:
    #find the grid mode parquet file
    if search_dir is None:
        search_dir = SCRIPT_DIR
    
    #search patterns
    patterns = [
        os.path.join(search_dir, 'Gridmode*.parquet'),
        os.path.join(search_dir, 'grid*.parquet'),
        os.path.join(search_dir, '*.parquet'),
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if 'mc' not in match.lower():  #skip multiple choice file
                return match
    
    return None


def load_puzzles(parquet_path: str, max_puzzles: int = None) -> List[CSPPuzzle]:
    #load puzzles from parquet file
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed!")
        print("Run: pip install pandas pyarrow")
        sys.exit(1)
    
    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    if max_puzzles:
        df = df.head(max_puzzles)
    
    print(f"Processing {len(df)} puzzles...")
    
    parser = PuzzleParser()
    puzzles = []
    
    for idx, row in df.iterrows():
        try:
            solution = None
            if 'solution' in row and row['solution']:
                if isinstance(row['solution'], str):
                    solution = json.loads(row['solution'])
                else:
                    solution = row['solution']
            
            puzzle = parser.parse(
                puzzle_text=row['puzzle'],
                puzzle_id=row.get('id', f'puzzle_{idx}'),
                size=row.get('size', ''),
                solution=solution
            )
            puzzle.solution = solution
            puzzles.append(puzzle)
            
        except Exception as e:
            print(f"  Warning: Failed to parse puzzle {idx}: {e}")
    
    print(f"Loaded {len(puzzles)} puzzles successfully")
    return puzzles


def run_evaluation(puzzles: List[CSPPuzzle], output_dir: str):
    #run evaluation with all solver configurations
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)
    
    solver_configs = {
        'Baseline': SolverFactory.create_baseline,
        'MRV': SolverFactory.create_mrv_only,
        'MRV+FC': SolverFactory.create_forward_checking,
        'Full': SolverFactory.create_full,
    }
    
    comparator = ComparativeEvaluator(puzzles)
    metrics = comparator.compare_solvers(solver_configs)
    
    #generate and save report
    report = comparator.generate_report(metrics)
    
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    metrics_dict = {name: m.to_dict() for name, m in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='ZebraLogicBench CSP Solver')
    parser.add_argument('--max_puzzles', type=int, default=None,
                       help='Max puzzles to process (default: all)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Path to parquet file (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ZEBRALOGICBENCH CSP SOLVER")
    print("=" * 60)
    
    #find parquet file
    if args.data_file:
        parquet_path = args.data_file
    else:
        parquet_path = find_parquet_file()
    
    if not parquet_path or not os.path.exists(parquet_path):
        print("\nERROR: Could not find parquet file!")
        print("Make sure Gridmode-00000-of-00001.parquet is in the same folder.")
        sys.exit(1)
    
    print(f"Found data file: {parquet_path}")
    
    #load puzzles
    puzzles = load_puzzles(parquet_path, args.max_puzzles)
    
    if not puzzles:
        print("No puzzles loaded!")
        return
    
    #run evaluation
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    metrics = run_evaluation(puzzles, output_dir)
    
    #print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    best = max(metrics.items(), key=lambda x: x[1].composite_score)
    print(f"\nBest solver: {best[0]}")
    print(f"  Accuracy: {best[1].accuracy*100:.1f}%")
    print(f"  Avg steps: {best[1].avg_steps:.1f}")
    print(f"  Composite score: {best[1].composite_score:.2f}")


if __name__ == "__main__":
    main()
