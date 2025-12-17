#!/usr/bin/env python3
#Generate Results CSV for ZebraLogicBench
#Outputs results in the required format:
  #id, grid_solution, steps

#Usage:
    #python generate_results.py
    #python generate_results.py --data_file Gridmode.csv --output results.csv

import argparse
import os
import sys
import csv
import json
import glob

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data.data_parser import PuzzleParser
from solver.csp_solver import SolverFactory


def find_data_file(search_dir=None):
    """Find the grid mode data file."""
    if search_dir is None:
        search_dir = SCRIPT_DIR
    
    parent_dir = os.path.dirname(search_dir)
    
    for sdir in [search_dir, parent_dir]:
        for pattern in ['Gridmode*.csv', 'Gridmode*.parquet', 'grid*.csv', 'grid*.parquet']:
            matches = glob.glob(os.path.join(sdir, pattern))
            for match in matches:
                if 'mc' not in match.lower():
                    return match
    return None


def load_puzzles_from_csv(csv_path, max_puzzles=None):
    #Load puzzles from CSV file
    puzzles = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            if max_puzzles and idx >= max_puzzles:
                break
            
            puzzles.append({
                'id': row.get('id', f'puzzle_{idx}'),
                'size': row.get('size', ''),
                'puzzle': row['puzzle'],
                'solution': row.get('solution', None)
            })
    
    return puzzles


def load_puzzles_from_parquet(parquet_path, max_puzzles=None):
    #Load puzzles from parquet file
    import pandas as pd
    
    df = pd.read_parquet(parquet_path)
    
    if max_puzzles:
        df = df.head(max_puzzles)
    
    puzzles = []
    for idx, row in df.iterrows():
        puzzles.append({
            'id': row.get('id', f'puzzle_{idx}'),
            'size': row.get('size', ''),
            'puzzle': row['puzzle'],
            'solution': row.get('solution', None)
        })
    
    return puzzles


def solution_to_grid_format(solution, puzzle):
    #Convert solver solution to grid format.
    if not solution:
        return None
    
    num_houses = puzzle.num_houses
    categories = list(puzzle.categories.keys())
    
    # Build header: House + categories (capitalized)
    header = ["House"] + [cat.replace('_', ' ').title() for cat in categories]
    
    # Build rows: one per house
    rows = []
    for house in range(1, num_houses + 1):
        row = [str(house)]
        
        for cat in categories:
            # Find which value of this category is in this house
            found_value = None
            for (c, v), h in solution.items():
                if c == cat and h == house:
                    # Capitalize the value
                    found_value = v.title() if isinstance(v, str) else str(v)
                    break
            
            row.append(found_value if found_value else "")
        
        rows.append(row)
    
    return {
        "header": header,
        "rows": rows
    }


def main():
    parser = argparse.ArgumentParser(description='Generate results CSV for ZebraLogicBench')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Path to data file (auto-detected if not specified)')
    parser.add_argument('--output', type=str, default='results.csv',
                       help='Output CSV file path')
    parser.add_argument('--max_puzzles', type=int, default=None,
                       help='Maximum puzzles to process')
    parser.add_argument('--solver', type=str, default='full',
                       choices=['baseline', 'mrv', 'mrv_fc', 'full'],
                       help='Solver configuration to use')
    
    args = parser.parse_args()
    
    # Find data file
    data_file = args.data_file or find_data_file()
    
    if not data_file or not os.path.exists(data_file):
        print("ERROR: Could not find data file!")
        print("Make sure Gridmode-00000-of-00001.csv or .parquet is in the same folder.")
        sys.exit(1)
    
    print(f"Loading: {data_file}")
    
    # Load puzzles
    if data_file.endswith('.csv'):
        raw_puzzles = load_puzzles_from_csv(data_file, args.max_puzzles)
    else:
        raw_puzzles = load_puzzles_from_parquet(data_file, args.max_puzzles)
    
    print(f"Loaded {len(raw_puzzles)} puzzles")
    
    # Select solver
    solver_factories = {
        'baseline': SolverFactory.create_baseline,
        'mrv': SolverFactory.create_mrv_only,
        'mrv_fc': SolverFactory.create_forward_checking,
        'full': SolverFactory.create_full,
    }
    solver_factory = solver_factories[args.solver]
    
    print(f"Using solver: {args.solver}")
    print(f"Processing...")
    print()
    
    # Process puzzles and collect results
    puzzle_parser = PuzzleParser()
    results = []
    
    solved_count = 0
    
    for idx, raw in enumerate(raw_puzzles):
        # Parse puzzle
        puzzle = puzzle_parser.parse(
            puzzle_text=raw['puzzle'],
            puzzle_id=raw['id'],
            size=raw['size']
        )
        
        # Solve
        solver = solver_factory(puzzle)
        solution = solver.solve()
        stats = solver.get_stats()
        
        # Convert to grid format
        grid_solution = solution_to_grid_format(solution, puzzle)
        
        # Store result
        results.append({
            'id': raw['id'],
            'grid_solution': json.dumps(grid_solution) if grid_solution else "",
            'steps': stats['total_steps']
        })
        
        if stats['solved']:
            solved_count += 1
        
        # Progress
        status = "✓" if stats['solved'] else "✗"
        print(f"  [{idx+1}/{len(raw_puzzles)}] {raw['id']}: {status} ({stats['total_steps']} steps)")
    
    print()
    print(f"Solved: {solved_count}/{len(raw_puzzles)} ({solved_count/len(raw_puzzles)*100:.1f}%)")
    
    # Write CSV
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'grid_solution', 'steps'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
