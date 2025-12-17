# CSP Solver for ZebraLogicBench

A comprehensive Constraint Satisfaction Problem (CSP) solver for logic grid puzzles (Zebra puzzles), designed for the ZebraLogicBench dataset.

## Features

- **Symbolic CSP Solver** with multiple heuristics:
  - Minimum Remaining Values (MRV)
  - Least Constraining Value (LCV)
  - Forward Checking
  - Arc Consistency (AC-3)

- **Natural Language Parsing** for puzzle clues
- **Trace Generation** for search analysis and ML training
- **Machine Learning Models** for improving variable/value selection
- **Comprehensive Evaluation** with leaderboard metrics

## Project Structure

```
csp_solver_fixed/
├── run_zebralogicbench.py  # Main entry point
├── generate_results.py     # Generates the result.csv
├── data/
│   └── data_parser.py      # Puzzle parsing and data loading
├── solver/
│   ├── csp_solver.py       # Main CSP solver
│   └── trace_generator.py  # Search trace generation
├── training/
│   └── model_training.py   # ML model training
├── evaluation/
│   └── evaluator.py        # Performance evaluation
├── traces/                 # Generated traces
└── models/                 # Trained models
```

## Quick Start

### Installation

```bash
pip install numpy pandas pyarrow
```

### Run Evaluation

Place your data file (`.parquet` or `.csv`) in the same folder as the scripts, then:

```bash
python run_zebralogicbench.py
```

### Options

```bash
# Limit number of puzzles (for testing)
python run_zebralogicbench.py --max_puzzles 50

# Specify data file
python run_zebralogicbench.py --data_file path/to/Gridmode.parquet

# Specify output directory
python run_zebralogicbench.py --output_dir ./my_results

# To get the result.csv you need to run:
generate_results.py
```

## Solver Configurations

The evaluation compares 4 solver configurations:

| Configuration | Description |
|--------------|-------------|
| `Baseline` | Simple backtracking, no heuristics |
| `MRV` | Backtracking + MRV heuristic |
| `MRV+FC` | Backtracking + MRV + Forward Checking |
| `Full` | All optimizations (MRV + LCV + FC + AC-3) |

## Data Format

### Supported Input Formats

- **Parquet**: `Gridmode-00000-of-00001.parquet`
- **CSV**: `Gridmode-00000-of-00001.csv`

### Expected Columns

| Column | Description |
|--------|-------------|
| `id` | Puzzle identifier |
| `size` | Format: "houses*features" (e.g., "5*4") |
| `puzzle` | Natural language puzzle text |
| `solution` | JSON solution object |

### Supported Constraint Types

- **Same Entity**: "Alice has a red house"
- **Different Entity**: "Alice does not have a dog"
- **Neighbor**: "Alice lives next to Bob"
- **Left/Right Of**: "Alice lives to the left of Bob"
- **Immediate Left/Right**: "Alice lives immediately left of Bob"
- **Between**: "Alice lives between Bob and Carol"
- **Position**: "Alice lives in house 1"
- **Distance**: "Alice lives 2 houses from Bob"

## Evaluation Metrics

### Composite Score

```
Composite Score = Accuracy (%) – α × (AvgSteps / MaxAvgSteps)
```

Where:
- **Accuracy**: Percentage of puzzles solved correctly
- **AvgSteps**: Average CSP search steps per puzzle
- **MaxAvgSteps**: Maximum average steps across all configurations
- **α = 10**: Efficiency penalty weight

### Output Files

After running, results are saved to `./results/`:

- `evaluation_report.txt` - Human-readable report
- `metrics.json` - All metrics as JSON

## API Usage

### Solving a Puzzle

```python
from data.data_parser import PuzzleParser
from solver.csp_solver import SolverFactory

# Parse puzzle
parser = PuzzleParser()
puzzle = parser.parse(puzzle_text, puzzle_id="001", size="5*4")

# Create solver
solver = SolverFactory.create_full(puzzle)

# Solve
solution = solver.solve()
stats = solver.get_stats()

print(f"Solved: {stats['solved']}")
print(f"Steps: {stats['total_steps']}")
```

### Generating Traces

```python
from solver.trace_generator import TraceGenerator

solver = SolverFactory.create_full(puzzle)
solver.solve()

generator = TraceGenerator(puzzle)
traces = generator.generate(solver)

# Get feature matrix for ML
X, y = generator.get_feature_matrix()
```

### Training Models

```python
from training.model_training import ModelTrainer

trainer = ModelTrainer(trace_dir='traces/', model_dir='models/')
results = trainer.train_all_models()
```

### Evaluation

```python
from evaluation.evaluator import Evaluator, ComparativeEvaluator

# Single solver
evaluator = Evaluator()
results, metrics = evaluator.evaluate_batch(puzzles)

# Compare solvers
comparator = ComparativeEvaluator(puzzles)
all_metrics = comparator.compare_solvers({
    'Baseline': SolverFactory.create_baseline,
    'Full': SolverFactory.create_full,
})
```

## Trace Format

Each trace entry contains a 17-dimensional feature vector:

1. `depth` - Current search depth
2. `num_assigned` - Variables assigned
3. `num_unassigned` - Variables remaining
4. `assignment_ratio` - Fraction assigned
5. `min_domain_size` - Smallest domain
6. `max_domain_size` - Largest domain
7. `avg_domain_size` - Average domain size
8. `std_domain_size` - Domain size std dev
9. `num_singleton_domains` - Domains with size 1
10. `chosen_var_domain_size` - Selected variable's domain
11. `chosen_var_constraint_count` - Constraints on selected variable
12. `chosen_var_category_assigned_count` - Same-category assignments
13. `total_constraints` - Total constraint count
14. `satisfied_constraints` - Currently satisfied
15. `constraint_satisfaction_ratio` - Satisfaction ratio
16. `chosen_value` - Selected value
17. `chosen_value_eliminates` - Values pruned by choice

## Dependencies

- Python 3.8+
- NumPy
- pandas
- pyarrow

## Performance

Tested on ZebraLogicBench dataset:
- **Best Solver**: Full (MRV + LCV + FC + AC-3)
- **Accuracy**: ~93%
- **Average Steps**: ~8

## References

- ZebraLogicBench Dataset: https://huggingface.co/datasets/allenai/ZebraLogicBench
- Russell & Norvig: "Artificial Intelligence: A Modern Approach" - Chapter 6
- Mackworth: "Consistency in Networks of Relations" (AC-3 algorithm)

## License

MIT License
