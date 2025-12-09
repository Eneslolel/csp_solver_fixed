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
csp_solver/
├── main.py                 # Main entry point
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
├── models/                 # Trained models
└── evaluation/             # Evaluation results
```

## Quick Start

### Demo Mode

Run a complete demonstration:

```bash
python main.py --mode demo
```

### Solve Puzzles

```bash
python main.py --mode solve --input puzzles.json --solver full
```

### Generate Traces

```bash
python main.py --mode traces --input puzzles.json --output traces/
```

### Train Models

```bash
python main.py --mode train --traces traces/ --output models/
```

### Evaluate

```bash
python main.py --mode evaluate --input puzzles.json --output eval/
```

## Solver Configurations

| Configuration | Description |
|--------------|-------------|
| `baseline` | Simple backtracking, no heuristics |
| `mrv` | Backtracking + MRV heuristic |
| `forward` | Backtracking + MRV + Forward Checking |
| `full` | All optimizations (MRV + LCV + FC + AC-3) |

## Data Format

### Input Puzzle JSON

```json
{
  "id": "puzzle_001",
  "size": "5*4",
  "puzzle": "There are 5 houses...\n1. Alice lives in house 1...",
  "solution": {
    "header": ["Name", "Color", "Pet"],
    "rows": [
      ["Alice", "red", "dog"],
      ...
    ]
  }
}
```

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

### Per-Puzzle Metrics

- Solved status
- Correctness (compared to ground truth)
- Search steps
- Backtrack count
- Solve time

### Generalization Metrics

- Performance breakdown by puzzle size
- Generalization score for unseen sizes

## Trace Format

Each trace entry contains:

```json
{
  "step_number": 1,
  "depth": 0,
  "chosen_variable": ["name", "alice"],
  "chosen_value": 1,
  "features": {
    "depth": 0,
    "num_assigned": 0,
    "min_domain_size": 5,
    "avg_domain_size": 5.0,
    "constraint_satisfaction_ratio": 1.0,
    ...
  },
  "action_type": "assign"
}
```

### Feature Vector (17 dimensions)

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

## ML Models

### Variable Selection Model

Neural network predicting probability that a variable selection leads to solution.

Architecture:
- Input: 17-dimensional feature vector
- Hidden layers: [64, 32, 16]
- Output: Sigmoid probability

### Value Ordering Model

Gradient boosting model for predicting value quality scores.

### Failure Prediction Model

Predicts if current search branch will fail, enabling early pruning.

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
    'baseline': SolverFactory.create_baseline,
    'full': SolverFactory.create_full,
})
```

## Dependencies

- Python 3.8+
- NumPy
- pandas (for parquet files)
- pyarrow (for parquet files)

## Performance Tips

1. **Use Full Solver**: The full configuration (MRV+LCV+FC+AC3) is typically fastest
2. **Enable Tracing Selectively**: Disable tracing for production runs
3. **Batch Processing**: Use BatchTraceGenerator for multiple puzzles
4. **Model Integration**: Train models on traces to improve heuristics

## References

- ZebraLogicBench Dataset: https://huggingface.co/datasets/allenai/ZebraLogicBench
- Russell & Norvig: "Artificial Intelligence: A Modern Approach" - Chapter 6
- Mackworth: "Consistency in Networks of Relations" (AC-3 algorithm)

## License

MIT License
