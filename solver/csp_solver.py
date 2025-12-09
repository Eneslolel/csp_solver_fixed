#CSP Solver for Logic Grid Puzzles

#implements a backtracking CSP solver with:
# + Minimum Remaining Values (MRV) heuristic
# + Least Constraining Value (LCV) heuristic  
# + Forward Checking
# + Arc Consistency (AC-3)
# + Trace generation for search analysis

import time
import os
import sys
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from copy import deepcopy
import json

#add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_parser import CSPPuzzle, Constraint, ConstraintType


@dataclass
class SearchState:
    #represents a state in the CSP search
    domains: Dict[Tuple[str, str], Set[int]]
    assignments: Dict[Tuple[str, str], int]
    depth: int
    step_number: int
    chosen_variable: Optional[Tuple[str, str]] = None
    chosen_value: Optional[int] = None
    domain_sizes: Dict[Tuple[str, str], int] = field(default_factory=dict)
    pruned_values: int = 0
    branching_factor: int = 0
    
    def to_dict(self) -> Dict:
        #convert to dictionary for JSON serialization
        return {
            'depth': self.depth,
            'step_number': self.step_number,
            'chosen_variable': self.chosen_variable,
            'chosen_value': self.chosen_value,
            'num_assigned': len(self.assignments),
            'domain_sizes': {str(k): v for k, v in self.domain_sizes.items()},
            'min_domain_size': min(self.domain_sizes.values()) if self.domain_sizes else 0,
            'pruned_values': self.pruned_values,
            'branching_factor': self.branching_factor,
        }


@dataclass
class SolverStats:
    #statistics collected during solving
    total_steps: int = 0
    backtracks: int = 0
    forward_checks: int = 0
    arc_consistency_iterations: int = 0
    domains_pruned: int = 0
    time_elapsed: float = 0.0
    solved: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'total_steps': self.total_steps,
            'backtracks': self.backtracks,
            'forward_checks': self.forward_checks,
            'arc_consistency_iterations': self.arc_consistency_iterations,
            'domains_pruned': self.domains_pruned,
            'time_elapsed': self.time_elapsed,
            'solved': self.solved,
        }


class ConstraintChecker:
    #checks if constraints are satisfied given current assignments
    
    def __init__(self, puzzle: CSPPuzzle):
        self.puzzle = puzzle
        self.num_houses = puzzle.num_houses
        
    def check_constraint(self, constraint: Constraint,
                        assignments: Dict[Tuple[str, str], int]) -> bool:
        #check if a constraint is satisfied by current assignments
        positions = []
        for cat, val in constraint.attributes:
            key = (cat, val.lower())
            if key in assignments:
                positions.append(assignments[key])
            else:
                return True  #can not evaluate yet
        
        ctype = constraint.constraint_type
        
        if ctype == ConstraintType.SAME_ENTITY:
            return len(set(positions)) == 1
        
        elif ctype == ConstraintType.DIFFERENT_ENTITY:
            return len(set(positions)) == len(positions)
        
        elif ctype == ConstraintType.NEIGHBOR:
            if len(positions) < 2:
                return True
            return abs(positions[0] - positions[1]) == 1
        
        elif ctype == ConstraintType.NOT_NEIGHBOR:
            if len(positions) < 2:
                return True
            return abs(positions[0] - positions[1]) != 1
        
        elif ctype == ConstraintType.LEFT_OF:
            if len(positions) < 2:
                return True
            return positions[0] < positions[1]
        
        elif ctype == ConstraintType.RIGHT_OF:
            if len(positions) < 2:
                return True
            return positions[0] > positions[1]
        
        elif ctype == ConstraintType.IMMEDIATE_LEFT:
            if len(positions) < 2:
                return True
            return positions[1] - positions[0] == 1
        
        elif ctype == ConstraintType.IMMEDIATE_RIGHT:
            if len(positions) < 2:
                return True
            return positions[0] - positions[1] == 1
        
        elif ctype == ConstraintType.BETWEEN:
            if len(positions) < 3:
                return True
            return (positions[1] < positions[0] < positions[2]) or \
                   (positions[2] < positions[0] < positions[1])
        
        elif ctype == ConstraintType.POSITION:
            if constraint.positions:
                return positions[0] == constraint.positions[0]
            return True
        
        elif ctype == ConstraintType.NOT_POSITION:
            if constraint.positions:
                return positions[0] != constraint.positions[0]
            return True
        
        elif ctype == ConstraintType.DISTANCE:
            if len(positions) < 2 or constraint.distance is None:
                return True
            return abs(positions[0] - positions[1]) == constraint.distance
        
        return True
    
    def check_all_constraints(self, assignments: Dict[Tuple[str, str], int]) -> bool:
        #check if all constraints are satisfied
        for constraint in self.puzzle.constraints:
            if not self.check_constraint(constraint, assignments):
                return False
        return True
    
    def get_related_constraints(self, var: Tuple[str, str]) -> List[Constraint]:
        #get all constraints involving a variable
        related = []
        for constraint in self.puzzle.constraints:
            for cat, val in constraint.attributes:
                if (cat, val.lower()) == var:
                    related.append(constraint)
                    break
        return related


class CSPSolver:
    #main CSP solver with configurable heuristics and inference
    
    def __init__(self, puzzle: CSPPuzzle, 
                 use_mrv: bool = True,
                 use_lcv: bool = True,
                 use_forward_checking: bool = True,
                 use_arc_consistency: bool = True,
                 trace_enabled: bool = True):
        self.puzzle = puzzle
        self.use_mrv = use_mrv
        self.use_lcv = use_lcv
        self.use_forward_checking = use_forward_checking
        self.use_arc_consistency = use_arc_consistency
        self.trace_enabled = trace_enabled
        
        self.checker = ConstraintChecker(puzzle)
        self.stats = SolverStats()
        self.trace: List[SearchState] = []
        
        self.initial_domains = self._initialize_domains()
        
    def _initialize_domains(self) -> Dict[Tuple[str, str], Set[int]]:
        #initialize variable domains, applying unary constraints
        domains = {}
        
        for cat, values in self.puzzle.categories.items():
            for val in values:
                var = (cat, val.lower())
                domains[var] = set(range(1, self.puzzle.num_houses + 1))
        
        #apply position constraints
        for constraint in self.puzzle.constraints:
            if constraint.constraint_type == ConstraintType.POSITION:
                for cat, val in constraint.attributes:
                    var = (cat, val.lower())
                    if var in domains and constraint.positions:
                        domains[var] = {constraint.positions[0]}
            
            elif constraint.constraint_type == ConstraintType.NOT_POSITION:
                for cat, val in constraint.attributes:
                    var = (cat, val.lower())
                    if var in domains and constraint.positions:
                        domains[var].discard(constraint.positions[0])
        
        return domains
    
    def solve(self) -> Optional[Dict[Tuple[str, str], int]]:
        #solve the CSP puzzle
        start_time = time.time()
        
        domains = deepcopy(self.initial_domains)
        
        if self.use_arc_consistency:
            if not self._ac3(domains, {}):
                self.stats.time_elapsed = time.time() - start_time
                return None
        
        result = self._backtrack(domains, {}, 0)
        
        self.stats.time_elapsed = time.time() - start_time
        self.stats.solved = result is not None
        
        return result
    
    def _backtrack(self, domains: Dict[Tuple[str, str], Set[int]],
                   assignments: Dict[Tuple[str, str], int],
                   depth: int) -> Optional[Dict[Tuple[str, str], int]]:
        #recursive backtracking search
        self.stats.total_steps += 1
        
        if len(assignments) == len(domains):
            if self.checker.check_all_constraints(assignments):
                return assignments
            return None
        
        var = self._select_variable(domains, assignments)
        if var is None:
            return None
        
        if self.trace_enabled:
            state = SearchState(
                domains=deepcopy(domains),
                assignments=deepcopy(assignments),
                depth=depth,
                step_number=self.stats.total_steps,
                chosen_variable=var,
                domain_sizes={k: len(v) for k, v in domains.items() if k not in assignments},
                branching_factor=len(domains[var])
            )
            self.trace.append(state)
        
        ordered_values = self._order_values(var, domains, assignments)
        
        for value in ordered_values:
            test_assignments = assignments.copy()
            test_assignments[var] = value
            
            if not self.checker.check_all_constraints(test_assignments):
                continue
            
            new_domains = deepcopy(domains)
            new_domains[var] = {value}
            
            pruned = 0
            if self.use_forward_checking:
                pruned = self._forward_check(var, value, new_domains, test_assignments)
                self.stats.forward_checks += 1
                
                if any(len(d) == 0 for v, d in new_domains.items() if v not in test_assignments):
                    self.stats.backtracks += 1
                    continue
            
            if self.use_arc_consistency:
                if not self._ac3(new_domains, test_assignments):
                    self.stats.backtracks += 1
                    continue
            
            self.stats.domains_pruned += pruned
            
            if self.trace_enabled and self.trace:
                self.trace[-1].chosen_value = value
                self.trace[-1].pruned_values = pruned
            
            result = self._backtrack(new_domains, test_assignments, depth + 1)
            if result is not None:
                return result
            
            self.stats.backtracks += 1
        
        return None
    
    def _select_variable(self, domains: Dict[Tuple[str, str], Set[int]],
                        assignments: Dict[Tuple[str, str], int]) -> Optional[Tuple[str, str]]:
        #select the next variable to assign using MRV heuristic
        unassigned = [v for v in domains if v not in assignments]
        
        if not unassigned:
            return None
        
        if self.use_mrv:
            return min(unassigned, key=lambda v: len(domains[v]))
        else:
            return unassigned[0]
    
    def _order_values(self, var: Tuple[str, str],
                     domains: Dict[Tuple[str, str], Set[int]],
                     assignments: Dict[Tuple[str, str], int]) -> List[int]:
        #order values for a variable using LCV heuristic
        values = list(domains[var])
        
        if not self.use_lcv or len(values) <= 1:
            return values
        
        def count_eliminated(value):
            count = 0
            test_assignments = assignments.copy()
            test_assignments[var] = value
            
            for other_var in domains:
                if other_var not in assignments and other_var != var:
                    for other_val in domains[other_var]:
                        test_assignments[other_var] = other_val
                        if not self.checker.check_all_constraints(test_assignments):
                            count += 1
                        del test_assignments[other_var]
            return count
        
        return sorted(values, key=count_eliminated)
    
    def _forward_check(self, var: Tuple[str, str], value: int,
                      domains: Dict[Tuple[str, str], Set[int]],
                      assignments: Dict[Tuple[str, str], int]) -> int:
        #apply forward checking after assigning var=value
        pruned = 0
        related_constraints = self.checker.get_related_constraints(var)
        
        cat = var[0]
        for other_var in domains:
            if other_var != var and other_var[0] == cat:
                if value in domains[other_var]:
                    domains[other_var].discard(value)
                    pruned += 1
        
        for constraint in related_constraints:
            pruned += self._prune_from_constraint(constraint, var, value, domains, assignments)
        
        return pruned
    
    def _prune_from_constraint(self, constraint: Constraint, var: Tuple[str, str],
                              value: int, domains: Dict[Tuple[str, str], Set[int]],
                              assignments: Dict[Tuple[str, str], int]) -> int:
        #prune domain values based on a constraint
        pruned = 0
        
        other_vars = []
        for cat, val in constraint.attributes:
            other_var = (cat, val.lower())
            if other_var != var and other_var not in assignments:
                other_vars.append(other_var)
        
        if not other_vars:
            return 0
        
        for other_var in other_vars:
            values_to_remove = set()
            
            for other_value in domains[other_var]:
                test_assignments = assignments.copy()
                test_assignments[var] = value
                test_assignments[other_var] = other_value
                
                if not self.checker.check_constraint(constraint, test_assignments):
                    values_to_remove.add(other_value)
            
            for val in values_to_remove:
                domains[other_var].discard(val)
                pruned += 1
        
        return pruned
    
    def _ac3(self, domains: Dict[Tuple[str, str], Set[int]],
            assignments: Dict[Tuple[str, str], int]) -> bool:
        #apply AC-3 arc consistency algorithm
        queue = deque()
        
        for constraint in self.puzzle.constraints:
            vars_in_constraint = []
            for cat, val in constraint.attributes:
                var = (cat, val.lower())
                if var in domains:
                    vars_in_constraint.append(var)
            
            for i, v1 in enumerate(vars_in_constraint):
                for v2 in vars_in_constraint[i+1:]:
                    queue.append((v1, v2, constraint))
                    queue.append((v2, v1, constraint))
        
        for cat, values in self.puzzle.categories.items():
            vars_in_cat = [(cat, v.lower()) for v in values if (cat, v.lower()) in domains]
            for i, v1 in enumerate(vars_in_cat):
                for v2 in vars_in_cat[i+1:]:
                    queue.append((v1, v2, None))
                    queue.append((v2, v1, None))
        
        while queue:
            self.stats.arc_consistency_iterations += 1
            v1, v2, constraint = queue.popleft()
            
            if v1 in assignments or v2 in assignments:
                continue
            
            if self._revise(v1, v2, constraint, domains, assignments):
                if len(domains[v1]) == 0:
                    return False
                
                for other_constraint in self.checker.get_related_constraints(v1):
                    for cat, val in other_constraint.attributes:
                        other_var = (cat, val.lower())
                        if other_var != v1 and other_var != v2 and other_var not in assignments:
                            queue.append((other_var, v1, other_constraint))
        
        return True
    
    def _revise(self, v1: Tuple[str, str], v2: Tuple[str, str],
               constraint: Optional[Constraint],
               domains: Dict[Tuple[str, str], Set[int]],
               assignments: Dict[Tuple[str, str], int]) -> bool:
        #remove values from domain of v1 that have no support in domain of v2
        revised = False
        
        for x in list(domains[v1]):
            has_support = False
            
            for y in domains[v2]:
                if constraint is None:
                    if x != y:
                        has_support = True
                        break
                else:
                    test = assignments.copy()
                    test[v1] = x
                    test[v2] = y
                    if self.checker.check_constraint(constraint, test):
                        has_support = True
                        break
            
            if not has_support:
                domains[v1].discard(x)
                revised = True
                self.stats.domains_pruned += 1
        
        return revised
    
    def get_trace(self) -> List[Dict]:
        #get search trace as list of dictionaries
        return [state.to_dict() for state in self.trace]
    
    def get_stats(self) -> Dict:
        #get solver statistics
        return self.stats.to_dict()


class SolverFactory:
    #factory for creating solvers with different configurations
    
    @staticmethod
    def create_baseline(puzzle: CSPPuzzle) -> CSPSolver:
        #simple backtracking without heuristics
        return CSPSolver(puzzle, use_mrv=False, use_lcv=False,
                        use_forward_checking=False, use_arc_consistency=False)
    
    @staticmethod
    def create_mrv_only(puzzle: CSPPuzzle) -> CSPSolver:
        #backtracking with MRV heuristic
        return CSPSolver(puzzle, use_mrv=True, use_lcv=False,
                        use_forward_checking=False, use_arc_consistency=False)
    
    @staticmethod
    def create_forward_checking(puzzle: CSPPuzzle) -> CSPSolver:
        #backtracking with MRV and forward checking
        return CSPSolver(puzzle, use_mrv=True, use_lcv=False,
                        use_forward_checking=True, use_arc_consistency=False)
    
    @staticmethod
    def create_full(puzzle: CSPPuzzle) -> CSPSolver:
        #full solver with all optimizations
        return CSPSolver(puzzle, use_mrv=True, use_lcv=True,
                        use_forward_checking=True, use_arc_consistency=True)


def solution_to_grid(solution: Dict[Tuple[str, str], int],
                    puzzle: CSPPuzzle) -> List[List[str]]:
    #convert solution to grid format
    grid = [[''] * (len(puzzle.categories) + 1) for _ in range(puzzle.num_houses)]
    
    categories = list(puzzle.categories.keys())
    
    for house in range(puzzle.num_houses):
        grid[house][0] = str(house + 1)
    
    for (cat, val), house in solution.items():
        if cat in categories:
            col_idx = categories.index(cat) + 1
            if 0 <= house - 1 < len(grid):
                grid[house - 1][col_idx] = val
    
    return grid


if __name__ == "__main__":
    from data.data_parser import create_sample_puzzle
    
    puzzle = create_sample_puzzle()
    
    print("Testing CSP Solver")
    print("=" * 50)
    
    configs = [
        ("Baseline", SolverFactory.create_baseline),
        ("MRV", SolverFactory.create_mrv_only),
        ("Forward Checking", SolverFactory.create_forward_checking),
        ("Full", SolverFactory.create_full),
    ]
    
    for name, factory in configs:
        solver = factory(puzzle)
        solution = solver.solve()
        stats = solver.get_stats()
        
        print(f"{name}: {'Solved' if stats['solved'] else 'Failed'} "
              f"in {stats['total_steps']} steps")
        