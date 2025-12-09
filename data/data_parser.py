#data Parser for ZebraLogicBench Dataset
#parses logic grid puzzles into CSP format

#components:
# + PuzzleParser: Extracts CSP variables, domains, and constraints from puzzles
# + ClueParser: Parses natural language clues into constraint functions
# + DataLoader: Loads and preprocesses the ZebraLogicBench dataset


import re
import json
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class ConstraintType(Enum):
    #Types of constraints found in Zebra puzzles
    SAME_ENTITY = "same_entity"           # Alice has a red house - same row
    DIFFERENT_ENTITY = "different_entity" # Alice does not have a dog
    NEIGHBOR = "neighbor"                 # Alice lives next to Bob
    LEFT_OF = "left_of"                   # Alice lives left of Bob
    RIGHT_OF = "right_of"                 # Alice lives right of Bob
    IMMEDIATE_LEFT = "immediate_left"     # Alice lives immediately left of Bob
    IMMEDIATE_RIGHT = "immediate_right"   # Alice lives immediately right of Bob
    BETWEEN = "between"                   # Alice lives between Bob and Carol
    NOT_NEIGHBOR = "not_neighbor"         # Alice does not live next to Bob
    POSITION = "position"                 # Alice lives in house 1
    NOT_POSITION = "not_position"         # Alice does not live in house 1
    DISTANCE = "distance"                 # Alice lives 2 houses from Bob


@dataclass
class Constraint:
    # represents a constraint in the CSP
    constraint_type: ConstraintType
    attributes: List[Tuple[str, str]]  #list of (category, value) pairs
    positions: Optional[List[int]] = None  # or position constraints
    distance: Optional[int] = None  #for distance constraints
    raw_clue: str = ""  #original clue text
    
    def __hash__(self):
        return hash((self.constraint_type, tuple(self.attributes), 
                    tuple(self.positions) if self.positions else None))


@dataclass
class CSPPuzzle:
    #CSP representation of a logic grid puzzle
    puzzle_id: str
    num_houses: int
    categories: Dict[str, List[str]]  # category_name -> list of values
    constraints: List[Constraint]
    solution: Optional[Dict[str, Dict[str, int]]] = None  # value -> house position
    raw_puzzle: str = ""
    size: str = ""
    
    @property
    def variables(self) -> List[Tuple[str, str]]:
        #all CSP variables as (category, value) pairs
        vars_list = []
        for cat, values in self.categories.items():
            for val in values:
                vars_list.append((cat, val))
        return vars_list
    
    @property
    def domains(self) -> Dict[Tuple[str, str], Set[int]]:
        #initial domains for all variables (house positions)
        return {var: set(range(1, self.num_houses + 1)) for var in self.variables}


class ClueParser:
    # parse natural language clues into constraint objects

    # handles various clue patterns found in Zebra puzzles:
    #  + direct assignment: Alice lives in house 1
    #  + same entity: The person in the red house has a dog
    #  + neighbors: Alice lives next to the person with the cat
    #  + ordering: Alice lives to the left of Bob
    #  + distance: Alice lives exactly 2 houses from Bob
    
    def __init__(self, categories: Dict[str, List[str]], num_houses: int):
        self.categories = categories
        self.num_houses = num_houses
        self.all_values = {}  # value -> category mapping
        for cat, values in categories.items():
            for val in values:
                self.all_values[val.lower()] = cat
    
    def parse_clue(self, clue: str) -> List[Constraint]:
        #parse a natural language clue into constraints
        clue_lower = clue.lower().strip()
        constraints = []
        
        #try each parsing pattern
        patterns = [
            self._parse_position,
            self._parse_same_entity,
            self._parse_different_entity,
            self._parse_immediate_neighbor,
            self._parse_neighbor,
            self._parse_left_right,
            self._parse_between,
            self._parse_distance,
            self._parse_not_neighbor,
            self._parse_not_position,
        ]
        
        for pattern_func in patterns:
            result = pattern_func(clue_lower, clue)
            if result:
                constraints.extend(result if isinstance(result, list) else [result])
                break
        
        return constraints
    
    def _find_values_in_text(self, text: str) -> List[Tuple[str, str]]:
        #find all known values mentioned in text
        found = []
        for val, cat in self.all_values.items():
            if val in text.lower():
                found.append((cat, val))
        return found
    
    def _parse_position(self, clue_lower: str, raw_clue: str) -> Optional[Constraint]:
        #parse direct position assignments like: Alice lives in house 1
        patterns = [
            r'(.+?)\s+(?:is|lives|works)\s+(?:in|at)\s+(?:house|position)\s+(\d+)',
            r'(?:house|position)\s+(\d+)\s+(?:is|has|contains)\s+(.+)',
            r'the\s+(.+?)\s+(?:is|lives)\s+(?:in|at)\s+(?:the\s+)?(?:first|second|third|fourth|fifth|sixth)',
        ]
        
        ordinal_map = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6}
        
        for pattern in patterns:
            match = re.search(pattern, clue_lower)
            if match:
                groups = match.groups()
                position = None
                value_text = None
                
                if groups[0].isdigit():
                    position = int(groups[0])
                    value_text = groups[1]
                elif len(groups) > 1 and groups[1].isdigit():
                    position = int(groups[1])
                    value_text = groups[0]
                else:
                    for word, pos in ordinal_map.items():
                        if word in clue_lower:
                            position = pos
                            value_text = groups[0]
                            break
                
                if position and value_text:
                    attrs = self._find_values_in_text(value_text)
                    if attrs:
                        return Constraint(
                            constraint_type=ConstraintType.POSITION,
                            attributes=attrs,
                            positions=[position],
                            raw_clue=raw_clue
                        )
        return None
    
    def _parse_same_entity(self, clue_lower: str, raw_clue: str) -> Optional[Constraint]:
        #parse same-entity constraints like 'Alice has a dog' or 'The red house has a cat'.
        patterns = [
            r'the\s+(?:person|one|owner|man|woman)\s+(?:who|that|with)\s+(.+?)\s+(?:also\s+)?(?:has|owns|keeps|drives|drinks|likes|lives)',
            r'(.+?)\s+(?:has|owns|keeps|drives|drinks|likes|plays|eats)\s+(?:a\s+|the\s+)?(.+)',
            r'the\s+(.+?)\s+(?:house|car|pet|person)\s+(?:belongs to|is owned by|is|has)\s+(.+)',
        ]
        
        attrs = self._find_values_in_text(clue_lower)
        if len(attrs) >= 2:
            #check for negation
            if 'not' in clue_lower or "n't" in clue_lower or 'neither' in clue_lower:
                return Constraint(
                    constraint_type=ConstraintType.DIFFERENT_ENTITY,
                    attributes=attrs,
                    raw_clue=raw_clue
                )
            return Constraint(
                constraint_type=ConstraintType.SAME_ENTITY,
                attributes=attrs,
                raw_clue=raw_clue
            )
        return None
    
    def _parse_different_entity(self, clue_lower: str, raw_clue: str) -> Optional[Constraint]:
        #parse different-entity constraints with explicit negation
        if 'not' not in clue_lower and "n't" not in clue_lower:
            return None
            
        attrs = self._find_values_in_text(clue_lower)
        if len(attrs) >= 2:
            return Constraint(
                constraint_type=ConstraintType.DIFFERENT_ENTITY,
                attributes=attrs,
                raw_clue=raw_clue
            )
        return None
    
    def _parse_immediate_neighbor(self, clue_lower: str, raw_clue: str) -> Optional[Constraint]:
        #parse immediate neighbor constraints
        patterns = [
            r'immediately\s+(?:to\s+the\s+)?(left|right)\s+of',
            r'directly\s+(?:to\s+the\s+)?(left|right)\s+of',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clue_lower)
            if match:
                direction = match.group(1)
                attrs = self._find_values_in_text(clue_lower)
                if len(attrs) >= 2:
                    ctype = ConstraintType.IMMEDIATE_LEFT if direction == 'left' else ConstraintType.IMMEDIATE_RIGHT
                    return Constraint(
                        constraint_type=ctype,
                        attributes=attrs,
                        raw_clue=raw_clue
                    )
        return None
    
    def _parse_neighbor(self, clue_lower: str, raw_clue: str) -> Optional[Constraint]:
        #parse neighbor constraints like 'Alice lives next to Bob'
        patterns = [
            r'next\s+to',
            r'beside',
            r'adjacent\s+to',
            r'neighbor',
        ]
        
        for pattern in patterns:
            if re.search(pattern, clue_lower):
                if 'not' in clue_lower or "n't" in clue_lower:
                    continue
                attrs = self._find_values_in_text(clue_lower)
                if len(attrs) >= 2:
                    return Constraint(
                        constraint_type=ConstraintType.NEIGHBOR,
                        attributes=attrs,
                        raw_clue=raw_clue
                    )
        return None
    
    def _parse_not_neighbor(self, clue_lower: str, raw_clue: str) -> Optional[Constraint]:
        #parse not-neighbor constraints
        if ('not' in clue_lower or "n't" in clue_lower) and \
           ('next to' in clue_lower or 'beside' in clue_lower or 'neighbor' in clue_lower):
            attrs = self._find_values_in_text(clue_lower)
            if len(attrs) >= 2:
                return Constraint(
                    constraint_type=ConstraintType.NOT_NEIGHBOR,
                    attributes=attrs,
                    raw_clue=raw_clue
                )
        return None
    
    def _parse_left_right(self, clue_lower: str, raw_clue: str) -> Optional[Constraint]:
        #parse left/right ordering constraints
        left_patterns = [
            r'(?:to\s+the\s+)?left\s+of',
            r'before',
        ]
        right_patterns = [
            r'(?:to\s+the\s+)?right\s+of',
            r'after',
        ]
        
        for pattern in left_patterns:
            if re.search(pattern, clue_lower):
                if 'immediately' in clue_lower or 'directly' in clue_lower:
                    continue
                attrs = self._find_values_in_text(clue_lower)
                if len(attrs) >= 2:
                    return Constraint(
                        constraint_type=ConstraintType.LEFT_OF,
                        attributes=attrs,
                        raw_clue=raw_clue
                    )
        
        for pattern in right_patterns:
            if re.search(pattern, clue_lower):
                if 'immediately' in clue_lower or 'directly' in clue_lower:
                    continue
                attrs = self._find_values_in_text(clue_lower)
                if len(attrs) >= 2:
                    return Constraint(
                        constraint_type=ConstraintType.RIGHT_OF,
                        attributes=attrs,
                        raw_clue=raw_clue
                    )
        return None
    
    def _parse_between(self, clue_lower: str, raw_clue: str) -> Optional[Constraint]:
        #parse between constraints like: Alice lives between Bob and Carol
        if 'between' in clue_lower:
            attrs = self._find_values_in_text(clue_lower)
            if len(attrs) >= 3:
                return Constraint(
                    constraint_type=ConstraintType.BETWEEN,
                    attributes=attrs,
                    raw_clue=raw_clue
                )
        return None
    
    def _parse_distance(self, clue_lower: str, raw_clue: str) -> Optional[Constraint]:
        #parse distance constraints like: Alice lives 2 houses from Bob
        patterns = [
            r'(\d+)\s+houses?\s+(?:away\s+)?from',
            r'exactly\s+(\d+)\s+(?:houses?\s+)?(?:away\s+)?from',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clue_lower)
            if match:
                distance = int(match.group(1))
                attrs = self._find_values_in_text(clue_lower)
                if len(attrs) >= 2:
                    return Constraint(
                        constraint_type=ConstraintType.DISTANCE,
                        attributes=attrs,
                        distance=distance,
                        raw_clue=raw_clue
                    )
        return None
    
    def _parse_not_position(self, clue_lower: str, raw_clue: str) -> Optional[Constraint]:
        #parse negated position constraints
        if ('not' in clue_lower or "n't" in clue_lower) and \
           ('house' in clue_lower or 'position' in clue_lower):
            match = re.search(r'(\d+)', clue_lower)
            if match:
                position = int(match.group(1))
                attrs = self._find_values_in_text(clue_lower)
                if attrs:
                    return Constraint(
                        constraint_type=ConstraintType.NOT_POSITION,
                        attributes=attrs,
                        positions=[position],
                        raw_clue=raw_clue
                    )
        return None


class PuzzleParser:
    #main parser that converts full puzzle text into CSP format.
    def __init__(self):
        self.ordinal_patterns = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4,
            'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8
        }
    
    def parse(self, puzzle_text: str, puzzle_id: str = "", size: str = "", 
              solution: Optional[Dict] = None) -> CSPPuzzle:
        
        #parse a puzzle from its natural language description.
        
        # args:
            #puzzle_text: Full puzzle description including setup and clues
            #puzzle_id: Unique identifier for the puzzle
            #size: Size string like "5*6" (5 houses, 6 attributes)
            #solution: Ground truth solution if available
        
        # returns:
            #CSPPuzzle object with variables, domains, and constraints
    
        #extract size info
        if size:
            parts = size.split('*')
            num_houses = int(parts[0])
            num_features = int(parts[1]) if len(parts) > 1 else 4
        else:
            num_houses = self._infer_num_houses(puzzle_text)
            num_features = 4
        
        # extract categories and values
        categories = self._extract_categories(puzzle_text, num_houses, num_features)
        
        # parse clues
        clue_parser = ClueParser(categories, num_houses)
        clues = self._extract_clues(puzzle_text)
        
        constraints = []
        for clue in clues:
            parsed = clue_parser.parse_clue(clue)
            constraints.extend(parsed)
        
        # Parse solution if available
        parsed_solution = None
        if solution:
            parsed_solution = self._parse_solution(solution, categories)
        
        return CSPPuzzle(
            puzzle_id=puzzle_id,
            num_houses=num_houses,
            categories=categories,
            constraints=constraints,
            solution=parsed_solution,
            raw_puzzle=puzzle_text,
            size=size
        )
    
    def _infer_num_houses(self, text: str) -> int:
        #infer number of houses from puzzle text
        patterns = [
            r'(\d+)\s+(?:houses|people|persons|friends)',
            r'there are\s+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return 5  # Default
    
    def _extract_categories(self, text: str, num_houses: int, 
                           num_features: int) -> Dict[str, List[str]]:
        #Extract categories and their values from puzzle text
        #This is a heuristic-based extraction that looks for common patterns
        #in how puzzle setup describes the attributes
        
        categories = {}
        text_lower = text.lower()
        
        # common category patterns
        category_patterns = [
            (r'names?\s*(?:are|:)\s*([^.]+)', 'name'),
            (r'nationali(?:ties|ty)\s*(?:are|:)\s*([^.]+)', 'nationality'),
            (r'colors?\s*(?:are|:)\s*([^.]+)', 'color'),
            (r'pets?\s*(?:are|:)\s*([^.]+)', 'pet'),
            (r'drinks?\s*(?:are|:)\s*([^.]+)', 'drink'),
            (r'(?:cigarette|smoke)s?\s*(?:are|:)\s*([^.]+)', 'smoke'),
            (r'jobs?\s*(?:are|:)\s*([^.]+)', 'job'),
            (r'foods?\s*(?:are|:)\s*([^.]+)', 'food'),
            (r'sports?\s*(?:are|:)\s*([^.]+)', 'sport'),
            (r'hobbies?\s*(?:are|:)\s*([^.]+)', 'hobby'),
            (r'cars?\s*(?:are|:)\s*([^.]+)', 'car'),
            (r'flowers?\s*(?:are|:)\s*([^.]+)', 'flower'),
            (r'(?:house\s+)?numbers?\s*(?:are|:)\s*([^.]+)', 'house_number'),
        ]
        
        for pattern, cat_name in category_patterns:
            match = re.search(pattern, text_lower)
            if match:
                values_text = match.group(1)
                values = self._parse_value_list(values_text)
                if values:
                    categories[cat_name] = values[:num_houses]
        
        # If we couldn't extract enough categories, use the solution if available
        if len(categories) < 2:
            # Fall back to generic extraction
            categories = self._generic_category_extraction(text, num_houses, num_features)
        
        return categories
    
    def _parse_value_list(self, text: str) -> List[str]:
        #parse a comma/and separated list of values
        # remove common artifacts
        text = re.sub(r'\s+and\s+', ', ', text)
        text = re.sub(r'\s+or\s+', ', ', text)
        
        values = [v.strip() for v in text.split(',')]
        values = [v for v in values if v and len(v) > 1]
        
        return values
    
    def _generic_category_extraction(self, text: str, num_houses: int,
                                    num_features: int) -> Dict[str, List[str]]:
        #generic extraction when specific patterns do not work
        #uses sentence structure and common puzzle conventions
        
        categories = {'house': [str(i) for i in range(1, num_houses + 1)]}
        
        # look for numbered list patterns
        lines = text.split('\n')
        for line in lines:
            # check for Category: value1, value2, etc. pattern
            match = re.match(r'([A-Za-z]+)\s*:\s*(.+)', line)
            if match:
                cat_name = match.group(1).lower()
                values = self._parse_value_list(match.group(2))
                if len(values) >= num_houses - 1:
                    categories[cat_name] = values[:num_houses]
        
        return categories
    
    def _extract_clues(self, text: str) -> List[str]:
        #extract individual clues from puzzle text
        clues = []
        
        # split by common delimiters
        lines = text.replace('\n', ' ').split('.')
        
        # also try numbered clues
        numbered_pattern = r'(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\Z)'
        numbered_matches = re.findall(numbered_pattern, text)
        
        if numbered_matches:
            clues = [m.strip() for m in numbered_matches if m.strip()]
        else:
            clues = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
        
        return clues
    
    def _parse_solution(self, solution: Dict, categories: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        #parse solution into value -> position mapping
        parsed = {}
        
        if 'header' in solution and 'rows' in solution:
            header = solution['header']
            rows = solution['rows']
            
            for row_idx, row in enumerate(rows):
                house_pos = row_idx + 1
                for col_idx, value in enumerate(row):
                    if col_idx < len(header):
                        cat = header[col_idx].lower()
                        if cat not in parsed:
                            parsed[cat] = {}
                        parsed[cat][value.lower()] = house_pos
        
        return parsed


class DataLoader:
    #loads and preprocesses ZebraLogicBench dataset.
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.parser = PuzzleParser()
    
    def load_grid_mode(self, parquet_path: str) -> List[CSPPuzzle]:
        #Load puzzles from grid_mode parquet file
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            
            puzzles = []
            for _, row in df.iterrows():
                puzzle = self.parser.parse(
                    puzzle_text=row['puzzle'],
                    puzzle_id=row['id'],
                    size=row.get('size', ''),
                    solution=row.get('solution')
                )
                puzzles.append(puzzle)
            
            return puzzles
        except ImportError:
            raise ImportError("pandas and pyarrow required for loading parquet files")
    
    def load_mc_mode(self, parquet_path: str) -> List[Dict]:
        #load multiple choice questions from mc_mode parquet file
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            
            questions = []
            for _, row in df.iterrows():
                questions.append({
                    'id': row['id'],
                    'puzzle': row['puzzle'],
                    'question': row['question'],
                    'choices': row['choices'],
                    'answer': row.get('answer'),
                })
            
            return questions
        except ImportError:
            raise ImportError("pandas and pyarrow required for loading parquet files")
    
    def load_from_json(self, json_path: str) -> List[CSPPuzzle]:
        #load puzzles from JSON format
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        puzzles = []
        for item in data:
            puzzle = self.parser.parse(
                puzzle_text=item['puzzle'],
                puzzle_id=item.get('id', ''),
                size=item.get('size', ''),
                solution=item.get('solution')
            )
            puzzles.append(puzzle)
        
        return puzzles
    
    def split_dataset(self, puzzles: List[CSPPuzzle], 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Tuple[List, List, List]:
        #split puzzles into train/val/test sets
        import random
        shuffled = puzzles.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return (
            shuffled[:train_end],
            shuffled[train_end:val_end],
            shuffled[val_end:]
        )
    
    def get_size_distribution(self, puzzles: List[CSPPuzzle]) -> Dict[str, int]:
        #get distribution of puzzle sizes
        sizes = defaultdict(int)
        for p in puzzles:
            sizes[p.size] += 1
        return dict(sizes)


def create_sample_puzzle() -> CSPPuzzle:
    #create a sample puzzle for testing
    sample_text = """
    There are 5 houses in a row, each of a different color.
    In each house lives a person of different nationality.
    Each person has a different pet, favorite drink, and smokes a different brand.
    
    Names are: Alice, Bob, Carol, David, Eve.
    Colors are: red, green, blue, yellow, white.
    Pets are: dog, cat, bird, fish, horse.
    Drinks are: coffee, tea, milk, juice, water.
    
    Clues:
    1. Alice lives in the red house.
    2. Bob has a dog.
    3. The person in the green house drinks coffee.
    4. Carol lives immediately to the left of the blue house.
    5. The person who owns the cat lives next to the person who drinks milk.
    6. David lives in house 1.
    7. Eve does not live next to Alice.
    """
    
    parser = PuzzleParser()
    return parser.parse(sample_text, puzzle_id="sample_001", size="5*5")


if __name__ == "__main__":
    #test the parser with a sample puzzle
    puzzle = create_sample_puzzle()
    
    print(f"Puzzle ID: {puzzle.puzzle_id}")
    print(f"Number of houses: {puzzle.num_houses}")
    print(f"Categories: {list(puzzle.categories.keys())}")
    print(f"Number of variables: {len(puzzle.variables)}")
    print(f"Number of constraints: {len(puzzle.constraints)}")
    
    print("\nConstraints parsed:")
    for c in puzzle.constraints:
        print(f"  {c.constraint_type.value}: {c.attributes}")
