"""
Corrupted Conway's Game of Life — Dataset Generator
Executive Functions Track | Kaggle AGI Benchmarks Hackathon

Generates 600 tasks total:
  - 450 random-seed tasks  (10 mutations × 3 tiers × 15 samples)
  - 150 pattern-seed tasks (10 mutations × 5 named patterns × 3 placements)

Named patterns (Blinker, Glider, Block, Toad, Beacon) are classic GoL structures
that LLMs have seen millions of times in training. Applying rule mutations to them
creates the sharpest possible test of inhibitory control — the model must suppress
its memorized pattern evolution and reason from first principles.

All answers are computed by code — zero ambiguity, zero human labeling.
"""

import random
import csv
from itertools import product

# ─────────────────────────────────────────────
# MUTATION REGISTRY
# ─────────────────────────────────────────────

MUTATION_DESCRIPTIONS = {
    "R1":  "Birth occurs with exactly 4 live neighbors (not 3).",
    "R2":  "A live cell survives with exactly 2 OR 4 neighbors (not 2–3).",
    "R3":  "Overpopulation threshold raised: a cell survives with 2, 3, OR 4 neighbors.",
    "R4":  "Birth occurs with exactly 2 OR 4 live neighbors.",   # FIXED from {3,6}
    "R5":  "Birth occurs with exactly 2 live neighbors (not 3).",
    "R6":  "A live cell survives ONLY with exactly 3 neighbors.",
    "R7":  "The grid wraps around at edges (toroidal array).",
    "R8":  "Cells interact only with 4 cardinal Von Neumann neighbors (N/S/E/W).",
    "R9":  "Temporal delay: grid only updates on even-numbered steps. Odd steps = no change.",
    "R10": "Alternating rules: Standard rules on odd steps; Birth=4 rules on even steps.",
}

# ─────────────────────────────────────────────
# NEIGHBOR CALCULATION
# ─────────────────────────────────────────────

def get_neighbors(r, c, rows, cols, mutation):
    """Return list of neighbor coordinates based on mutation type."""
    # R8: Von Neumann (4-directional)
    if mutation == "R8":
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        # Moore (8-directional)
        directions = [
            (dr, dc)
            for dr, dc in product([-1, 0, 1], [-1, 0, 1])
            if not (dr == 0 and dc == 0)
        ]

    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if mutation == "R7":
            # Toroidal wrap
            neighbors.append((nr % rows, nc % cols))
        elif 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))

    return neighbors


# ─────────────────────────────────────────────
# SURVIVAL / BIRTH RULES
# ─────────────────────────────────────────────

def get_birth_survival(mutation):
    """Return (birth_set, survival_set) for a given mutation."""
    birth    = {3}       # Standard Conway
    survival = {2, 3}    # Standard Conway

    if mutation == "R1": birth    = {4}
    elif mutation == "R2": survival = {2, 4}
    elif mutation == "R3": survival = {2, 3, 4}
    elif mutation == "R4": birth    = {2, 4}    # FIXED: was {3,6}, unreachable on small grids
    elif mutation == "R5": birth    = {2}
    elif mutation == "R6": survival = {3}
    # R7, R8 → handled in get_neighbors (rules unchanged)
    # R9, R10 → handled in step() (rules switch per step)

    return birth, survival


# ─────────────────────────────────────────────
# SIMULATION STEP
# ─────────────────────────────────────────────

def resolve_active_mutation(mutation, current_step):
    """Return the effective mutation rule for this step (handles R9/R10)."""
    if mutation == "R9":
        return None if current_step % 2 != 0 else "R9"  # None = skip step
    if mutation == "R10":
        return "R1" if current_step % 2 == 0 else "Standard"
    return mutation


def step(live_cells, rows, cols, mutation, current_step):
    """Advance the grid by one step under the given mutation."""
    active = resolve_active_mutation(mutation, current_step)

    # R9 temporal delay: odd steps are frozen
    if active is None:
        return live_cells

    # "Standard" fallthrough: birth={3}, survival={2,3} — same as default
    birth, survival = get_birth_survival(active)

    # Collect all candidate cells (live cells + their neighbors)
    candidates = set(live_cells)
    for (r, c) in live_cells:
        for n in get_neighbors(r, c, rows, cols, active):
            candidates.add(n)

    new_live = set()
    for (r, c) in candidates:
        neighbors = get_neighbors(r, c, rows, cols, active)
        live_count = sum(1 for n in neighbors if n in live_cells)
        is_alive   = (r, c) in live_cells

        alive_next = (live_count in survival) if is_alive else (live_count in birth)
        if alive_next:
            new_live.add((r, c))

    return frozenset(new_live)


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

def build_prompt(rows, cols, initial_sorted, mutation, steps):
    return f"""You are simulating a modified Game of Life on a {rows}x{cols} grid.

Standard rules apply EXCEPT: {MUTATION_DESCRIPTIONS[mutation]}

Standard rules (apply unless the exception above overrides them):
- A live cell with 2 or 3 live neighbors survives.
- A dead cell with exactly 3 live neighbors becomes alive.
- All other live cells die; all other dead cells stay dead.

Currently live cells (row, col): {initial_sorted}

What are the exact live cell coordinates after {steps} steps?
Output ONLY a sorted Python list of (row, col) tuples.
No markdown, no explanation, no code blocks.
Example format: [(0, 1), (1, 2), (3, 4)]"""


# ─────────────────────────────────────────────
# DATASET GENERATOR
# ─────────────────────────────────────────────

TIERS = [
    {
        "name":         "Easy",
        "rows":          5,
        "cols":          5,
        "steps":         2,
        "n_live_range": (6, 10),
        "r9_steps":      4,   # R9-specific: need ≥4 steps to show 2 real updates
    },
    {
        "name":         "Medium",
        "rows":          7,
        "cols":          7,
        "steps":         4,
        "n_live_range": (12, 18),
        "r9_steps":      6,
    },
    {
        "name":         "Hard",
        "rows":         10,
        "cols":         10,
        "steps":         5,
        "n_live_range": (20, 30),
        "r9_steps":      8,
    },
]

MUTATIONS   = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
SAMPLES_PER = 15   # 10 mutations × 3 tiers × 15 = 450 tasks


# ─────────────────────────────────────────────
# NAMED PATTERN LIBRARY
# ─────────────────────────────────────────────
# Each pattern is defined as relative (row, col) offsets from an anchor point.
# These are the most memorized GoL structures — perfect inhibitory control traps.

NAMED_PATTERNS = {
    "Blinker": {
        "cells":       [(0, 0), (0, 1), (0, 2)],          # horizontal blinker
        "description": "a Blinker (3 horizontal live cells — period-2 oscillator)",
        "grid":        (7, 7),
        "steps":       3,
        "r9_steps":    6,
    },
    "Glider": {
        "cells":       [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],  # standard glider
        "description": "a Glider (classic diagonal spaceship)",
        "grid":        (9, 9),
        "steps":       4,
        "r9_steps":    8,
    },
    "Block": {
        "cells":       [(0, 0), (0, 1), (1, 0), (1, 1)],  # 2×2 still life
        "description": "a Block (2×2 still life — stable under standard rules)",
        "grid":        (7, 7),
        "steps":       3,
        "r9_steps":    6,
    },
    "Toad": {
        "cells":       [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],  # period-2
        "description": "a Toad (period-2 oscillator)",
        "grid":        (8, 8),
        "steps":       4,
        "r9_steps":    8,
    },
    "Beacon": {
        "cells":       [(0, 0), (0, 1), (1, 0), (2, 3), (3, 2), (3, 3)],  # period-2
        "description": "a Beacon (period-2 oscillator made of two touching blocks)",
        "grid":        (8, 8),
        "steps":       4,
        "r9_steps":    8,
    },
}

# 3 anchor placements per pattern — top-left, center, near-bottom-right
# Defined as (row_offset, col_offset) applied to pattern cells
PLACEMENTS = ["top_left", "center", "offset"]

def place_pattern(pattern_cells, placement, rows, cols):
    """Translate pattern cells to a specific grid position."""
    max_r = max(r for r, c in pattern_cells)
    max_c = max(c for r, c in pattern_cells)

    if placement == "top_left":
        anchor_r, anchor_c = 1, 1
    elif placement == "center":
        anchor_r = (rows - max_r) // 2
        anchor_c = (cols - max_c) // 2
    else:  # offset — lower-right quadrant
        anchor_r = max(1, rows - max_r - 3)
        anchor_c = max(1, cols - max_c - 3)

    placed = []
    for r, c in pattern_cells:
        nr, nc = r + anchor_r, c + anchor_c
        if 0 <= nr < rows and 0 <= nc < cols:
            placed.append((nr, nc))

    return frozenset(placed)


def build_pattern_prompt(rows, cols, pattern_name, description,
                         initial_sorted, mutation, steps, placement):
    placement_labels = {
        "top_left": "upper-left area",
        "center":   "center",
        "offset":   "lower-right area",
    }
    return f"""You are simulating a modified Game of Life on a {rows}x{cols} grid.

Standard rules apply EXCEPT: {MUTATION_DESCRIPTIONS[mutation]}

Standard rules (apply unless the exception above overrides them):
- A live cell with 2 or 3 live neighbors survives.
- A dead cell with exactly 3 live neighbors becomes alive.
- All other live cells die; all other dead cells stay dead.

The initial state is {description}, placed in the {placement_labels[placement]} of the grid.
Currently live cells (row, col): {initial_sorted}

What are the exact live cell coordinates after {steps} steps?
Output ONLY a sorted Python list of (row, col) tuples.
No markdown, no explanation, no code blocks.
Example format: [(0, 1), (1, 2), (3, 4)]"""


def generate_pattern_tasks(task_id_start=451):
    """
    Generate 150 pattern-seeded tasks.
    10 mutations × 5 patterns × 3 placements = 150 tasks.

    These tasks are harder for models because:
    1. The model recognises the pattern name → activates memorised evolution
    2. The mutation breaks that memorised path → requires inhibitory control
    3. The prompt explicitly names the pattern, maximising the memorisation trap
    """
    tasks   = []
    task_id = task_id_start

    for mutation in MUTATIONS:
        for pattern_name, pattern_info in NAMED_PATTERNS.items():
            rows, cols = pattern_info["grid"]
            steps = (pattern_info["r9_steps"]
                     if mutation == "R9"
                     else pattern_info["steps"])

            for placement in PLACEMENTS:
                initial_live   = place_pattern(
                    pattern_info["cells"], placement, rows, cols
                )

                # Skip if pattern clipped outside grid (safety check)
                if len(initial_live) < len(pattern_info["cells"]) * 0.6:
                    continue

                initial_sorted = sorted(initial_live)

                # Ground-truth computed by code
                current = initial_live
                for s in range(1, steps + 1):
                    current = step(current, rows, cols, mutation, s)

                answer = str(sorted(current))
                prompt = build_pattern_prompt(
                    rows, cols,
                    pattern_name, pattern_info["description"],
                    initial_sorted, mutation, steps, placement
                )

                tasks.append({
                    "task_id":         f"task_{task_id:04d}",
                    "tier":            "Pattern",
                    "mutation":        mutation,
                    "rows":            rows,
                    "cols":            cols,
                    "steps":           steps,
                    "source_pattern":  pattern_name,
                    "placement":       placement,
                    "prompt":          prompt,
                    "expected_output": answer,
                })
                task_id += 1

    return tasks


def generate_dataset(output_path="corrupted_gol_benchmark.csv", seed=42):
    random.seed(seed)
    dataset  = []
    task_id  = 1

    # ── Part 1: 450 random-seed tasks ──────────────────────────────────────
    for mutation in MUTATIONS:
        for tier in TIERS:
            rows  = tier["rows"]
            cols  = tier["cols"]
            steps = tier["r9_steps"] if mutation == "R9" else tier["steps"]

            for _ in range(SAMPLES_PER):
                n_live         = random.randint(*tier["n_live_range"])
                all_cells      = [(r, c) for r in range(rows) for c in range(cols)]
                initial_live   = frozenset(random.sample(all_cells, n_live))
                initial_sorted = sorted(initial_live)

                current = initial_live
                for s in range(1, steps + 1):
                    current = step(current, rows, cols, mutation, s)

                dataset.append({
                    "task_id":        f"task_{task_id:04d}",
                    "tier":           tier["name"],
                    "mutation":       mutation,
                    "rows":           rows,
                    "cols":           cols,
                    "steps":          steps,
                    "source_pattern": "random",
                    "placement":      "random",
                    "prompt":         build_prompt(rows, cols, initial_sorted, mutation, steps),
                    "expected_output": str(sorted(current)),
                })
                task_id += 1

    random_count = len(dataset)
    print(f"  Part 1 complete: {random_count} random tasks")

    # ── Part 2: 150 pattern-seeded tasks ───────────────────────────────────
    pattern_tasks = generate_pattern_tasks(task_id_start=task_id)
    dataset.extend(pattern_tasks)
    print(f"  Part 2 complete: {len(pattern_tasks)} pattern-seeded tasks")

    # ── Export ─────────────────────────────────────────────────────────────
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["task_id", "tier", "mutation", "rows", "cols",
                      "steps", "source_pattern", "placement", "prompt",
                      "expected_output"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"\nTotal: {len(dataset)} tasks → {output_path}")
    print(f"  Random tasks:  {random_count}")
    print(f"  Pattern tasks: {len(pattern_tasks)}")
    return dataset


# ─────────────────────────────────────────────
# EVALUATOR (use this in Kaggle Benchmarks SDK)
# ─────────────────────────────────────────────

def evaluate_submission(expected: str, predicted: str) -> float:
    """
    Strict exact match after stripping whitespace.
    Returns 1.0 for correct, 0.0 for wrong.
    Both inputs are string representations of sorted coordinate lists.
    """
    return 1.0 if expected.replace(" ", "") == predicted.replace(" ", "") else 0.0


# ─────────────────────────────────────────────
# SELF-TEST — run before submitting
# ─────────────────────────────────────────────

def run_tests():
    print("Running self-tests...")
    errors = 0

    # Test 1: Standard blinker (vertical → horizontal)
    blinker = frozenset([(1, 0), (1, 1), (1, 2)])
    result  = step(blinker, 3, 3, "Standard", 1)
    assert sorted(result) == [(0, 1), (1, 1), (2, 1)], "FAIL: Standard blinker"
    print("  PASS: Standard blinker")

    # Test 2: R1 — dead cell with 3 neighbors should NOT be born
    three_around_center = frozenset([(0, 1), (1, 0), (1, 2)])
    result = step(three_around_center, 3, 3, "R1", 1)
    assert (1, 1) not in result, "FAIL: R1 should not birth with 3 neighbors"
    print("  PASS: R1 birth=4 (no birth at 3 neighbors)")

    # Test 3: R4 — dead cell with 2 live neighbors SHOULD be born (FIXED mutation)
    two_around_center = frozenset([(0, 1), (1, 0)])
    result_r4  = step(two_around_center, 3, 3, "R4", 1)
    result_std = step(two_around_center, 3, 3, "Standard", 1)
    assert (1, 1) in result_r4,  "FAIL: R4 should birth with 2 neighbors"
    assert (1, 1) not in result_std, "FAIL: Standard should NOT birth with 2 neighbors"
    print("  PASS: R4 birth=2|4 (distinguishable from Standard)")

    # Test 4: R9 — odd step should be frozen
    some_cells = frozenset([(1, 1), (1, 2), (2, 1)])
    frozen = step(some_cells, 5, 5, "R9", 1)  # step 1 = odd = no change
    assert frozen == some_cells, "FAIL: R9 odd step should freeze"
    print("  PASS: R9 temporal delay (odd step frozen)")

    # Test 5: R10 — step 2 (even) uses R1 rules
    # Dead cell with 3 live neighbors → should NOT be born (R1 requires 4)
    three_neighbors = frozenset([(0, 1), (1, 0), (2, 1)])
    result = step(three_neighbors, 3, 3, "R10", 2)  # even step → R1
    assert (1, 1) not in result, "FAIL: R10 even step should use R1 (birth=4)"
    print("  PASS: R10 even step uses R1 rules")

    # Test 6: Evaluator
    assert evaluate_submission("[(0, 1), (1, 2)]", "[(0, 1), (1, 2)]") == 1.0
    assert evaluate_submission("[(0, 1), (1, 2)]", "[(0,1),(1,2)]")    == 1.0
    assert evaluate_submission("[(0, 1), (1, 2)]", "[(0, 1)]")         == 0.0
    print("  PASS: Evaluator (exact match + space stripping)")

    # Test 7: Pattern placement — Blinker centered on 7x7
    blinker_cells  = NAMED_PATTERNS["Blinker"]["cells"]
    placed_center  = place_pattern(blinker_cells, "center", 7, 7)
    placed_topleft = place_pattern(blinker_cells, "top_left", 7, 7)
    assert len(placed_center)  == 3, "FAIL: Blinker center placement lost cells"
    assert len(placed_topleft) == 3, "FAIL: Blinker top_left placement lost cells"
    assert placed_center != placed_topleft, "FAIL: Placements should differ"
    print("  PASS: Pattern placement (Blinker center vs top_left)")

    # Test 8: Block under R1 — Block is a still life under Standard (all cells survive,
    # no new births). Under R1 (birth=4), it should STILL be a still life because
    # no dead neighbor has exactly 4 live neighbors around a 2x2 block.
    block_placed = place_pattern(NAMED_PATTERNS["Block"]["cells"], "center", 7, 7)
    block_r1     = step(block_placed, 7, 7, "R1", 1)
    block_std    = step(block_placed, 7, 7, "Standard", 1)
    assert block_std == block_placed, "FAIL: Block should be still life under Standard"
    # Under R1 block still survives (2-3 neighbors → still in survival {2,3})
    assert block_r1 == block_placed, "FAIL: Block should survive under R1 (survival unchanged)"
    print("  PASS: Block still life (Standard and R1 survival)")

    # Test 9: Glider under R5 (birth=2) — should produce MORE births than standard
    glider_placed = place_pattern(NAMED_PATTERNS["Glider"]["cells"], "top_left", 9, 9)
    glider_r5_1   = step(glider_placed, 9, 9, "R5", 1)
    glider_std_1  = step(glider_placed, 9, 9, "Standard", 1)
    # R5 births with 2 neighbors too → generally more live cells
    print(f"  INFO: Glider step1 — Standard:{len(glider_std_1)} cells, R5:{len(glider_r5_1)} cells")
    assert glider_r5_1 != glider_std_1, "FAIL: R5 Glider should differ from Standard"
    print("  PASS: Glider diverges under R5 (mutation changes evolution)")

    # Test 10: Pattern task count
    pattern_tasks = generate_pattern_tasks()
    assert len(pattern_tasks) == 150, f"FAIL: Expected 150 pattern tasks, got {len(pattern_tasks)}"
    print(f"  PASS: Pattern task count = {len(pattern_tasks)}")

    print(f"\nAll tests passed. No errors." if errors == 0 else f"\n{errors} test(s) FAILED.")
    return errors == 0


if __name__ == "__main__":
    ok = run_tests()
    if ok:
        print()
        generate_dataset()