import pickle
import time
from copy import deepcopy
from collections import deque, defaultdict
from itertools import combinations
from functools import reduce
from io import BytesIO

import numpy as np
from PIL import Image
from hexalattice import hexalattice
from matplotlib import pyplot as plt
from termcolor import cprint
from test_cases import lattices

GUESSES = 0
CALLS = 0
CACHE = dict()


class Hidato:
    def __init__(self, lattice, domains=None, assignment=None, initial_cells=None, anchors=None, distance=None):
        self.lattice = mydeepcopy(lattice)
        self.height = len(lattice)
        self.min_width = len(lattice[0])
        self.max_value = sum(self.min_width + i for i in range(self.height // 2 + 1)) * 2 - len(
            self.lattice[self.height // 2])
        self.variables = [i for i in range(1, self.max_value+1)]
        self.cells = {(i, j) for i in range(self.height) for j in range(len(self.lattice[i]))}

        if initial_cells is None:
            self.initial_cells = {cell for cell in self.cells if self.lattice[cell[0]][cell[1]] != 0}
        else:
            self.initial_cells = initial_cells.copy()

        if assignment is None:
            self.assignment = {}
            for i, j in self.cells:
                if self.lattice[i][j] != 0:
                    self.assignment[self.lattice[i][j]] = (i, j)
        else:
            self.assignment = assignment.copy()

        if anchors is None:
            self.anchors = self.get_anchors()
        else:
            self.anchors = mydeepcopy(anchors)

        if distance is None:
            self.distance = self.get_distances_from_anchors()
        else:
            self.distance = mydeepcopy(distance)

        if domains is None:
            self.domains = {n: (self.cells - set(self.assignment.values())).copy()
                            if n not in self.assignment else {self.assignment[n]}
                            for n in self.variables}
            make_inferences(self)
        else:
            self.domains = mydeepcopy(domains)

    def __eq__(self, other):
        return self.lattice == other.lattice

    def assign(self, n, cell, get_anchors=False, get_distance_from_anchors=False):
        """Assign value cell to variable n and consequently:
            1) Update the lattice at cell to be n
            2) Add n to assignment
            3) Update the domain of n to be {cell} and remove cell from the domains of all other variables
            4) If get_anchors flag is True: Update the anchors
            5) If get_distance_from_anchors is True: Update distance to anchors

            returns True if assignment was successful or False if unsuccessful (found contradiction)"""

        self.lattice[cell[0]][cell[1]] = n
        self.assignment[n] = cell
        self.domains[n] = {cell}
        for var in self.domains:
            if var != n and cell in self.domains[var]:
                self.domains[var].remove(cell)
                if self.domains[var] == set():
                    return False
        if get_anchors:
            self.anchors = self.get_anchors()
        if get_distance_from_anchors:
            self.distance = self.get_distances_from_anchors()
        return True

    def get_anchors(self):
        """return a dictionary that for every unassigned number keeps the anchors of that number. the anchors of
        of a number are the closet assigned numbers from below and from above"""
        anchors = dict()
        curr_anchor = None
        for n in range(1, self.max_value+1):
            if n not in self.assignment:
                anchors[n] = [curr_anchor]
            else:
                curr_anchor = n
        curr_anchor = None
        for n in range(self.max_value, 0, -1):
            if n not in self.assignment:
                anchors[n].append(curr_anchor)
            else:
                curr_anchor = n
        return anchors

    def get_distances_from_anchors(self):
        """Returns a dictionary that maps every anchor to another dictionary of all reachable cells from anchor
        and the distance from anchors to the cell"""

        distance = dict()
        all_anchors = set([anchor for anchors_pair in self.anchors.values() for anchor in anchors_pair
                           if anchor is not None])
        for anchor in all_anchors:
            distance[anchor] = self.get_distances_from_cell(self.assignment[anchor])

        return distance

    def get_distances_from_cell(self, s):
        """return a dictionary mapping every cell reachable from s using only empty cells to the distance to that cell
        from s"""

        distance = {s: 0}
        queue = deque([s])

        while queue:
            v = queue.popleft()
            for u in self.neighbors(v):
                if u not in distance and self.lattice[u[0]][u[1]] == 0:
                    distance[u] = distance[v] + 1
                    queue.append(u)
        del distance[s]
        return distance

    def print_cell(self, cell, end='\n'):
        on_color = 'on_blue' if cell in self.initial_cells else 'on_white'
        value = self.lattice[cell[0]][cell[1]]
        value_str = '  ' if value == 0 else (value if len(str(value)) == 2 else ' ' + str(value))
        cprint(value_str, on_color=on_color, end=end)

    def print(self):
        n_spaces = self.height // 2
        for i, row in enumerate(self.lattice):
            print(n_spaces * '  ', end='')
            for j, value in enumerate(row):
                if j != len(row) - 1:
                    self.print_cell((i, j), end='  ')  # print(cell if len(str(cell))==2 else f'0{cell}',end='  ')
                else:
                    self.print_cell((i, j))  # print(cell if len(str(cell))==2 else f'0{cell}')
            if i < self.height // 2:
                n_spaces -= 1
            else:
                n_spaces += 1

    def plot(self, initial_cells_only=False, return_as_image=False):
        lattice_flattened = [((i, len(row) - 1 - j), n)
                             for i, row in enumerate(self.lattice)
                             for j, n in enumerate(row[::-1])]
        num_cells = len(lattice_flattened)
        min_width = len(self.lattice[0])
        r = min_width - 1
        colors = np.ones((num_cells, 3))
        for i in range(num_cells):
            if lattice_flattened[i][0] in self.initial_cells:
                colors[i] = [193 / 255, 252 / 255, 239 / 255]
        hex_centers, _ = hexalattice.create_hex_grid(nx=3 * r,
                                                     ny=3 * r,
                                                     crop_circ=r,
                                                     face_color=colors[::-1],
                                                     do_plot=True)
        hex_centers = hex_centers[::-1]
        for i, hex_center in enumerate(hex_centers):
            if (lattice_flattened[i][1] != 0
                    and (not initial_cells_only or
                         (initial_cells_only and lattice_flattened[i][0] in self.initial_cells))):
                plt.text(hex_center[0], hex_center[1], lattice_flattened[i][1], fontsize=15, ha='center', va='center')
        plt.axis('off')
        if return_as_image:
            buf = BytesIO()
            plt.savefig(buf,bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            return img
        else:
            plt.show()

    def neighbors(self, cell):
        """Returns a list containing all neighbors of cell"""
        i, j = cell
        if (not (0 <= i < self.height)) or (not (0 <= j < len(self.lattice[i]))):
            raise ValueError(f" cell {cell} does not exist")

        ngbs = []
        if j > 0:
            ngbs.append((i, j - 1))
        if j < len(self.lattice[i]) - 1:
            ngbs.append((i, j + 1))

        if i > 0:  # Add neighbors from row above
            if i <= self.height // 2:
                if j - 1 >= 0:
                    ngbs.append((i - 1, j - 1))
                if j < len(self.lattice[i - 1]):
                    ngbs.append((i - 1, j))
            else:
                ngbs.append((i - 1, j))
                ngbs.append((i - 1, j + 1))

        if i < self.height - 1:  # Add neighbors from row below
            if i < self.height // 2:
                ngbs.append((i + 1, j))
                ngbs.append((i + 1, j + 1))
            else:
                if j - 1 >= 0:
                    ngbs.append((i + 1, j - 1))
                if j < len(self.lattice[i + 1]):
                    ngbs.append((i + 1, j))

        return ngbs

    def copy(self):
        return Hidato(self.lattice, self.domains, self.assignment, self.initial_cells, self.anchors, self.distance)


def solved(hidato):
    """Check whether the hidato is solved. Defining N as the number of cells in the hidato lattice then A Hidato is
     solved if all cells are filled with numbers from 1 to N and there is a connected path through neighbors path from 1
     to N"""
    return (all(n in hidato.assignment for n in hidato.variables)
            and all(hidato.assignment[n] in hidato.neighbors(hidato.assignment[n - 1])
                    for n in range(2, hidato.max_value + 1)))


def select_unassigned_variable(hidato):
    """Return an unassigned variable using the Most Restricted Variable heuristic"""
    return min([var for var in hidato.variables
                if var not in hidato.assignment],
               key=lambda var: len(hidato.domains[var]))


def order_domain_values(var, hidato):
    """Return the domain of var"""
    return hidato.domains[var]


def possible(cell, n, hidato):
    """Check if it is possible to assign n to cell. That is, the cell is empty and  if n-1 or n+1 are assigned,
    they must be in cells adjacent to cell"""
    if hidato.lattice[cell[0]][cell[1]] != 0:
        return False

    if n-1 in hidato.assignment and hidato.assignment[n-1] not in hidato.neighbors(cell):
        return False

    if n+1 in hidato.assignment and hidato.assignment[n+1] not in hidato.neighbors(cell):
        return False
    return True


def reduce_domains_by_distance(hidato):
    """
    Go over all unassigned variables. for every unassigned variable m, go over all its possible cells in domain
    and keep in domain only cells where, for each anchor of n, the cell is reachable from the anchor and
    0 < distance[cell_anchor][cell_n] <= abs(anchor-n). If upon reducing the domains a variable is left with an
    empty domain, then it means there is a contradiction. If a variable is left with a single value in its domain
    then, assuming there is no contradiction, this is a new inference that means we can later assign the variable to the
    this value.

    Arguments:
    hidato - an instance of class Hidato representing a hidato puzzle

    Returns:
    contradiction - boolean indicating whether hidato contains a contradiction, i.e, it is unsolvable
    new_inferences - dictionary of n:cell pairs meaning that assuming contradiction is false then
                     value cell can be assigned to variable n for every n:cell pair in new_inferences.
    """
    contradiction = False
    new_inferences = dict()
    new_cells = set()
    for m in hidato.variables:
        if m not in hidato.assignment:
            hidato.domains[m] = set(filter(lambda c: all([c in hidato.distance[a]
                                                          and 0 < hidato.distance[a][c] <= abs(a - m)
                                                          for a in hidato.anchors[m] if a is not None]),
                                           hidato.domains[m]))
            if len(hidato.domains[m]) == 1:
                m_cell = list(hidato.domains[m])[0]
                if m_cell in new_cells:
                    contradiction = True
                    break
                new_inferences[m] = m_cell
                new_cells.add(m_cell)
            if hidato.domains[m] == set():
                contradiction = True
                break
    return contradiction, new_inferences


def get_cell_domains_of_empty_cells(hidato):
    empty_cells = hidato.cells - set(hidato.assignment.values())
    cell_domains = defaultdict(set)

    for cell in empty_cells:
        for n, domain in hidato.domains.items():
            if cell in domain:
                cell_domains[cell].add(n)

    return cell_domains


def find_naked_cells(hidato):
    """Make new inferences by looking at the domains and finding cells which only appear in a domain of a single
    variable n.

    Arguments:
    hidato - an instance of class Hidato representing a hidato puzzle

    Returns:
    contradiction - boolean indicating whether hidato contains a contradiction, i.e, it is unsolvable
    new_inferences - dictionary of n:cell pairs meaning that assuming contradiction is false then
                     value cell can be assigned to variable n for every n:cell pair in new_inferences.
    """

    cell_domains = get_cell_domains_of_empty_cells(hidato)
    new_inferences = dict()
    for cell, cell_domain in cell_domains.items():
        if cell_domain == set():
            return True, new_inferences
        elif len(cell_domain) == 1:
            n = list(cell_domain)[0]
            new_inferences[n] = cell

    return False, new_inferences


def make_inferences(hidato):
    """
    Make inferences from current state of hidato and return whether a contradiction was found when making the
    inferences.

    Arguments:
    hidato - an instance of class Hidato representing a hidato puzzle

    Returns:
    contradiction - boolean indicating whether hidato contains a contradiction, i.e, it is unsolvable
        """
    while True:
        new_assignments = dict()
        contradiction, assignments = reduce_domains_by_distance(hidato)
        if contradiction:
            return contradiction
        new_assignments.update(assignments)
        contradiction, assignments = find_naked_cells(hidato)
        if contradiction:
            return contradiction
        new_assignments.update(assignments)
        # contradiction, assignments = hidden_subset(hidato, 2)
        # if contradiction:
        #     return contradiction
        # new_assignments.update(assignments)
        # naked_subset(hidato, n=2)
        if new_assignments:
            for var, val in new_assignments.items():
                if not hidato.assign(var, val):
                    return True
            hidato.anchors = hidato.get_anchors()
            hidato.distance = hidato.get_distances_from_anchors()
        else:
            return False


def hidden_subset(hidato, n):
    """Go over all possible subsets of size n of the unassigned numbers (Here, by unassigned I mean has more than 1 cell
     in its domain since there may be numbers with just a single cell in they're domain that are not formally assigned
     yet). If the union of the domains of the numbers in the subset has n cells, remove these n cells from the domains
     of all other numbers"""

    contradiction = False
    new_inferences = dict()
    new_cells = set()
    candidates = {var for var, domain in hidato.domains.items() if len(domain) > 1}
    # if n == 2:
    #     subsets = [(m-1, m) for m in candidates if m>1 and m-1 in candidates]
    # else:
    subsets = combinations(candidates, n)
    
    for subset in subsets:
        subset_domain_union = reduce(lambda d1, d2: d1 | d2, [hidato.domains[candidate] for candidate in subset])
        if len(subset_domain_union) == n:
            for candidate in candidates:
                if candidate not in subset:
                    hidato.domains[candidate] -= subset_domain_union
                    if len(hidato.domains[candidate]) == 1:
                        candidate_cell = list(hidato.domains[candidate])[0]
                        if candidate_cell in new_cells:
                            contradiction = True
                            break
                        new_inferences[candidate] = candidate_cell
                        new_cells.add(candidate_cell)
                    if hidato.domains[candidate] == set():
                        contradiction = True
                        break
    return contradiction, new_inferences


def naked_subset(hidato, n):
    """We find naked subsets by going over all subset of empty cells (cells) (by empty we mean cells that currently have
    more than 1 number in they're domains. for each subset of cells we look to see at the union the domains of numbers
    for each cell in the subset. If the length of the union is equal to n, Remove all other cells from all numbers in
    union)"""
    cell_domains = get_cell_domains_of_empty_cells(hidato)

    candidates = {cell for cell, domain in cell_domains.items() if len(domain) > 1}
    subsets = combinations(candidates, n)
    for subset in subsets:
        subset_domain_union = reduce(lambda d1, d2: d1 | d2, [cell_domains[candidate] for candidate in subset])
        if len(subset_domain_union) == n:
            for number in subset_domain_union:
                hidato.domains[number] = hidato.domains[number] & set(subset)


def report_hidden_subsets(hidato, n):
    """Find the number of hidden subsets of size n and print it"""

    hidden_subsets_found = dict()
    candidates = {var for var, domain in hidato.domains.items() if len(domain) > 1}
    subsets = combinations(candidates, n)
    for subset in subsets:
        subset_domain_union = reduce(lambda d1, d2: d1 | d2, [hidato.domains[candidate] for candidate in subset])
        if len(subset_domain_union) == n:
            hidden_subsets_found[subset] = subset_domain_union
    if hidden_subsets_found:
        subset_cells = list(next(iter(hidden_subsets_found.values())))
        all_neighbors = all(any(cell2 in hidato.neighbors(cell1) for cell2 in subset_cells) for cell1 in subset_cells)
        print(f"Found {hidden_subsets_found} hidden subsets of size {n}. They are all neighbors -> {all_neighbors}")
        

def report_naked_subsets(hidato, n):
    """Find the number of naked subsets of size n and print it."""

    naked_subsets_found = dict()
    cell_domains = get_cell_domains_of_empty_cells(hidato)

    candidates = {cell for cell, domain in cell_domains.items() if len(domain) > 1}
    subsets = combinations(candidates, n)
    for subset in subsets:
        subset_domain_union = reduce(lambda d1, d2: d1 | d2, [cell_domains[candidate] for candidate in subset])
        if len(subset_domain_union) == n:
            naked_subsets_found[subset] = subset_domain_union
    if naked_subsets_found:
        subset_cells = list(next(iter(naked_subsets_found.keys())))
        all_neighbors = all(any(cell2 in hidato.neighbors(cell1) for cell2 in subset_cells) for cell1 in subset_cells)
        print(f"Found {naked_subsets_found} naked subsets of size {n}. They are all neighbors -> {all_neighbors}")


def solve(hidato):
    global GUESSES
    global CALLS
    CALLS += 1
    if solved(hidato):
        print("solved!")
        return hidato
    var = select_unassigned_variable(hidato)
    if len(hidato.domains[var]) > 1:
        GUESSES += 1
    for val in order_domain_values(var, hidato):
        if possible(val, var, hidato):
            new_hidato = hidato.copy()
            if not new_hidato.assign(var, val, get_anchors=True, get_distance_from_anchors=True):
                continue
            contradiction = make_inferences(new_hidato)
            if contradiction:
                continue
            result = solve(new_hidato)
            if result is not None:
                return result
    return None


def valid(hidato):
    """Does a simple check of hidato validity. Checks to see if all the values in the lattice cells are unique.
    and for every cell with number n verifys that if n-1 or n+1 are assigned then they are neibhors of cell"""

    vals = [n for row in hidato.lattice for n in row if n != 0]
    if len(set(vals)) != len(vals):
        return False

    for n in hidato.assignment:
        if n-1 in hidato.assignment and hidato.assignment[n-1] not in hidato.neighbors(hidato.assignment[n]):
            return False
        if n+1 in hidato.assignment and hidato.assignment[n+1] not in hidato.neighbors(hidato.assignment[n]):
            return False

    return True
# TODO - maybe incorporate naked pairs/triples/quadruplets etc.. instead of just naked singles.


def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


def solve_with_cache(lattice):
    lattice_tuple = to_tuple(lattice)
    if lattice_tuple in CACHE:
        return CACHE[to_tuple(lattice)]
    hidato = Hidato(lattice)
    if not valid(hidato):
        raise ValueError("Unvalid hidato, cannot solve")
    solution = solve(hidato)
    solution_lattice = solution.lattice if solution is not None else None
    CACHE[lattice_tuple] = solution_lattice
    return solution_lattice


def get_model_benchmark(f):
    benchmark = []
    for i in range(6):
        puzzle = Hidato(lattices[f'l{i}'])
        t0 = time.perf_counter()
        f(puzzle)
        t1 = time.perf_counter()
        benchmark.append(t1-t0)
    return benchmark


def mydeepcopy(obj):
    try:
        return pickle.loads(pickle.dumps(obj, -1))
    except pickle.PicklingError:
        return deepcopy(obj)


def main():
    puzzle = Hidato(lattices['l6'])
    puzzle.plot(initial_cells_only=False)
    t0 = time.perf_counter()
    solution = solve(puzzle)
    t1 = time.perf_counter()
    print(f"Solved the hidato In {t1-t0} seconds using {GUESSES} guesses and {CALLS} calls")
    solution.plot()


if __name__ == '__main__':
    main()
