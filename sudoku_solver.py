import numpy as np

from itertools import permutations 

class SudokuSolver:
    def __init__(self, sudoku_grid):
        """
        sudoku_grid: flat L[81] representing a sudoku grid
                    zeros represent empty cells
        """
        self.grid = np.reshape(sudoku_grid, (9, 9))
    
    def solve(self):
        return self._recurr_solve(self.grid)

    def _recurr_solve(self, grid):
        if not self._verify(grid):
            return None
        if self._test_finished(grid):
            return grid
        indc, zeros = self._get_fullest_part(grid)
        missing_nums = np.setdiff1d(np.arange(1, 10), np.unique(grid[indc]))
        
        for perm in permutations(missing_nums):
            perm = list(perm)
            curr_g = np.copy(grid)
            pos = 0
            for i in zip(*indc):
                if curr_g[i] == 0:
                    curr_g[i] = perm[pos]
                    pos += 1
            sol = self._recurr_solve(curr_g)
            del curr_g
            if sol is not None:
                return sol
        return None
        

    def _get_fullest_part(self, grid):
        """
        returns the indicies of the fullest part of the grid
        so it faster to solve (less permutations)
        """
        min_zeros = 10
        min_ind = None
        for i in range(9):
            c = grid[:, i]
            zeros = np.sum(c == 0)
            if zeros < min_zeros and zeros != 0:
                min_zeros = zeros
                min_ind = [(j, i) for j in range(9)]
            
        for i in range(9):
            c = grid[i, :]
            zeros = np.sum(c == 0)
            if zeros < min_zeros and zeros != 0:
                min_zeros = zeros
                min_ind = [(i, j) for j in range(9)]
        
        for i in range(9):
            ir = (i%3)*3
            ic = (i//3)*3
            c = grid[ir:ir+3, ic:ic+3]
            zeros = np.sum(c == 0)
            if zeros < min_zeros and zeros != 0:
                min_zeros = zeros
                min_ind = [(ir + j%3, ic + j//3) for j in range(9)]
        
        min_ind = tuple([list(p) for p in zip(*min_ind)])
            
        return min_ind, min_zeros


    def _verify(self, grid):
        return self._verify_c(grid) and self._verify_r(grid) and self._verify_b(grid)

    def _verify_c(self, grid):
        for i in range(9):
            c = grid[:, i]
            c = c[c!=0]
            n_unique = len(np.unique(c))
            if len(c) != n_unique:
                return False
        return True

    def _verify_r(self, grid):
        for i in range(9):
            c = grid[i, :]
            c = c[c!=0]
            n_unique = len(np.unique(c))
            if len(c) != n_unique:
                return False
        return True
    
    def _verify_b(self, grid):
        for i in range(9):
            ir = (i%3)*3
            ic = (i//3)*3
            c = grid[ir:ir+3, ic:ic+3]
            
            c = c[c!=0]
            n_unique = len(np.unique(c))
            if len(c) != n_unique:
                return False
        return True
    
    def _test_finished(self, grid):
        """
        checks if there's no empty cells anymore 
        but doesn't check the validity of the grid
        """
        zeros = np.sum(grid==0)
        if zeros == 0:
            return True
        return False
    