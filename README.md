# Sudoku CV Solver

A sudoku pipeline implemented for a [Blitz#3 competition](https://www.aicrowd.com/challenges/ai-for-good-ai-blitz-3/problems/sudoku) with score 97.8% accuracy.

I tried training a small CNN with MNIST dataset but it did not generalize well to the digits used in the Sudoko, so I created a dataset from the [Competition Train dataset](https://www.aicrowd.com/challenges/ai-for-good-ai-blitz-3/problems/sudoku/dataset_files) which I could fit with 0.997 val accuracy which has resulted to 0.978 accuracy over testing dataset.

I have also implemented a simple algorithm to solve the sudoku grid which searches for the most occupied part of the grid and fills the empty cells by trying different permutations over the missng numbers while backtracking when the grid is not valid.

This solution can't handle realworld sudoku images (from newspapers) but it would be possible to add an extra stage to the beginning of the pipeline, However I was just interested to implement it for practice.

## Digits extraction pipeline

* Thresholding
* Morphological opening, finding countour of the grid and then applying rotation.
* Removing gridlines using 1D morphological kernels.
* extracting digits by dividing the grid. (with extra 4 pixels as border to make sure the cell fits the number)
* refining the digits by finding contours and perspective transformation.

(check the review jupyternotebook)

![Pipeline](https://i.ibb.co/gvzvFpJ/final-pipeline.png)

## Final Notes

I would appreciate any advice regarding my code, different techniques or my presentation.
