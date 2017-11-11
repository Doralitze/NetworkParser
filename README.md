# NetworkParser
A tool to 3d render a network from a csv file

## PyPy dependancies
 * vispy
 * numpy

## Python version
Python >= 3.6 is required.

## File format
Each row of the csv file should have its items in the following order:
1. node name
2. x-coordinate
3. y-coordinate
4. z-coordinate
All elements after these will be node names of connected nodes.
