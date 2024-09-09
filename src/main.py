#from numpy import average, save
from image import *
from preprocess import *
from multiprocess import *
from grid_search import grid_search


def main():
    # TODO: add a way to select the output format
    # TODO: add a way to select the input folder
    # TODO: add a way to select aligning algorithm
    # TODO: add a way to select stacking algorithm
    
    # grid search
    grid_search()
     
if __name__ == '__main__':
    import main
    main.main()