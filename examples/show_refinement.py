from scipy.misc import imread
from pycb import extract_chessboards, draw_boards

img = imread("scene1.jpg")
corners, chessboards, unrefined = extract_chessboards(img, include_unrefined=True)
draw_boards(img, corners, chessboards, unrefined)
