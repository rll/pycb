from scipy.misc import imread
from pycb import extract_chessboards, draw_boards

img = imread("scene1.jpg")
corners, chessboards = extract_chessboards(img)
draw_boards(img, corners, chessboards)
