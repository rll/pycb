#cython: boundscheck=False, wraparound=False

from __future__ import division

import numpy as np
cimport numpy as np

cdef inline double dbl_max(double a, double b): return a if a >= b else b
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

cdef extern from "math.h":
    double sqrt(double x)
    double fabs(double x)
    double cos(double x)
    double sin(double x)
    double exp(double x)

cdef double pi = np.pi
cdef double NORM_PDF_C = 1.0/sqrt(2*pi)
cdef inline double normpdf(double x, double mu, double sigma):
    return NORM_PDF_C*(1.0/sigma)*exp((-(x-mu)**2)/(2.0*sigma**2))

cdef double structure_energy(np.ndarray[np.float_t, ndim=2] x):
    #norm(x[0,:] + x[2,:] -2*x[1,:]) / norm(x[0,:]-x[2,:])
    cdef double t1 = 0
    cdef double t2 = 0
    t1 = t2 = 0

    cdef int i

    for i in range(2):
        t1 += (x[0,i] + x[2,i] - 2*x[1,i])**2
        t2 += (x[0,i] - x[2,i])**2
    if t2 == 0:
        return 10000.0
    return sqrt(t1) / sqrt(t2)

def chessboard_energy(np.ndarray[np.int_t, ndim=2] chessboard, np.ndarray[np.float_t, ndim=2] corners):

    cdef int num_corners = chessboard.shape[0] * chessboard.shape[1]

    # energy: number of corners
    cdef double E_corners = -num_corners

    # energy: structure
    cdef double E_structure = 0

    cdef np.ndarray x
    cdef double e_s

    cdef int j, k

    # walk through rows
    for j in range(chessboard.shape[0]):
        for k in range(chessboard.shape[1]-2):
            x = corners[chessboard[j,k:k+3],:]
            e_s = structure_energy(x)
            E_structure = dbl_max(E_structure, e_s)

    # walk through columns
    for j in range(chessboard.shape[1]):
        for k in range(chessboard.shape[0]-2):
            x = corners[chessboard[k:k+3,j],:];
            e_s = structure_energy(x)
            E_structure = dbl_max(E_structure, e_s)

    # final energy
    return E_corners + num_corners*E_structure

def rel_pixel_distance(int u, int v, int cu, int cv, np.ndarray[np.float_t, ndim=1] vx):
    ## compute rel. position of pixel and distance to vectors
    #w  = np.array([u, v])-np.array([cu, cv])
    #d1 = norm(w-w.dot(v1[i,np.newaxis].T.dot(v1[i,np.newaxis])))
    cdef double w[2] 
    w[0] = u - cu
    w[1] = v - cv
    cdef double vx11 = vx[0] * vx[0]
    cdef double vx12 = vx[0] * vx[1]
    cdef double vx22 = vx[1] * vx[1]
    cdef double d
    d = (w[0] - w[0] * vx11 - w[1] * vx12)**2
    d += (w[1] - w[0] * vx12 - w[2] * vx22)**2
    return sqrt(d)

def refine_orientation(int cu, int cv, int r, int width, int height, 
                       np.ndarray[np.float_t, ndim=1] norm_oris, 
                       np.ndarray[np.float_t, ndim=2] oris, 
                       np.ndarray[np.float_t, ndim=2] img_du, 
                       np.ndarray[np.float_t, ndim=2] img_dv, 
                       np.ndarray[np.float_t, ndim=1] v1, 
                       np.ndarray[np.float_t, ndim=1] v2):

    cdef np.ndarray[np.float_t, ndim=2] A1 = np.zeros((2,2))
    cdef np.ndarray[np.float_t, ndim=2] A2 = np.zeros((2,2))

    cdef int ori_idx = -1

    cdef double prod_mag = 0

    cdef double du = 0, dv = 0, dudu = 0, dudv = 0, dvdv = 0

    cdef int u, v

    for u in range(int_max(cu-r,0), min(cu+r+1,width)):
        for v in range(int_max(cv-r,0), min(cv+r+1,height)):

            ori_idx += 1

            if  norm_oris[ori_idx] < 0.1:
                continue

            prod_mag = fabs(oris[ori_idx, 0] * v1[0] + oris[ori_idx, 1] * v1[1])

            du = img_du[v,u]
            dv = img_dv[v,u]
            dudu = du*du
            dudv = du*dv
            dvdv = dv*dv

            # robust refinement of orientation 1
            if prod_mag < 0.25: # inlier?
                A1[0,0] += dudu
                A1[0,1] += dudv
                A1[1,0] += dudv
                A1[1,1] += dvdv
                #A1[0,:] += img_du[v,u] * np.array([img_du[v,u],img_dv[v,u]])
                #A1[1,:] += img_dv[v,u] * np.array([img_du[v,u],img_dv[v,u]])

            prod_mag = fabs(oris[ori_idx, 0] * v2[0] + oris[ori_idx, 1] * v2[1])

            # robust refinement of orientation 2
            if prod_mag < 0.25: # inlier?
                A2[0,0] += dudu
                A2[0,1] += dudv
                A2[1,0] += dudv
                A2[1,1] += dvdv
                #A2[0,:] += img_du[v,u] * np.array([img_du[v,u],img_dv[v,u]])
                #A2[1,:] += img_dv[v,u] * np.array([img_du[v,u],img_dv[v,u]])

    return A1, A2

cdef class Template(object):

    cdef public np.ndarray a1, a2, b1, b2

    def __init__(self, double angle_1, double angle_2, int radius):
        cdef int width  = radius*2+1
        cdef int height = radius*2+1
        
        cdef np.ndarray[np.float_t, ndim=2] a1, a2, b1, b2

        a1 = np.zeros((height, width))
        a2 = np.zeros((height, width))
        b1 = np.zeros((height, width))
        b2 = np.zeros((height, width))

        # midpoint
        cdef int mu = radius
        cdef int mv = radius

        # compute normals from angles
        cdef double n1[2] 
        cdef double n2[2] 
        n1[0] = -np.sin(angle_1)
        n1[1] = np.cos(angle_1)
        n2[0] = -np.sin(angle_2)
        n2[1] = np.cos(angle_2)

        cdef int idx
        cdef double x0, x1, dist, value, s1, s2
        cdef double a1sum = 0, a2sum = 0, b1sum = 0, b2sum = 0

        cdef int u, v

        for u in range(width):
            for v in range(height):
                idx = v * width + u
                x0 = u-mu
                x1 = v-mv
                dist = sqrt((x0)**2 + (x1)**2)
                value = normpdf(dist, 0, radius/2)
                s1 = x0 * n1[0] + x1 * n1[1]
                s2 = x0 * n2[0] + x1 * n2[1]

                if s1 <=-0.1:
                    if s2 <=-0.1:
                        a1[v, u] = value
                        a1sum += value
                    elif s2 >=0.1:
                        b1[v, u] = value
                        b1sum += value
                elif s1 >=0.1:
                    if s2 >=0.1:
                        a2[v, u] = value
                        a2sum += value
                    elif s2 <=-0.1:
                        b2[v, u] = value
                        b2sum += value

        a1 /= a1sum
        a2 /= a2sum
        b1 /= b1sum
        b2 /= b2sum

        self.a1 = a1
        self.a2 = a2
        self.b1 = b1
        self.b2 = b2
