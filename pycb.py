import numpy as np
import scipy as sp
from numpy import pi
from numpy.linalg import norm
from scipy.misc import imresize
from scipy import ndimage
from scipy.signal import convolve2d
import c_pycb

## theano ## 
import theano
from theano.tensor.signal.conv import conv2d
####

import time

#class Template(object):
#
#    def __init__(self, angle_1, angle_2, radius):
#        width  = radius*2+1
#        height = radius*2+1
#
#        if int(width) != width or int(height) != height:
#            raise Exception("bad width/height")
#        else:
#            width = int(width)
#            height = int(height)
#
#        self.a1 = np.zeros((height, width), dtype=np.double)
#        self.a2 = np.zeros((height, width), dtype=np.double)
#        self.b1 = np.zeros((height, width), dtype=np.double)
#        self.b2 = np.zeros((height, width), dtype=np.double)
#
#        # midpoint
#        mu = radius
#        mv = radius
#
#        # compute normals from angles
#        n1 = np.array([[-np.sin(angle_1)],  [np.cos(angle_1)]])
#        n2 = np.array([[-np.sin(angle_2)],  [np.cos(angle_2)]])
#
#        vecs = np.empty((width*height, 2))
#        for u in range(width):
#            for v in range(height):
#                idx = v * width + u
#                vecs[idx, 0] = u-mu
#                vecs[idx, 1] = v-mv
#
#        dist = np.sqrt(np.sum(vecs**2,axis=-1))
#        s1 = vecs.dot(n1)
#        s2 = vecs.dot(n2)
#        values = normpdf(dist, 0, radius/2)
#        
#        for u in range(width):
#            for v in range(height):
#                idx = v * width + u
#                if s1[idx]<=-0.1 and s2[idx]<=-0.1:
#                    self.a1[v, u] = values[idx]
#                elif s1[idx]>=0.1 and s2[idx]>=0.1:
#                    self.a2[v, u] = values[idx]
#                elif s1[idx]<=-0.1 and s2[idx]>=0.1:
#                    self.b1[v, u] = values[idx]
#                elif s1[idx]>=0.1 and s2[idx]<=-0.1:
#                    self.b2[v, u] = values[idx]
#
#        #n1 = np.array([-np.sin(angle_1),  np.cos(angle_1)])
#        #n2 = np.array([-np.sin(angle_2),  np.cos(angle_2)])
#
#        ## for all points in template do
#        #for u in range(width):
#        #    for v in range(height):
#        #        vec  = np.array([u-mu, v-mv])
#        #        dist = np.sqrt(vec.dot(vec))
#
#        #        # check on which side of the normals we are
#        #        s1 = vec.dot(n1)
#        #        s2 = vec.dot(n2)
#
#        #        if s1<=-0.1 and s2<=-0.1:
#        #            self.a1[v, u] = normpdf(dist, 0, radius/2)
#        #        elif s1>=0.1 and s2>=0.1:
#        #            self.a2[v, u] = normpdf(dist, 0, radius/2)
#        #        elif s1<=-0.1 and s2>=0.1:
#        #            self.b1[v, u] = normpdf(dist, 0, radius/2)
#        #        elif s1>=0.1 and s2<=-0.1:
#        #            self.b2[v, u] = normpdf(dist, 0, radius/2)
#
#        # normalize
#        self.a1 = self.a1/self.a1.sum()
#        self.a2 = self.a2/self.a2.sum()
#        self.b1 = self.b1/self.b1.sum()
#        self.b2 = self.b2/self.b2.sum()

def edge_orientations(img_angle, img_weight):

    v1 = np.zeros(2)
    v2 = np.zeros(2)

    # number of bins (histogram parameter)
    bin_num = 32

    # convert images to vectors
    vec_angle  = img_angle.flatten('F')
    vec_weight = img_weight.flatten('F')

    # convert angles from normals to directions
    vec_angle += pi/2
    vec_angle[vec_angle > pi] -= pi

    # create histogram
    angle_hist = np.zeros(bin_num)
    for i in range(len(vec_angle)):
        bin = max(min(np.floor(vec_angle[i]/(pi/bin_num)),bin_num-1),0)
        angle_hist[bin] += vec_weight[i]

    modes, angle_hist_smoothed = find_modes_mean_shift(angle_hist, 1)

    # if only one or no mode => return invalid corner
    if modes.shape[0] <= 1:
        return (v1, v2)

    # compute orientation at modes and add a third column
    orientations = (modes[:,0]-1)*pi/bin_num
    modes = np.append(modes, orientations[:,np.newaxis], 1)

    # extract 2 strongest modes and sort by angle
    modes = modes[0:2,:]
    modes = modes[modes[:, 2].argsort()]

    # compute angle between modes
    delta_angle = min(modes[1,2] - modes[0,2], modes[0,2] + pi - modes[1,2])

    # if angle too small => return invalid corner
    if delta_angle <= 0.3:
        return (v1, v2)

    # set statistics: orientations
    v1[:] = [np.cos(modes[0,2]), np.sin(modes[0,2])]
    v2[:] = [np.cos(modes[1,2]), np.sin(modes[1,2])]

    return (v1, v2)

NORM_PDF_C = 1/np.sqrt(2*pi)
def normpdf(x, mu, sigma):
    return NORM_PDF_C*(1.0/sigma)*np.exp((-(x-mu)**2)/(2.0*sigma**2))

def find_modes_mean_shift(hist, sigma):

    # compute smoothed histogram
    hist_smoothed = np.zeros(len(hist))
    for i in range(len(hist)):
        j = np.arange(-round(2*sigma), round(2*sigma)+1, dtype=np.int)
        idx = np.mod(i+j, len(hist))
        hist_smoothed[i] = (hist[idx]*normpdf(j, 0, sigma)).sum()

    modes = []

    # check if at least one entry is non-zero
    # (otherwise mode finding may run infinitly)
    if np.all(abs(hist_smoothed-hist_smoothed[0]) < 1e-5):
      return modes, hist_smoothed

    found_bins = set()

    # mode finding
    for i in range(len(hist_smoothed)):
        j = i
        while True:
            h0 = hist_smoothed[j]
            j1 = np.mod(j+1, len(hist))
            j2 = np.mod(j-1, len(hist))
            h1 = hist_smoothed[j1]
            h2 = hist_smoothed[j2]
            if h1>=h0 and h1>=h2:
                j = j1
            elif h2>h0 and h2>h1:
                j = j2
            else:
                break

        if len(modes) == 0 or not j in found_bins:
            found_bins.add(j)
            modes.append((j, hist_smoothed[j]))

    # sort
    modes = np.array(modes)
    idx = np.argsort(modes[:,1], axis=0)[::-1]
    modes = modes[idx]

    return modes, hist_smoothed

def do_refine_corners(img_du, img_dv, img_angle, img_weight, corners, r):

    # image dimensions
    width = img_du.shape[1]
    height = img_du.shape[0]

    # init orientations to invalid (corner is invalid iff orientation=0)
    v1 = np.zeros((corners.shape[0], 2))
    v2 = np.zeros((corners.shape[0], 2))

    refined = np.zeros(corners.shape, dtype=np.float)

    for i in range(corners.shape[0]):

        #print "on corner %d of %d" % (i, corners.shape[0])
        cu = corners[i, 0]
        cv = corners[i, 1]

        # estimate edge orientations
        subwindow_v = slice(max(cv-r, 0), min(cv+r+1, height))
        subwindow_u = slice(max(cu-r, 0), min(cu+r+1, width))
        img_angle_sub  = img_angle[subwindow_v, subwindow_u]
        img_weight_sub = img_weight[subwindow_v, subwindow_u]

        orientations = edge_orientations(img_angle_sub,img_weight_sub)

        v1[i, :] = orientations[0]
        v2[i, :] = orientations[1]

        # continue, if invalid edge orientations
        if np.all(v1[i, :] == 0) or np.all(v2[i, :] == 0):
            continue

        #################################
        # corner orientation refinement #
        #################################

        window_u = slice(max(cu-r,0), min(cu+r+1,width))
        window_v = slice(max(cv-r,0), min(cv+r+1,height))

        sub_du = img_du[window_v,window_u].flatten('F')
        sub_dv = img_dv[window_v,window_u].flatten('F')
        os = np.column_stack([sub_du, sub_dv])
        norm_os = np.sqrt(np.sum(os**2,axis=-1))
        os = os / norm_os[:, np.newaxis]


        A1, A2 = c_pycb.refine_orientation(cu, cv, r, width, height, norm_os, os, img_du, img_dv, v1[i, :], v2[i, :])
        #A1 = np.zeros((2,2))
        #A2 = np.zeros((2,2))
        #o_idx = -1
        #for u in range(max(cu-r,0), min(cu+r+1,width)):
        #    for v in range(max(cv-r,0), min(cv+r+1,height)):
        #        o_idx += 1
        #        # pixel orientation vector
        #        #o = np.array([img_du[v,u], img_dv[v,u]])
        #        #norm_o = np.sqrt(o.dot(o))
        #        #if norm_o != norm_os[o_idx]:
        #        #    print o
        #        #    print norm_o
        #        #    print o/norm_o
        #        #    print os[o_idx]
        #        #    print norm_os[o_idx]
        #        #    asd
        #        if  norm_os[o_idx] < 0.1:
        #            continue
        #        #o = o/norm_o

        #        # robust refinement of orientation 1
        #        if abs(os[o_idx].dot(v1[i,:])) < 0.25: # inlier?
        #            A1[0,:] += img_du[v,u] * np.array([img_du[v,u],img_dv[v,u]])
        #            A1[1,:] += img_dv[v,u] * np.array([img_du[v,u],img_dv[v,u]])

        #        # robust refinement of orientation 2
        #        if abs(os[o_idx].dot(v2[i,:])) < 0.25: # inlier?
        #            A2[0,:] += img_du[v,u] * np.array([img_du[v,u],img_dv[v,u]])
        #            A2[1,:] += img_dv[v,u] * np.array([img_du[v,u],img_dv[v,u]])


        # set new corner orientation
        w1, vec1 = np.linalg.eigh(A1)
        w2, vec2 = np.linalg.eigh(A2)
        v1[i, :] = vec1[:, 0].T
        v2[i, :] = vec2[:, 0].T

        ################################
        #  corner location refinement  #
        ################################

        G = np.zeros((2,2), dtype=np.float)
        b = np.zeros((2,1), dtype=np.float)
        o_idx = -1
        for u in range(max(cu-r,0), min(cu+r+1,width)):
            for v in range(max(cv-r,0), min(cv+r+1,height)):
                o_idx += 1
                # pixel orientation vector
                #o = np.array([img_du[v,u], img_dv[v,u]])
                #norm_o = np.sqrt(o.dot(o))
                if  norm_os[o_idx] < 0.1:
                    continue
                #o = o/norm_o

                # robust subpixel corner estimation

                # do not consider center pixel
                if u!=cu or v!=cv:

                    d1 = c_pycb.rel_pixel_distance(u,v,cu,cv,v1[i])
                    d2 = c_pycb.rel_pixel_distance(u,v,cu,cv,v2[i])

                    ## compute rel. position of pixel and distance to vectors
                    #w  = np.array([u, v])-np.array([cu, cv])
                    #d1 = norm(w-w.dot(v1[i,np.newaxis].T.dot(v1[i,np.newaxis])))
                    #d2 = norm(w-w.dot(v2[i,np.newaxis].T.dot(v2[i,np.newaxis])))

                    # if corresponds with either of the vectors / directions
                    if ((d1 < 3 and abs(os[o_idx].dot(v1[i,:])) < 0.25) or
                        (d2 < 3 and abs(os[o_idx].dot(v2[i,:])) < 0.25)):
                        du = img_du[v,u]
                        dv = img_dv[v,u]
                        dvec = np.array([du, dv])
                        H = dvec[np.newaxis].T.dot(dvec[np.newaxis])
                        G += H
                        b += H.dot(np.array([u, v])[:, np.newaxis])


        # set new corner location if G has full rank
        if np.linalg.matrix_rank(G) == 2:
            corner_pos_old = corners[i,:]
            corner_pos_new = np.linalg.solve(G, b).T
            refined[i,:] = corner_pos_new

            # set corner to invalid, if position update is very large
            if norm(corner_pos_new-corner_pos_old) >= 4:
                v1[i,:] = 0
                v2[i,:] = 0
        else:
            v1[i,:] = 0
            v2[i,:] = 0

    return refined, v1, v2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def non_maximum_supression(img, n, tau, margin):

    height = img.shape[0]
    width = img.shape[1]

    maxima = []

    for i in range(n + margin, width - n - margin, n+1):
        for j in range(n + margin, height - n - margin, n+1):

            max_i = i
            max_j = j
            max_val = img[j,i]

            for i2 in range(i, i + n + 1):
                for j2 in range(j, j + n + 1):
                    currval = img[j2,i2]
                    if currval > max_val:
                        max_i   = i2
                        max_j   = j2
                        max_val = currval

            failed = False
            for i2 in range(max_i - n, min(max_i + n + 1, width - margin)):
                for j2 in range(max_j - n, min(max_j + n + 1, height - margin)):
                    currval = img[j2,i2]
                    not_in_window = (i2 < i or
                                     i2 > (i + n) or
                                     j2 < j or
                                     j2 > (j + n))
                    if currval > max_val and not_in_window:
                        failed = True
                        break
                if failed:
                    break

            if max_val >= tau and not failed:
                maxima.append((max_i, max_j))

    return np.array(maxima)

def score_corners(img,img_angle,img_weight,corners,v1,v2,radius):

    width  = img.shape[1]
    height = img.shape[0]

    scores = np.zeros(len(corners), dtype=np.float)
    # for all corners do
    for i in range(len(corners)):
      # corner location
      u = round(corners[i,0])
      v = round(corners[i,1])

      # compute corner statistics @ radius 1
      for j in range(len(radius)):
          if u >= radius[j] and u < width-radius[j] and v >= radius[j] and v < height-radius[j]:
              sub_v = slice(v-radius[j], v+radius[j]+1)
              sub_u = slice(u-radius[j], u+radius[j]+1)
              img_sub = img[sub_v,sub_u]
              img_angle_sub  = img_angle[sub_v, sub_u]
              img_weight_sub = img_weight[sub_v, sub_u]
              ccs = corner_correlation_score(img_sub,img_weight_sub,v1[i,:],v2[i,:])
              scores[i] = max(scores[i], ccs)

    return scores

def ccs():
    import theano.tensor as T
    img = T.matrix('img')
    img_weight = T.matrix('img_weight')
    v1 = T.vector('v1')
    v2 = T.vector('v2')
    img_filter = T.matrix('img_filter')

def corner_correlation_score(img, img_weight, v1, v2):

    # center
    c = np.ones((1,2),dtype=np.float)*(img_weight.shape[0]+1)/2.0
    
    p1 = np.empty((np.prod(img.shape), 2), dtype=np.float)
    for x in range(img_weight.shape[1]):
        for y in range(img_weight.shape[0]):
            # idx needs to be ordered this way to be consistent with the way
            # vec_img is flattened below
            idx = y * img_weight.shape[1] + x
            p1[idx,0] = c[0,0]-x
            p1[idx,1] = c[0,1]-y

    v1_2 = v1[np.newaxis].T.dot(v1[np.newaxis])
    v2_2 = v2[np.newaxis].T.dot(v2[np.newaxis])
    p2 = p1 - p1.dot(v1_2)
    p3 = p1 - p1.dot(v2_2)
    
    p2_norm = np.sqrt(np.sum(p2**2, axis=-1))
    p3_norm = np.sqrt(np.sum(p3**2, axis=-1))

    vec_filter = -1*np.ones(np.prod(img.shape), dtype=np.float)
    vec_filter[(p2_norm <= 1.5) | (p3_norm <= 1.5)] = 1
            
    #xy = np.empty((1,2))
    ### compute gradient filter kernel (bandwith = 3 px)
    #img_filter = -1*np.ones(img_weight.shape)
    #for x in range(img_weight.shape[1]):
    #    for y in range(img_weight.shape[0]):
    #        xy[0,0] = x
    #        xy[0,1] = y
    #        p1 = xy - c
    #        p2 = p1.dot(v1[np.newaxis].T.dot(v1[np.newaxis]))
    #        p3 = p1.dot(v2[np.newaxis].T.dot(v2[np.newaxis]))
    #        if norm(p1-p2)<=1.5 or norm(p1-p3)<=1.5:
    #            img_filter[y,x] = 1
    #vec_filter = img_filter.flatten()

    # convert into vectors
    vec_weight = img_weight.flatten()

    # normalize
    vec_weight = (vec_weight-np.mean(vec_weight))/np.std(vec_weight)
    vec_filter = (vec_filter-np.mean(vec_filter))/np.std(vec_filter)

    # compute gradient score
    score_gradient = max(np.sum(vec_weight*vec_filter)/(len(vec_weight)),0)

    # create intensity filter kernel
    template = c_pycb.Template(np.arctan2(v1[1],v1[0]),np.arctan2(v2[1],v2[0]),c[0,0]-1)

    # checkerboard responses
    vec_img = img.flatten()
    a1 = np.sum(template.a1.flatten()*vec_img)
    a2 = np.sum(template.a2.flatten()*vec_img)
    b1 = np.sum(template.b1.flatten()*vec_img)
    b2 = np.sum(template.b2.flatten()*vec_img)

    # mean
    mu = (a1+a2+b1+b2)/4

    # case 1: a=white, b=black
    score_a = min(a1-mu,a2-mu)
    score_b = min(mu-b1,mu-b2)
    score_1 = min(score_a,score_b)

    # case 2: b=white, a=black
    score_a = min(mu-a1,mu-a2)
    score_b = min(b1-mu,b2-mu)
    score_2 = min(score_a,score_b)

    # intensity score: max. of the 2 cases
    score_intensity = max(max(score_1,score_2),0)

    # final score: product of gradient and intensity score
    return score_gradient*score_intensity

def gradient(img, img_t=None):

    height, width = img.shape

    if img_t is None:
        img_t = theano.shared(img)

    # compute image derivatives (for principal axes estimation)

    # sobel masks
    mask_u = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=np.float)
    mask_v = mask_u.T

    ### theano ###
    masks = np.empty((2, mask_u.shape[0], mask_u.shape[1]),dtype=mask_u.dtype)
    masks[0,:,:] = mask_u
    masks[1,:,:] = mask_v
    masks_t = theano.shared(masks)
    ds = conv2d(img_t, masks_t, border_mode='full').eval()
    offset = (mask_u.shape[0]-1)/2
    du = ds[0, offset:height+offset, offset:width+offset]
    dv = ds[1, offset:height+offset, offset:width+offset]
    ######

    #du = convolve2d(img, mask_u, 'same')
    #dv = convolve2d(img, mask_v, 'same')

    angle = np.arctan2(dv, du)
    weight = np.sqrt(np.power(du, 2) + np.power(dv, 2))

    # correct angle to lie in between [0,pi]
    angle[angle<0] = angle[angle<0] + pi
    angle[angle>pi] = angle[angle>pi] - pi

    return du, dv, angle, weight

def prepare_image(img):
    img = img.astype(np.float64) / 255.0
    if len(img.shape) == 3:
        img = rgb2gray(img)
    return img

def find_corners(img, tau=0.01, refine_corners=True):

    img = prepare_image(img)

    height, width = img.shape

    ### theano ###
    img_t = theano.shared(img)
    ######

    du, dv, angle, weight = gradient(img, img_t)

    #import scipy.io
    #stuff = scipy.io.loadmat("img_angle.mat")
    #du = stuff["img_du"]
    #dv = stuff["img_dv"]
    #angle = stuff["img_angle"]
    #weight = stuff["img_weight"]

    # scale input image
    min = np.min(img)
    max = np.max(img)
    img = (img-min)/(max-min)

    radius = [4, 8, 12]

    # template properties
    template_props = [
                      [   0,  pi/2, radius[0]],
                      [pi/4, -pi/4, radius[0]],
                      [   0,  pi/2, radius[1]],
                      [pi/4, -pi/4, radius[1]],
                      [   0,  pi/2, radius[2]],
                      [pi/4, -pi/4, radius[2]],
                     ]

    # Filter image
    corners_img = np.zeros(img.shape)
    for template_params in template_props:
        template = c_pycb.Template(*template_params)

        ### theano ###
        stuff = np.empty((4, template.a1.shape[0], template.a1.shape[1]),dtype=template.a1.dtype)
        stuff[0,:,:] = template.a1
        stuff[1,:,:] = template.a2
        stuff[2,:,:] = template.b1
        stuff[3,:,:] = template.b2
        ts=theano.shared(stuff)

        r = template_params[2]
        s = conv2d(img_t, ts, border_mode='full').eval()
        corners_a1 = s[0, r:height+r,r:width+r]
        corners_a2 = s[1, r:height+r,r:width+r]
        corners_b1 = s[2, r:height+r,r:width+r]
        corners_b2 = s[3, r:height+r,r:width+r]
        ###

        #corners_a1 = convolve2d(img, template.a1, 'same')
        #corners_a2 = convolve2d(img, template.a2, 'same')
        #corners_b1 = convolve2d(img, template.b1, 'same')
        #corners_b2 = convolve2d(img, template.b2, 'same')

        # Compute mean
        corners_mu = (corners_a1 + corners_a2 + corners_b1 + corners_b2)/4

        # case 1: a=white, b=black
        corners_a = np.minimum(corners_a1 - corners_mu,corners_a2 - corners_mu)
        corners_b = np.minimum(corners_mu - corners_b1,corners_mu - corners_b2)
        corners_1 = np.minimum(corners_a,corners_b)

        # case 2: b=white, a=black
        corners_a = np.minimum(corners_mu - corners_a1, corners_mu - corners_a2)
        corners_b = np.minimum(corners_b1 - corners_mu, corners_b2 - corners_mu)
        corners_2 = np.minimum(corners_a, corners_b)

        # update corner map
        corners_img = np.maximum(corners_img, corners_1)
        corners_img = np.maximum(corners_img, corners_2)

    # extract corner candidates via non maximum suppression
    corners = non_maximum_supression(corners_img,3,0.025,5)

    # subpixel refinement
    if refine_corners:
        corners, v1, v2 = do_refine_corners(du,dv,angle,weight,corners,10)

    # remove corners without edges
    idx = np.where((v1[:,0]==0) & (v1[:,1]==0))[0]
    corners = np.delete(corners, idx, 0)
    v1 = np.delete(v1, idx, 0)
    v2 = np.delete(v2, idx, 0)

    #% score corners
    scores = score_corners(img,angle,weight,corners,v1,v2,radius)

    # remove low scoring corners
    idx = np.where(scores<tau)
    corners = np.delete(corners, idx, 0)
    v1 = np.delete(v1, idx, 0)
    v2 = np.delete(v2, idx, 0)
    scores = np.delete(scores, idx, 0)

    # make v1(:,1)+v1(:,2) positive (=> comparable to c++ code)
    idx = np.where((v1[:,0]+v1[:,1])<0)
    v1[idx,:] = -v1[idx,:]

    # make all coordinate systems right-handed (reduces matching ambiguities from 8 to 4)
    corners_n1 = np.column_stack([v1[:,1], -v1[:,0]])
    flip       = -np.sign(corners_n1[:,0]*v2[:,0]+corners_n1[:,1]*v2[:,1])
    v2 = v2*flip[:,np.newaxis].dot(np.ones((1,2)))

    return corners, v1, v2

def chessboards_from_corners(corners, v1, v2):

    chessboards = []
    chessboard_energies = []

    # for all seed corners do
    for i in range(corners.shape[0]):

      # init 3x3 chessboard from seed i
      chessboard = init_chessboard(corners, v1, v2, i)

      # compute current energy

      # check if there's a valid board
      if chessboard is None:
          continue

      # check if this is a useful initial guess
      energy = c_pycb.chessboard_energy(chessboard, corners)
      if energy > 0:
        continue

      # try growing chessboard
      while True:

        # compute proposals and energies
        proposals = []
        p_energy = np.empty(4)
        for j in range(4):
            proposal = grow_chessboard(chessboard, corners, j)
            p_energy[j] = c_pycb.chessboard_energy(proposal, corners)
            proposals.append(proposal)

        # find best proposal
        min_idx = np.argmin(p_energy)
        min_val = p_energy[min_idx]

        # accept best proposal, if energy is reduced
        if min_val < energy:
          chessboard = proposals[min_idx]
          energy = min_val
          #if 0
          #  figure, hold on, axis equal;
          #  chessboards{1} = chessboard;
          #  plotChessboards(chessboards,corners);
          #  keyboard;
        else:
          break

      # if chessboard has low energy (corresponding to high quality)
      if energy <- 10:

        # check if new chessboard proposal overlaps with existing chessboards
        if len(chessboards) == 0:
            chessboards.append(chessboard)
            chessboard_energies.append(energy)
        else:
            overlap = np.zeros((len(chessboards),2))
            for j in range(len(chessboards)):
                for k in range(len(chessboards[j].flatten())):
                    if np.any(chessboards[j].flatten()[k]==chessboard.flatten()):
                        overlap[j,0] = 1
                        overlap[j,1] = chessboard_energies[j]
                        break

            # add chessboard (and replace overlapping if neccessary)
            if not np.any(overlap[:,0]):
                chessboards.append(chessboard)
                chessboard_energies.append(energy)
            else:
                idx = np.where(overlap[:,0]==1)[0]
                if not np.any(overlap[idx,1] <= energy):
                    for x in sorted(idx, reverse=True):
                        del chessboards[x]
                        del chessboard_energies[x]
                    chessboards.append(chessboard)
                    chessboard_energies.append(energy)

    return chessboards

def grow_chessboard(chessboard, corners, border_type):

    # return immediately, if there do not exist any chessboards
    if chessboard is None:
        return

    # list of neighboring elements, which are currently not in use
    unused = np.arange(corners.shape[0])
    used  = chessboard[chessboard!=0]
    unused = np.delete(unused, used)

    # candidates from unused corners
    cand = corners[unused,:]

    # switch border type 1..4
    if border_type == 0:
        p1 = corners[chessboard[:, -3], :]
        p2 = corners[chessboard[:, -2], :]
        p3 = corners[chessboard[:, -1], :]
        pred = predict_corners(p1, p2, p3)
        idx = assign_closest_corners(cand, pred)
        if idx is not None:
            chessboard = np.column_stack([chessboard, unused[idx].T])

    elif border_type == 1:
        p1 = corners[chessboard[-3, :], :]
        p2 = corners[chessboard[-2, :], :]
        p3 = corners[chessboard[-1, :], :]
        pred = predict_corners(p1, p2, p3)
        idx = assign_closest_corners(cand, pred)
        if idx is not None:
            chessboard = np.row_stack([chessboard, unused[idx]])
    elif border_type == 2:
        p1 = corners[chessboard[:, 2], :]
        p2 = corners[chessboard[:, 1], :]
        p3 = corners[chessboard[:, 0], :]
        pred = predict_corners(p1, p2, p3)
        idx = assign_closest_corners(cand, pred)
        if idx is not None:
            chessboard = np.column_stack([unused[idx].T, chessboard])
    elif border_type == 3:
        p1 = corners[chessboard[2, :], :]
        p2 = corners[chessboard[1, :], :]
        p3 = corners[chessboard[0, :], :]
        pred = predict_corners(p1, p2, p3)
        idx = assign_closest_corners(cand, pred)
        if idx is not None:
            chessboard = np.row_stack([unused[idx], chessboard])

    return chessboard

def predict_corners(p1,p2,p3):

    # linear prediction (old)
    # function pred = predictCorners(p1,p2,p3)
    # pred = 2*p3-p2;

    # replica prediction (new)

    # compute vectors
    v1 = p2-p1
    v2 = p3-p2

    # predict angles
    a1 = np.arctan2(v1[:,1],v1[:,0])
    a2 = np.arctan2(v2[:,1],v2[:,0])
    a3 = (2*a2-a1)[:,np.newaxis]

    # predict scales
    s1 = np.sqrt(v1[:,0]**2+v1[:,1]**2)
    s2 = np.sqrt(v2[:,0]**2+v2[:,1]**2)
    s3 = (2*s2-s1)[:,np.newaxis]

    # predict p3 (the factor 0.75 ensures that under extreme
    # distortions (omnicam) the closer prediction is selected)
    pred = p3 + 0.75*s3.dot(np.ones((1,2))) * np.column_stack([np.cos(a3), np.sin(a3)])

    return pred

def assign_closest_corners(cand, pred):

    # return error if not enough candidates are available
    if cand.shape[0] < pred.shape[0]:
        return None

    # build distance matrix
    D = np.zeros((cand.shape[0], pred.shape[0]))
    for i in range(pred.shape[0]):
        delta = cand-np.ones((cand.shape[0],1)).dot(pred[i,np.newaxis])
        D[:,i] = np.sqrt(delta[:,0]**2+delta[:,1]**2)

    idx = np.zeros((1, pred.shape[0]), dtype=np.int)
    # search greedily for closest corners
    for i in range(pred.shape[0]):
        (row,col) = np.where(D==D.min())
        row = row[0]
        col = col[0]
        idx[0, col] = row
        D[row,:] = np.inf
        D[:,col] = np.inf

    return idx

def init_chessboard(corners, v1, v2, idx):

    # return if not enough corners
    if corners.shape[0] < 9:
        return None

    # init chessboard hypothesis
    chessboard = np.zeros((3,3), dtype=np.int)

    # extract feature index and orientation (central element)
    v1 = v1[idx,:]
    v2 = v2[idx,:]
    chessboard[1,1] = idx

    dist1 = np.zeros(2)
    dist2 = np.zeros(6)

    # find left/right/top/bottom neighbors
    chessboard[1,2], dist1[0] = directional_neighbor(idx,+v1,chessboard,corners)
    chessboard[1,0], dist1[1] = directional_neighbor(idx,-v1,chessboard,corners)
    chessboard[2,1], dist2[0] = directional_neighbor(idx,+v2,chessboard,corners)
    chessboard[0,1], dist2[1] = directional_neighbor(idx,-v2,chessboard,corners)

    # find top-left/top-right/bottom-left/bottom-right neighbors
    chessboard[0,0], dist2[2] = directional_neighbor(chessboard[1,0],-v2,chessboard,corners)
    chessboard[2,0], dist2[3] = directional_neighbor(chessboard[1,0],+v2,chessboard,corners)
    chessboard[0,2], dist2[4] = directional_neighbor(chessboard[1,2],-v2,chessboard,corners)
    chessboard[2,2], dist2[5] = directional_neighbor(chessboard[1,2],+v2,chessboard,corners)

    # initialization must be homogenously distributed
    if (np.any(np.isinf(dist1)) or
        np.any(np.isinf(dist2)) or
        np.std(dist1)/np.mean(dist1) > 0.3 or
        np.std(dist2)/np.mean(dist2) > 0.3):
        return None

    return chessboard

def directional_neighbor(idx,v,chessboard,corners):

    # list of neighboring elements, which are currently not in use
    unused = np.arange(corners.shape[0])
    used  = chessboard[chessboard!=0]
    unused = np.delete(unused, used)

    # direction and distance to unused corners
    dir  = corners[unused,:] - np.ones((len(unused), 1)).dot(corners[idx,np.newaxis])
    dist = dir[:,0]*v[0] + dir[:,1]*v[1]

    # distances
    dist_edge = dir-dist[:,np.newaxis]*v[np.newaxis]
    dist_edge = np.sqrt(dist_edge[:,0]**2+dist_edge[:,1]**2)
    dist_point = dist
    dist_point[dist_point<0] = np.inf

    # find best neighbor
    dist = dist_point+5*dist_edge
    min_idx = np.argmin(dist)
    neighbor_idx = unused[min_idx]
    min_dist = dist[min_idx]
    return neighbor_idx, min_dist

def chessboard_energy(chessboard, corners):

    num_corners = np.prod(chessboard.shape)
    # energy: number of corners
    E_corners = -num_corners

    # energy: structure
    E_structure = 0

    # walk through rows
    for j in range(chessboard.shape[0]):
        for k in range(chessboard.shape[1]-2):
            x = corners[chessboard[j,k:k+3],:]
            e_s = np.sqrt(np.sum((x[0,:] + x[2,:] -2*x[1,:])**2,axis=-1)) / np.sqrt(np.sum((x[0,:]-x[2,:])**2,axis=-1))
            E_structure = max(E_structure, e_s)

    # walk through columns
    for j in range(chessboard.shape[1]):
        for k in range(chessboard.shape[0]-2):
            x = corners[chessboard[k:k+3,j],:];
            e_s = np.sqrt(np.sum((x[0,:]+x[2,:]-2*x[1,:])**2,axis=-1)) / np.sqrt(np.sum((x[0,:]-x[2,:])**2,axis=-1))
            E_structure = max(E_structure, e_s)

    # final energy
    return E_corners + num_corners*E_structure

def main():
    from scipy.misc import imread
    import matplotlib.cm as cm

    img_big = imread('t.jpg')
    scale_factor = .2
    img = imresize(img_big, scale_factor, interp='bicubic')

    print "Finding corners..."
    corners, v1, v2 = find_corners(img)

    print "Getting board..."
    chessboards = chessboards_from_corners(corners, v1, v2)

    print "Refining corners..."

    corners = np.round(corners * (1.0/scale_factor)).astype(np.int)
    img_big = prepare_image(img_big)
    du, dv, angle, weight = gradient(img_big)
    refined, v1, v2 = do_refine_corners(du, dv, angle, weight, corners, 15)

    return img_big, corners, refined, chessboards

def draw_boards(img, corners, refined, chessboards):
    from pylab import imshow, hold, show, scatter
    import matplotlib.cm as cm
    if len(img.shape) != 3:
        imshow(img, cmap=cm.Greys_r)
    else:
        imshow(img)
    hold(True)
    for board in chessboards:
        cs = corners[board.flatten()]
        scatter(cs[:, 0], cs[:, 1], color='blue')
        rs = refined[board.flatten()]
        scatter(rs[:, 0], rs[:, 1], color='green')
    show()

def extract_chessboards(filename):

    img_big = imread(filename)

    if img_big.shape[1] > 4000:
        scale_factor = .2
    elif img_big.shape[1] > 2000:
        scale_factor = .5
    else:
        scale_factor = 1

    if scale_factor < 1:
        img = imresize(img_big, scale_factor, interp='bicubic')
    else:
        img = img_big

    print "Finding corners..."
    corners, v1, v2 = find_corners(img)

    print "Getting board..."
    chessboards = chessboards_from_corners(corners, v1, v2)

    print "Refining corners..."

    corners = np.round(corners * (1.0/scale_factor)).astype(np.int)
    img_big = prepare_image(img_big)
    du, dv, angle, weight = gradient(img_big)
    refined, v1, v2 = do_refine_corners(du, dv, angle, weight, corners, 15)

    return refined, chessboards

def write_chessboards(filename, output_filename, board_sizes):

    corners, chessboards = extract_chessboards(filename)

    boards_to_write = [x if x.shape in board_sizes]

    if len(boards_to_write) == 0:
        return
    else:
        board = boards_to_write[0]


if __name__ == "__main__":

    img, corners, refined, chessboards = main()
    draw_boards(img, corners, chessboards)
