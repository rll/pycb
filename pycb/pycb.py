import numpy as np
import scipy as sp
from numpy import pi
from scipy.misc import imresize

import theano
#from theano.tensor.signal.conv import conv2d as theano_conv2d
#from scipy.signal import convolve2d

import c_pycb

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

        corner_pos_old = corners[i,:]
        refined[i,:] = corner_pos_old

        #print "on corner %d of %d" % (i, corners.shape[0])
        cu = corner_pos_old[0]
        cv = corner_pos_old[1]

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

        # pixel orientation vectors
        sub_du = img_du[window_v,window_u].flatten('F')
        sub_dv = img_dv[window_v,window_u].flatten('F')
        os = np.column_stack([sub_du, sub_dv])
        norm_os = np.sqrt(np.sum(os**2,axis=-1))
        os = os / norm_os[:, np.newaxis]

        A1, A2 = c_pycb.refine_orientation(cu, cv, r, width, height, norm_os, os, img_du, img_dv, v1[i, :], v2[i, :])

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

                if  norm_os[o_idx] < 0.1:
                    continue

                # robust subpixel corner estimation

                # do not consider center pixel

                if u!=cu or v!=cv:

                    d1 = c_pycb.rel_pixel_distance(u,v,cu,cv,v1[i])
                    d2 = c_pycb.rel_pixel_distance(u,v,cu,cv,v2[i])

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
            corner_pos_new = np.linalg.solve(G, b).T

            # set corner to invalid, if position update is very large
            if np.linalg.norm(corner_pos_new-corner_pos_old) >= 4:
                v1[i,:] = 0
                v2[i,:] = 0
            else:
                refined[i,:] = corner_pos_new
        else:
            v1[i,:] = 0
            v2[i,:] = 0

    return refined, v1, v2

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
    
    # Vectorize these computations
    p2_norm = np.sqrt(np.sum(p2**2, axis=-1))
    p3_norm = np.sqrt(np.sum(p3**2, axis=-1))

    vec_filter = -1*np.ones(np.prod(img.shape), dtype=np.float)
    vec_filter[(p2_norm <= 1.5) | (p3_norm <= 1.5)] = 1
            
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

def gradient(img, img_theano=None):

    height, width = img.shape

    #if img_theano is None:
    #    img_theano = theano.shared(img)

    # compute image derivatives (for principal axes estimation)

    # sobel masks
    #mask_u = np.array([[-1, 0, 1],
    #                   [-1, 0, 1],
    #                   [-1, 0, 1]], dtype=np.float)
    #mask_v = mask_u.T

    ## batch masks for speed
    #masks = np.empty((2, mask_u.shape[0], mask_u.shape[1]),dtype=mask_u.dtype)
    #masks[0,:,:] = mask_u
    #masks[1,:,:] = mask_v

    #ds = theano_conv2d(img, masks, border_mode='full').eval()

    ## Strip off padding
    #offset = (mask_u.shape[0]-1)/2
    #du_2 = ds[0, offset:height+offset, offset:width+offset]
    #dv_2 = ds[1, offset:height+offset, offset:width+offset]

    du, dv = c_pycb.sobel(img)

    angle = np.arctan2(dv, du)
    weight = np.sqrt(np.power(du, 2) + np.power(dv, 2))

    # correct angle to lie in between [0,pi]
    angle[angle<0] = angle[angle<0] + pi
    angle[angle>pi] = angle[angle>pi] - pi

    return du, dv, angle, weight

def find_corners(img, tau=0.001, refine_corners=True, use_corner_thresholding=True):

    img = prepare_image(img)

    # Use a shared variable so we aren't copying the image each time we use it
    #img_theano = theano.shared(img)

    height, width = img.shape

    du, dv, angle, weight = gradient(img)#, img_theano)

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

        ## Batch filters for speed
        #filters = np.empty((4, template.a1.shape[0], template.a1.shape[1]),dtype=template.a1.dtype)
        #filters[0,:,:] = template.a1
        #filters[1,:,:] = template.a2
        #filters[2,:,:] = template.b1
        #filters[3,:,:] = template.b2

        #s = theano_conv2d(img_theano, filters, border_mode='full').eval()

        ## Take off padding
        #r = template_params[2]
        #corners_a1_2 = s[0, r:height+r,r:width+r]
        #corners_a2_2 = s[1, r:height+r,r:width+r]
        #corners_b1_2 = s[2, r:height+r,r:width+r]
        #corners_b2_2 = s[3, r:height+r,r:width+r]

        #template = c_pycb.Template(*template_params)
        corners_a1, corners_a2, corners_b1, corners_b2 = c_pycb.conv_template(img, template.a1, template.a2, template.b1, template.b2)
        #corners_a1 = c_pycb.conv2(img, template.a1)
        #corners_a2 = c_pycb.conv2(img, template.a2)
        #corners_b1 = c_pycb.conv2(img, template.b1)
        #corners_b2 = c_pycb.conv2(img, template.b2)
        #corners_a1_3 = convolve2d(img, template.a1, mode='same')
        #import pdb; pdb.set_trace()
        #print corners_a1, corners_a2, corners_b1, corners_b2
        #corners_a1 = convolve2d(img, template.a1, mode='same')
        #corners_a2 = convolve2d(img, template.a2, mode='same')
        #corners_b1 = convolve2d(img, template.b1, mode='same')
        #corners_b2 = convolve2d(img, template.b2, mode='same')

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

    if len(corners) == 0:
        return corners, v1, v2

    # remove corners without edges
    idx = np.where((v1[:,0]==0) & (v1[:,1]==0))[0]
    corners = np.delete(corners, idx, 0)
    v1 = np.delete(v1, idx, 0)
    v2 = np.delete(v2, idx, 0)

    if len(corners) == 0:
        return corners, v1, v2

    if use_corner_thresholding:
        #% score corners
        scores = score_corners(img,angle,weight,corners,v1,v2,radius)

        # remove low scoring corners
        idx = np.where(scores<tau)
        corners = np.delete(corners, idx, 0)
        v1 = np.delete(v1, idx, 0)
        v2 = np.delete(v2, idx, 0)
        scores = np.delete(scores, idx, 0)

    # make v1(:,1)+v1(:,2) positive (=> comparable to c++ code)
    idx = np.where((v1[:,0]+v1[:,1])<0)[0]
    if len(idx) > 0:
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

def fix_orientations(cbs, points, img):
    cbs = [fix_orientation(cb, points, img) for cb in cbs]
    cbs = [cb for cb in cbs if cb is not None]
    return cbs

def fix_orientation(chessboard, points, img, debug=False):

    corners = np.array([chessboard[0,0],
                        chessboard[0,-1],
                        chessboard[-1,0],
                        chessboard[-1,-1]])

    if chessboard.shape[0] < chessboard.shape[1]:
        opposing_corners = np.array([chessboard[-1, -2],
                                     chessboard[-1, 1],
                                     chessboard[0, -2],
                                     chessboard[0, 1]])
    else:
        opposing_corners = np.array([chessboard[-2, -1],
                                     chessboard[-2, 0],
                                     chessboard[1, -1],
                                     chessboard[1, 0]])

    corner_points = points[corners]
    opposing_corner_points = points[opposing_corners]

    max_squares = min(chessboard.shape)
    steps = 20

    differences = opposing_corner_points - corner_points
    distances = np.sqrt(np.sum(differences**2, axis=-1))
    differences = differences / distances[:, np.newaxis]

    # Look at 2 squares
    deltas = (2.0/max_squares) * distances / steps
    deltas = deltas[:, np.newaxis]

    if debug:
        from pylab import imshow, plot, scatter, show, hold
        import matplotlib.cm as cm
        imshow(img, cmap=cm.Greys_r)
        hold(True)
        scatter(corner_points[0, 0], corner_points[0, 1], color='green')
        scatter(corner_points[1, 0], corner_points[1, 1], color='blue')
        scatter(corner_points[2, 0], corner_points[2, 1], color='purple')
        scatter(corner_points[3, 0], corner_points[3, 1], color='orange')
        plot([corner_points[0, 0], opposing_corner_points[0, 0]], 
             [corner_points[0, 1], opposing_corner_points[0, 1]], color='green')
        plot([corner_points[1, 0], opposing_corner_points[1, 0]], 
             [corner_points[1, 1], opposing_corner_points[1, 1]], color='blue')
        plot([corner_points[2, 0], opposing_corner_points[2, 0]], 
             [corner_points[2, 1], opposing_corner_points[2, 1]], color='purple')
        plot([corner_points[3, 0], opposing_corner_points[3, 0]], 
             [corner_points[3, 1], opposing_corner_points[3, 1]], color='orange')
        show()

    dark_counts = np.zeros(4)

    for step in range(steps):

        current_points = corner_points + (step + 1) * deltas * differences

        coords = np.round(current_points).astype(np.int)

        if len(img.shape) == 3:
            values = img[coords[:,1], coords[:,0], :].mean(axis=1)
        else:
            values = img[coords[:,1], coords[:,0]]

        idx = np.argsort(values)
        dark_counts[idx[:2]] += 1

        if debug:
            print idx
            print dark_counts
            imshow(img, cmap=cm.Greys_r)
            hold(True)
            scatter(coords[:, 0], coords[:, 1])
            show()

    idx = np.argsort(dark_counts)[::-1]
    dark = np.sort(idx[:2])

    if debug:
        print "After loop"
        print idx
        print dark

    # We know that 0 and 3 can't be the same color, and that 1 and 2 can't be
    # the same color.
    if (dark[0] == 0 and dark[1] == 3) or (dark[0] == 1 and dark[1] == 2):
        print "Error finding orientation."
        return None

    if dark[0] == 0 and dark[1] == 1:
        return np.rot90(chessboard, 2)
    if dark[0] == 1 and dark[1] == 3:
        return np.rot90(chessboard, 3)
    if dark[0] == 2 and dark[1] == 3:
        return chessboard
    if dark[0] == 0 and dark[1] == 2:
        return np.rot90(chessboard, 1)

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
            x = corners[chessboard[k:k+3,j],:]
            e_s = np.sqrt(np.sum((x[0,:]+x[2,:]-2*x[1,:])**2,axis=-1)) / np.sqrt(np.sum((x[0,:]-x[2,:])**2,axis=-1))
            E_structure = max(E_structure, e_s)

    # final energy
    return E_corners + num_corners*E_structure

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def prepare_image(img):
    img = img.astype(np.float64) / 255.0
    if len(img.shape) == 3:
        img = rgb2gray(img)
    return img

def draw_corners(img, corners):
    from pylab import imshow, hold, show, scatter, plot, xlim, ylim
    import matplotlib.cm as cm
    if len(img.shape) != 3:
        imshow(img, cmap=cm.Greys_r)
    else:
        imshow(img)
    hold(True)
    scatter(corners[:, 0], corners[:, 1])
    xlim(corners[:,0].min()*.9, corners[:, 0].max()*1.1)
    ylim(corners[:,1].max()*1.1, corners[:, 1].min()*0.9)
    show()

def draw_boards(img, corners, chessboards, old_corners=None):
    colors = ['blue', 'purple', 'red', 'orange', 'yellow', 'green']
    from pylab import imshow, hold, show, scatter, plot, xlim, ylim
    import matplotlib.cm as cm
    if len(img.shape) != 3:
        imshow(img, cmap=cm.Greys_r)
    else:
        imshow(img)
    hold(True)
    for board in chessboards:
        if old_corners is not None:
            cs = old_corners[board.flatten()]
            scatter(cs[:, 0], cs[:, 1], color='red')
        rs = corners[board.flatten()]
        scatter(rs[:, 0], rs[:, 1], color='green')
        xlim(rs[:,0].min()*.9, rs[:, 0].max()*1.1)
        ylim(rs[:,1].max()*1.1, rs[:, 1].min()*0.9)
        color_idx = 0
        for i in range(board.shape[0]):
            row = board[i]
            indices = [0, -1]
            color = colors[color_idx % len(colors)]
            color_idx += 1
            if i == 0:
                linewidth = 3
                color = 'chartreuse'
            else:
                linewidth = 2
            plot(corners[row[indices], 0], 
                 corners[row[indices], 1], 
                 color=color, linewidth=linewidth)
            if i > 0:
                plot([corners[board[i-1, 0], 0], corners[board[i, -1], 0]], 
                     [corners[board[i-1, 0], 1], corners[board[i, -1], 1]], 
                     color=color, linewidth=2)
    show()

def extract_chessboards(img, include_unrefined=False, use_corner_thresholding=True):

    if img.shape[1] > 4000:
        scale_factor = .2
    elif img.shape[1] > 2000:
        scale_factor = .5
    else:
        scale_factor = 1

    if scale_factor < 1:
        img_scaled = imresize(img, scale_factor, interp='bicubic')
    else:
        img_scaled = img

    corners, v1, v2 = find_corners(img_scaled, use_corner_thresholding=use_corner_thresholding)
    chessboards = chessboards_from_corners(corners, v1, v2)
    chessboards = fix_orientations(chessboards, corners, img_scaled)

    if scale_factor < 1:
        corners = np.round(corners * (1.0/scale_factor)).astype(np.int)
        img = prepare_image(img)
        du, dv, angle, weight = gradient(img)
        refined, v1, v2 = do_refine_corners(du, dv, angle, weight, corners, 15)
    else:
        refined = corners

    if include_unrefined:
        return refined, chessboards, corners
    else:
        return refined, chessboards

def save_chessboard(output_filename, corners, chessboards, allowed_board_sizes, verbose=True):
    if len(chessboards) == 0:
        if verbose:
            print "WARNING: no boards found. Nothing written to %s" % (output_filename)
        return
    board = None
    for board_size in allowed_board_sizes:
        boards = [x for x in chessboards if x.shape == board_size] 
        if len(boards) != 0:
            board = boards[0]
            break
    if board is None:
        if verbose:
            print "WARNING: couldn't find board with right size. Nothing written to file %s. Had " \
                "board with size (%d, %d)" % (output_filename, 
                                              chessboards[0].shape[0], 
                                              chessboards[0].shape[1])
        return
    else:
        if verbose:
            print "Found board. Written to file %s" % output_filename

    board_points = corners[board.flatten()]

    file = open(output_filename, 'w')
    for point in board_points:
        file.write("%f, %f\n" % (point[0], point[1]))
    file.close()

if __name__ == "__main__":

    from scipy.misc import imread
    img = imread("../examples/scene2.jpg")
    corners, chessboards = extract_chessboards(img)
    #draw_boards(img, corners, chessboards)
