"""
    Codes for averaging contours of the same subject among researchers
    and plotting the contours on flatmaps
    
"""

from hcpannot.proc import (proc, rigid_align_points)
from hcpannot.config import (ventral_raters, meanrater)
import neuropythy as ny
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# list of raters
raters = ventral_raters
raters.append(meanrater)

# generate a list of colors
colors = ['r', 'g', 'b', 'c', 'm', 'k']
rater_colors = {r: c for r, c in zip(raters, colors)}


def nestget(d, k):
    """Retrieves nested data from the proc dictionaries.
    
    Certain keys such as `'boundaries'` are accessible in the dictionaries that
    are returned by the `proc` function only via the `'nested_data'` key, which
    typically contains another proc dictionary with additional data. The
    `nestget` function gets data from these embedded dictionaries.
    """
    while k not in d:
        d = d['nested_data']
    return d[k]

def gen_contour_coords(rater, subject_id, hemi, contour, proc_path, contour_save_path, npoints=500):
    """
    Divide a contour of a given subject, rater, and hemisphere into a given number
    (default 500) of points and save the coordinates.
    """
    
    # check if the contour already exists, if so, skip the processing
    cache_file = os.path.join(proc_path, 'fsaverage', f'cacherater_{rater}_{subject_id}_{hemi}_{contour}.mgz')
    
    data = proc('ventral', rater=rater, sid=subject_id, hemisphere=hemi, 
               save_path=proc_path, load_path=contour_save_path)
    
    if contour != 'V3v': 
        trace = data['traces'][contour]
        # break the contour into evenly spaced points (default is 500)
        coords = trace.curve.linspace(npoints)
    
    # V3v was not annotated by researchers in this project, and the data are stored differently
    else:
        v3v = nestget(data, 'v3v_contour')
        v3v_curve = ny.curve_spline(v3v[0], v3v[1])
        coords = v3v_curve.linspace(npoints)
        
    # align the points to fsaverage space
    cortex = data['cortex']
    fmap1 = data['flatmap']
    fmap2 = ny.to_flatmap('occipital_pole', cortex)
    addr = fmap1.address(coords)
    coordinates = fmap2.unaddress(addr)
    
    return coordinates

def plot_contours(raters, subject_id, hemi, contours, 
                  proc_path, contour_save_path, npoints=500, ax=None, lw=None, autogen=True):
    """
    Plot contours of the same subject from different raters on a flatmap. If the contours 
    are not already saved, generate and save them.
    """
    # check if raters and contours are lists
    # if not, convert them to lists
    if not isinstance(raters, list):
        raters = [raters]
        
    if not isinstance(contours, list):
        contours = [contours]
        
    if ax is None:
        ax = plt.gca()
        
    raters_legend = set() # to avoid duplicate legend entries
    
    for r in raters:
        for c in contours:
            
            # define cache file path
            cache_file = os.path.join(proc_path, 'fsaverage', f'cacherater_{r}_{subject_id}_{hemi}_{c}.mgz')
            
            # check if the contour coordinates are already saved
            # if the coordinates are already saved, load them
            if os.path.isfile(cache_file):
                coords = ny.load(cache_file)
            
            # if not, generate and save them
            else:
                if autogen:
                    print(f'Contour coordinates for {r} {subject_id} {hemi} {c} not found. Generating...')
                    coords = gen_contour_coords(r, subject_id, hemi, c, proc_path, contour_save_path, npoints)
                    ny.save(cache_file, coords)
                else:
                    print(f'Contour coordinates for {r} {subject_id} {hemi} {c} not found.')
                    continue
               
            # plot the contours 
            color = rater_colors.get(r, 'gray')
            
            # plot the contours with the same color for the same rater
            # and add the rater to the legend only once
            if r not in raters_legend:
                raters_legend.add(r)
                ax.plot(coords[0], coords[1], color=color, lw=lw, label=f'{r}')
            else:
                ax.plot(coords[0], coords[1], color=color, lw=lw)    
    ax.legend()
            
    return

def create_LineCollection(sids, hemi, roi, space,
                          proc_path, rater, 
                          lw=0.25, alpha=0.3, color=None):
    
    """
        Create a LineCollection object for contours of the same ROI from the same
        researcher from different subjects to be plotted together effectively
    """
    
    missing_contour = [] # list of subjects for which the contour is missing
    lines = [] # list of coordinates for each subject
    
    # for each subject, try to load the contour coordinates
    for sid in sids:
        cache_file = os.path.join(proc_path, space, f'cacherater_{rater}_{sid}_{hemi}_{roi}.mgz')
        
        # check if the contour coordinates are already saved
        if os.path.isfile(cache_file):
            
            # if coordinates are saved, load them
            coords = ny.load(cache_file)
            
            # x and y coordinates were saved as separate arrays (npoints * 2)
            # stack them together to create a list of coordinates
            lines.append(np.column_stack(coords))
        
        else:
            # if the coordinates are not saved, add the subject to the missing list
            missing_contour.append(sid)
            
    lc = LineCollection(lines, linewidths=lw, alpha=alpha, colors=color)
    return (lc, missing_contour)

def plot_lc(sids, hemi, rois, space, proc_path, rater=meanrater,
            plot_v123=False, meanlines=None, npoints=500,flatmap=True, 
            ax=None, lw=0.25, alpha=0.3, colors=None):
    """
    Plot a LineCollection object on a flatmap. By default, the contours are plotted on
    flatmap with the mean V1-V3 contours.
    
    """
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(3.5,3.5), dpi=1200)
    
    if not isinstance(rois, list):
        rois = [rois]
        
    if plot_v123:
        if meanlines is None:
            print('Mean V1-V3 contours not provided.')
            return
        else:
            meanv123 = meanlines[hemi]
            for t in meanv123.keys():
                for c in meanv123[t]:
                    trace = meanv123[t][c].curve.linspace(npoints)
                    
                    if hemi == 'lh':
                        ax.plot(trace[0]+8, trace[1], 'k-') # shift the x coordinates by 8
                    else:
                        ax.plot(trace[0]-8, trace[1], 'k-') # shift the x coordinates by -8
    
    for roi in rois:
        lc, missing_contour = create_LineCollection(sids, hemi, roi, space, proc_path, rater, 
                                                    color=colors.get(roi, None))
        ax.add_collection(lc)
        
    ax.set_title('Ventral Contours')
    ax.set_ylim(-75, 40)
    ax.axis('equal')
    ax.axis('off')

    return

def align_fsnative(sid, hemi, contours, rater, proc_path, npoints=500):
    """
        For a given subject in a given hemisphere delineated by a given rater,
        align the contours to mean contours in the fsaverage space.
    """
    
    mean_coords = []

    # TODO: check if order of contours matters
    
    for contour in contours:
        
        # define cache file path
        mean_cache_file = os.path.join(proc_path, 'fsaverage', f'cacherater_{meanrater}_{sid}_{hemi}_{contour}.mgz')
        
        # check if the mean contour coordinates are already saved
        if os.path.isfile(mean_cache_file):
            # if the coordinates are already saved, load them
            # and append them to the list of mean coordinates
            coords = ny.load(mean_cache_file)
            mean_coords.extend(coords)
        else:
            print(f'Mean contour coordinates for {sid} {hemi} {contour} not found. Skipping...')
            return
            
    # align the contours to the mean contours in fsaverage space
    # and save the aligned coordinates

    rater_coords = []
    
    # define cache file path for the rater
    for contour in contours:
        cache_file = os.path.join(proc_path, 'fsnative', f'cacherater_{rater}_{sid}_{hemi}_{contour}.mgz')
        
        if os.path.isfile(cache_file):
            coords = ny.load(cache_file)
            rater_coords.extend(coords)
        else:
            print(f'Contour coordinates for {rater} {sid} {hemi} {contour} not found. Skipping...')
            return
        
    # try to align the rater contours to the mean contours
    # TODO: inpsect the aligned contours
    try:
        aligned_contours = rigid_align_points(rater_coords, mean_coords)
    except Exception as e:
        print(f'Error aligning contours: {e}')
        return
    
    # break the aligned contours into separate contours
    for i, contour in enumerate(contours):
        aligned_contour = aligned_contours[:, npoints*i:npoints*(i+1)]
        
        # save the aligned contours
        file = os.path.join(proc_path, 'fsnative_aligned', f'cacherater_{rater}_{sid}_{hemi}_{contour}.mgz')
        ny.save(file, aligned_contour)
        
    return