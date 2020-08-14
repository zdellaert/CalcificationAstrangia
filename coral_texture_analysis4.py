import numpy as np
import pylab as py
import matplotlib as plt
import imageio
#import pytesseract

import os 
from os import listdir
from os.path import isfile, join
from skimage import exposure
from skimage.transform import rescale
from skimage.morphology import remove_small_holes, remove_small_objects

import feature_textures as ft
import visualization_tools as vt
from skimage.filters import threshold_otsu
from skimage.filters import try_all_threshold
from skimage import filters

#Reads in imagefile names
infile_root = '/Users/vargasp/Desktop/Coral/Data2/'
infile_root = '/Users/vargasp/Research/Projects/Coral/Data2/'

infile_dirs = ['Apo_Hot/','Apo_Ambient/','Apo_Cold/','Sym_Hot/','Sym_Ambient/','Sym_Cold/']
ImageFiles = []
nImages = np.empty(len(infile_dirs),dtype=np.int)
for i, infile_dir, in enumerate(infile_dirs):
    ImageFilesDir = [f for f in sorted(listdir(infile_root+infile_dir)) if isfile(join(infile_root+infile_dir, f))]
    nImages[i] = len(ImageFilesDir)
    ImageFiles += ImageFilesDir

#Reads in the images
nPixels = (768, 1024)
count = 0
Images = np.empty([nPixels[0],nPixels[1],nImages.sum()])
for i, infile_dir, in enumerate(infile_dirs):
    for j in range(nImages[i]):
        Images[:,:,count] = imageio.imread(infile_root + infile_dir +ImageFiles[count])
        count += 1


#Reads in imagefile names
infile_root = '/Users/vargasp/Desktop/Coral/Masks/'
infile_root = '/Users/vargasp/Research/Projects/Coral/Masks/'

infile_dirs = ['Apo_Hot/','Apo_Ambient/','Apo_Cold/','Sym_Hot/','Sym_Ambient/','Sym_Cold/']
ImageFiles = []
nMasks = np.empty(len(infile_dirs),dtype=np.int)
for i, infile_dir, in enumerate(infile_dirs):
    ImageFilesDir = [f for f in sorted(listdir(infile_root+infile_dir)) if isfile(join(infile_root+infile_dir, f))]
    nMasks[i] = len(ImageFilesDir)
    ImageFiles += ImageFilesDir

#Reads in the images
nPixels = (768, 1024)
count = 0
Masks = np.empty([nPixels[0],nPixels[1],nMasks.sum()])
for i, infile_dir, in enumerate(infile_dirs):
    for j in range(nMasks[i]):
        Masks[:,:,count] = imageio.imread(infile_root + infile_dir +ImageFiles[count])
        count += 1

Masks = Masks.astype(np.bool)

#Reads the pixel size from the image
dPixels = np.load("dPixels.npy")
samplingFreq = 1.0/dPixels

#Masks the image
Images = Images*Masks

#Crops the images
Images = Images[:687,:,:]
Masks = Masks[:687,:,:]

ImagesHS = np.empty([Images.shape[2],6,Images.shape[0],Images.shape[1]])

for i in range(Images.shape[2]):
    Image = Images[:,:,i]
    Mask = Masks[:,:,i]
    
    # Contrast stretching
    p2, p98 = np.percentile(Image, (2, 98))
    ImagesHS[i,0,:,:] = Image
    ImagesHS[i,1,:,:] = exposure.rescale_intensity(Image, in_range=(p2, p98))*Mask
    ImagesHS[i,2,:,:] = exposure.equalize_hist(Image)*Mask # Equalization
    ImagesHS[i,3,:,:] = exposure.equalize_adapthist(Image/Image.max())*Mask # Adaptive Equalization
    ImagesHS[i,4,:,:] = exposure.adjust_gamma(Image)*Mask # Adjust gamma
    ImagesHS[i,5,:,:] = exposure.adjust_sigmoid(Image)*Mask    # Adjust gamma





FMP32, ROIs32 = ROI_FMP_Metrics(ImagesHS[:,:5,:,:],Masks,dPixels,R_size=32,ROI_return=True)
FMP64, ROIs64 = ROI_FMP_Metrics(ImagesHS[:,:5,:,:],Masks,dPixels,R_size=64,ROI_return=True)
FMP128, ROIs128 = ROI_FMP_Metrics(ImagesHS[:,:5,:,:],Masks,dPixels,R_size=128,ROI_return=True)
FMP256, ROIs256 = ROI_FMP_Metrics(ImagesHS[:,:5,:,:],Masks,dPixels,R_size=256,ROI_return=True)





labels = ['No_Equalization','Rescale_intensity','Equalization','Adaptive_Equalization','Adjust_Gamma']

for idx, label in enumerate(labels):
    output_files(FMP32[:,idx,:], ROIs32[:,idx,:,:,:],ImageFiles,label)

for idx, label in enumerate(labels):
    output_files(FMP64[:,idx,:], ROIs64[:,idx,:,:,:],ImageFiles,label)

for idx, label in enumerate(labels):
    output_files(FMP128[:,idx,:], ROIs128[:,idx,:,:,:],ImageFiles,label)

for idx, label in enumerate(labels):
    output_files(FMP256[:,idx,:], ROIs256[:,idx,:,:,:],ImageFiles,label)




vt.CreateBoxPlot(FMP32[0,:,:].T,remove_zeros=True,showfliers=False,labels=["a","b","c","d","e"])





def ROI_FMP_Metrics(Images,Masks,samplingSizes,R_size=32,ROI_return=False):
    '''
    Calcualtes the FMP of an array of ROIs

    Parameters
    ----------
    Images : np.array of images of with sizes (nImages, nTypes, nCols, Nrows)
    Masks : np.array of masks corresponding to the images of with sizes (nImages, nCols, Nrows)
    samplingSizes : Sampling size of the pixels 
        DESCRIPTION.
    R_size : int, size of the ROI. The default is 32.
    ROI_return : Bool, Returns all ROI images. DESCRIPTION. The default is False.

    Returns
    -------
    np.array of the FMP vlaues of the ROIs
    '''
    nImages, nTypes, nCols, nRows = Images.shape

    MetricsROI = np.zeros([nImages,nTypes,10000])
    ROIs = np.zeros([nImages,nTypes,10000,R_size,R_size])
    
    for n in range(nImages):
        print("Image: "+str(n))
        Mask = Masks[:,:,n]
        s = samplingSizes[n]
        r_count = 0

        for x in np.arange(0,Mask.shape[0] - int(R_size),int(R_size/2)):
            for y in np.arange(0,Mask.shape[1] - int(R_size),int(R_size/2)):
                if(np.where(Mask[x:x+int(R_size),y:y+int(R_size)] == False)[0].size == 0):
 
                    for m in range(nTypes):
                        ROI = Images[n,m,x:x+int(R_size),y:y+int(R_size)]
                        ROIc = background_correction(ROI)
                        MetricsROI[n,m,r_count] = ft.FMP(ROIc,s)

                        ROIs[n,m,r_count,:,:] = ROI

                    r_count = r_count + 1
    


    cut_list = np.count_nonzero(MetricsROI,axis=2).max()
    
    MetricsROI = MetricsROI[:,:,:cut_list]
    ROIs = ROIs[:,:,:cut_list,:,:]
    
    
    if(ROI_return):
        return MetricsROI, ROIs
    else:
        return MetricsROI


def background_correction(ROI):
    '''
    Fits a second-order polynomial surface to the ROI and subtracts it to background correct.

    Parameters
    ----------
    ROI : ROI of an image

    Returns
    -------
    bacckground corrected ROI
    '''

    xn,yn = ROI.shape

    x = np.linspace(0, xn, xn)
    y = np.linspace(0, yn, yn)
    X, Y = np.meshgrid(x, y, copy=False)

    X = X.flatten()
    Y = Y.flatten()

    A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
    B = ROI.flatten()
    c, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    X, Y = np.meshgrid(x, y, copy=False)
    s = c[0] + X*c[1] + Y*c[2] + X**2*c[3] + X**2*Y*c[4] + X**2*Y**2*c[5] + Y**2*c[6] + X*Y**2*c[7] + X*Y*c[8]

    return ROI - s


def output_files(MetricsROI, ROIs,ImageFiles,type_label):
    
    nImages,nSamples = MetricsROI.shape
    nImages,nSamples,R_sizeX,R_sizeY = ROIs.shape     
    
    csv_file_name = 'FMP'+str(R_sizeX)+'_'+type_label+'.csv'
    np.savetxt(csv_file_name,np.array(ImageFiles)[np.newaxis,:],delimiter=',',fmt='%s')
    f=open(csv_file_name,'a')                           
    np.savetxt(f,MetricsROI.T,delimiter=',',fmt='%s')
    f.close()    

    
    image_file_name_folder = 'FMP'+str(R_sizeX)+'_'+type_label    
    if not os.path.exists(image_file_name_folder):
        os.makedirs(image_file_name_folder)
    
    
    MetUn = np.unique(MetricsROI)
    lFMP = MetUn[1:21]
    hFMP = MetUn[-20:]
    
    for n in lFMP:
        idx_image, idx_sample = np.where(MetricsROI == n)
        idx_image = idx_image[0]
        idx_sample = idx_sample[0]
    
        image_name = '/LowFMP'+"{:0.2f}".format(n) +'_' +ImageFiles[idx_image][:-4] +'_sample'+str(idx_sample).zfill(4)+'.tif'
        imageio.imwrite(image_file_name_folder +image_name,ROIs[idx_image,idx_sample,:,:].astype(np.float32))
    
    
    for n in hFMP:
        idx_image, idx_sample = np.where(MetricsROI == n)
        idx_image = idx_image[0]
        idx_sample = idx_sample[0]
    
        image_name = '/HighFMP'+"{:0.2f}".format(n) + '_' +ImageFiles[idx_image][:-4] +'_sample'+str(idx_sample).zfill(4)+'.tif'
        imageio.imwrite(image_file_name_folder +image_name,ROIs[idx_image,idx_sample,:,:].astype(np.float32))
    


