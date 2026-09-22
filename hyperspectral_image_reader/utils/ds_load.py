'''
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences https://www.iitis.pl

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
************************************************************************

HSI blood classification dataset by M. Romaszewski, P.Glomb, M. Cholewa, A. Sochan 
Institute of Theoretical and Applied Informatics, Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
Dataset DOI: 10.5281/zenodo.3984905

HyperBlood API
Basic loader for dataset files

Warning:
    * By default, data is cleared by removing noisy bands and broken line in the image. 
    * Note that the 'F(2k)' image was captured with different camera. Its bands were interpolated 
    to match remaining images. However, due to spectral range differences between cameras, it has 
    less bands. After cleaning (default) all images have the same matching 113 bands. 

NOISY_BANDS_INDICES = np.array([0,1,2,3,4,48,49,50,121,122,123,124,125,126,127])

@author: mromaszewski@iitis.pl
'''
import warnings
warnings.filterwarnings("ignore")
import unittest
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
# from geotiff import GeoTiff
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler')
# from osgeo import gdal_array

IMAGES = ['A(1)','B(1)','C(1)','D(1)','E(1)','E(7)','E(21)','F(1)','F(1a)','F(1s)','F(2)','F(2k)','F(7)','F(21)'] 

#change this to your DS location
# PATH_DATA = '../'

#------------------------ DATA LOADING ------------------------------------

def get_data(name,remove_bands=True,clean=True):
    """
    Returns HSI data from a datacube
    
    Parameters:
    ---------------------
    name: name
    remove_bands: if True, noisy bands are removed (leaving 113 bands)
    clean: if True, remove damaged line
    
    Returns:
    -----------------------
    data, wavelenghts as numpy arrays (float32)
    """
    # filename = "{}data/{}".format(PATH_DATA,name)
    hsimage = gdal.Open(name)
    hsimage = hsimage.ReadAsArray()
    wavs = np.asarray([376.8200 , 381.7583 , 386.7018 , 391.6505 , 396.6044 , 401.5636 , 406.5280 , 411.4977 , 416.4725 , 421.4525 , 426.4379 , 431.4284 , 436.4241 , 441.4251 , 446.4313 , 451.4427 , 456.4594 , 461.4813 , 466.5084 , 471.5408 , 476.5783 , 481.6211 , 486.6691 , 491.7224 , 496.7808 , 501.8445 , 506.9134 , 511.9876 , 517.0670 , 522.1515 , 527.2413 , 532.3364 , 537.4367 , 542.5422 , 547.6529 , 552.7689 , 557.8901 , 563.0164 , 568.1481 , 573.2849 , 578.4270 , 583.5743 , 588.7269 , 593.8846 , 599.0476 , 604.2158 , 609.3892 , 614.5679 , 619.7518 , 624.9409 , 630.1353 , 635.3348 , 640.5396 , 645.7496 , 650.9649 , 656.1853 , 661.4111 , 666.6420 , 671.8782 , 677.1195 , 682.3661 , 687.6179 , 692.8750 , 698.1372 , 703.4047 , 708.6775 , 713.9554 , 719.2386 , 724.5271 , 729.8207 , 735.1196 , 740.4236 , 745.7329 , 751.0475 , 756.3672 , 761.6923 , 767.0225 , 772.3580 , 777.6987 , 783.0445 , 788.3956 , 793.7520 , 799.1135 , 804.4803 , 809.8524 , 815.2296 , 820.6121 , 825.9998 , 831.3927 , 836.7909 , 842.1942 , 847.6028 , 853.0167 , 858.4358 , 863.8600 , 869.2896 , 874.7243 , 880.1642 , 885.6095 , 891.0598 , 896.5155 , 901.9764 , 907.4425 , 912.9138 , 918.3903 , 923.8721 , 929.3591 , 934.8514 , 940.3488 , 945.8514 , 951.3594 , 956.8725 , 962.3909 , 967.9144 , 973.4432 , 978.9773 , 984.5165 , 990.0610 , 995.6107 , 1001.1656 , 1006.7258 , 1012.2913 , 1017.8618 , 1023.4377 , 1029.0188 , 1034.6050 , 1040.1965 , 1045.7932]) 
    data = np.asarray(hsimage[:,:,:],dtype=np.float32).transpose(1,2,0) # (520, 696, 128)
    
    #removal of damaged sensor line
    fname = name.split('/')[-1].replace('.tif','')
    if clean and fname!='F_2k':
        data = np.delete(data,445,0)
    
    
    if not remove_bands:
        return data,wavs
    return data[:,:,get_good_indices(fname)],wavs[get_good_indices(fname)] 

def get_anno(name,remove_uncertain_blood=True,clean=True):
    """
    Returns annotation (GT) for data files as 2D int numpy array
    Classes:
    0 - background
    1 - blood
    2 - ketchup
    3 - artificial blood
    4 - beetroot juice
    5 - poster paint
    6 - tomato concentrate
    7 - acrtylic paint
    8 - uncertain blood
    
    Parameters:
    ---------------------
    name: name
    clean: if True, remove damaged line
    remove_uncertain_blood: if True, removes class 8 
    
    Returns:
    -----------------------
    annotation as numpy 2D array 
    """
    name = convert_name(name)
    filename = "{}anno/{}".format(PATH_DATA,name)
    anno = np.load(filename+'.npz')['gt']
    #removal of damaged sensor line
    if clean and name!='F_2k':
        anno = np.delete(anno,445,0)
    #remove uncertain blood + technical classes
    if remove_uncertain_blood:
        anno[anno>7]=0 
    else:
        anno[anno>8]=0
           
    return anno    


#------------------------ UTILITY ------------------------------------


def get_good_indices(name=None):
    """
    Returns indices of bands which are not noisy

    Parameters:
    ---------------------
    name: name
    Returns:
    -----------------------
    numpy array of good indices         
    """
    name = convert_name(name)
    if name!='F_2k':
        indices = np.arange(128)
        indices = indices[5:-7]
    else:
        indices = np.arange(116)    
    indices=np.delete(indices,[43,44,45])
    return indices

def convert_name(name):
    """
    Ensures that the name is in the filename format
    Parameters:
    ---------------------
    name: name
    
    Returns:
    -----------------------
    cleaned name
    """
    name = name.replace('(','_')
    name = name.replace(')','')
    return name



def get_rgb(data,wavelengths,gamma=0.7,vnir_bands=[600, 550, 450]):
    """
   Treturns an (over)simplified RGB visualization of HSI data
    
    Parameters:
    ---------------------
    data: data cube as nparray
    annotation: wavelengths - band wavelenghts
    gamma: gamma correction value
    vnir_bands: bands used for RGB
    
    Returns:
    -----------------------
    rgb image as numpy array     
    """
    assert data.shape[2]==len(wavelengths)
    max_data = np.max(data)
    rgb_i = [np.argmin(np.abs(wavelengths - b)) for b in vnir_bands]
    ret = data[:,:,rgb_i].copy()/max_data

    if gamma!=1.0:
        for i in range(3):
            ret[:,:,i]=np.power(ret[:,:,i],gamma)
    
    return ret 

class LoadTest(unittest.TestCase):
    def test_load(self):
        """
        test image loading
        """
        for name in IMAGES:
            data,wavelengths = get_data(name,remove_bands=True)
            anno = get_anno(name)
            self.assertEqual(data.shape[2],113)
            self.assertEqual(data.shape[2],wavelengths.shape[0])
            rgb = get_rgb(data,wavelengths)
            plt.subplot(1,2,1)
            plt.imshow(rgb,interpolation='nearest')
            plt.subplot(1,2,2)
            plt.imshow(anno,interpolation='nearest')
            plt.show()
            plt.close()
            

    def dis_test_indices(self):    
        '''
        Ensure F_2k is loaded correctly 
        '''
        _,wavs = get_data('F_2k',remove_bands=False)
        assert 619.7518 in wavs 
        _,wavs = get_data('F_2k',remove_bands=True)
        assert 619.7518 not in wavs
        _,wavs2 = get_data('F_1',remove_bands=True)
        assert np.sum(wavs-wavs2)==0
        
        data,wavelengths = get_data('F_1',remove_bands=False)
        self.assertEqual(data.shape[2],128)
        self.assertEqual(data.shape[2],wavelengths.shape[0])
        data,wavelengths = get_data('F_2k',remove_bands=False)
        self.assertEqual(data.shape[2],116)
        self.assertEqual(data.shape[2],wavelengths.shape[0])
        anno = get_anno('F_1')
        
        
if __name__ == '__main__':
    unittest.main()
