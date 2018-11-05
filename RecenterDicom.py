'''
Proteus

S.PIchardo
November 18, 2016
Library to recenter and rotate DICOM datasets
This can also used to calculate the spatial  location of pixels in images
'''
from __future__ import print_function
import numpy as np
import pydicom as dicom
import os
import time
import glob
from scipy.spatial.distance import euclidean

def PrintDicomDirSeries(base_dir):
    '''
    Open a DICOMDIR file located in the base_dir and prints the list of studies and series.
    This is useful to select an specific dataset
    '''
    DICOMDIR=dicom.filereader.read_dicomdir(base_dir+os.sep+'DICOMDIR')
    for patrec in DICOMDIR.patient_records:
        if hasattr(patrec, 'PatientID') and hasattr(patrec, 'PatientsName'):
            print("Patient: {0.PatientID}: {0.PatientsName}".format(patrec))
        studies = patrec.children
        for study in studies:
            print("    Study {0.StudyID}: {0.StudyDate}:"
                  " {0.StudyDescription}".format(study))
            all_series = study.children
            for series in all_series:
                image_count = len(series.children)
                plural = ('', 's')[image_count > 1]

                # Write basic series info and image count

                # Put N/A in if no Series Description
                if 'SeriesDescription' not in series:
                    series.SeriesDescription = "N/A"
                if 'ProtocolName' not in series:
                    series.ProtocolName=series.SeriesDescription
                print(" " * 8 + "Series {0.SeriesNumber}: {0.ProtocolName} : {0.Modality}: {0.SeriesDescription}: "
                      " ({1} image{2})".format(series, image_count, plural))
                print(" " * 10+ "{0.SeriesInstanceUID}".format(series))

def ReadSpecificSeries(base_dir,SeriesInstanceUID,bReturnOnlyFileNames=False):
    '''
    Read the files associated to a UID of series contained in  DICOMDIR file located in  base_dir.
    Function returns a list of pydicom images associated to the series.
    '''
    DICOMDIR=dicom.filereader.read_dicomdir(base_dir+os.sep+'DICOMDIR')
    datasets=[]
    for patrec in DICOMDIR.patient_records:
        studies = patrec.children
        for study in studies:
            all_series = study.children
            for series in all_series:
                image_count = len(series.children)
                plural = ('', 's')[image_count > 1]

                # Write basic series info and image count

                if 'SeriesDescription' not in series:
                        series.SeriesDescription = "N/A"
                if 'ProtocolName' not in series:
                    series.ProtocolName=series.SeriesDescription

                if not (series.SeriesInstanceUID==SeriesInstanceUID):
                    continue

                print(" " * 8 + "Series {0.SeriesNumber}: {0.ProtocolName} : {0.Modality}: {0.SeriesInstanceUID}"
                      " ({1} image{2})".format(series, image_count, plural))

                # Open and read something from each image, for demonstration purposes
                # For simple quick overview of DICOMDIR, leave the following out
                print(" " * 12 + "Reading images...")
                image_records = series.children
                image_filenames = [os.path.join(base_dir, *image_rec.ReferencedFileID)
                                   for image_rec in image_records]

                # slice_locations = [pydicom.read_file(image_filename).SliceLocation
                #                   for image_filename in image_filenames]
                datasets=[]
                for image_filename in image_filenames:
                    image=dicom.read_file(image_filename)

                    if 'ImagePositionPatient' not in image:
                        print ('skipping file {0} since it does not contain image information '.format(image_filename))
                        continue
                    if bReturnOnlyFileNames:
                        datasets.append(image_filename)
                    else:
                        datasets.append(image)
                        patient_names = set(ds.PatientName for ds in datasets)
                        patient_IDs = set(ds.PatientID for ds in datasets)

                # List the image filenames
                #print("\n" + " " * 12 + "Image filenames:")
                #print(" " * 12, end=' ')
                #pprint(image_filenames, indent=12)
    if len(datasets)==0:
        print("Warning: No series was found with the specified series number [{0}] and protocol [{1}]".format(SeriesNumber,ProtocolName))
    return datasets

def ReadImagesInDirectory(base_dir,SeriesInstanceUID,fileExt='dcm'):
        '''
        Read the dicom files stored flattly in a directory.
        '''
        datasets=[]
        SeriesUID=[]

        tempDatasets=[]
        for image_filename in glob.glob(base_dir+os.sep+'*.'+fileExt):
            image=dicom.read_file(image_filename)
            #if 'ImagePositionPatient' not in image:
            #    print ('skipping file {0} since it does not contain image information '.format(image_filename))
            #    continue
            tempDatasets.append(image)
            if image.SeriesInstanceUID not in SeriesUID:
                SeriesUID.append(image.SeriesInstanceUID)



        for image in tempDatasets:
            if not( image.SeriesInstanceUID == SeriesInstanceUID):
                continue
            datasets.append(image)

        if len(datasets)==0:
            print("Warning: No series was found with the specified series number [{0}] and protocol [{1}]".format(SeriesNumber,ProtocolName))
        return datasets

def PrintSeriesInDir(base_dir,fileExt='dcm'):
    '''
    Print series contained in a flat directory, this will be helpful when pushing data via DICOM network
    '''
    SeriesUID=[]
    for image_filename in glob.glob(base_dir+os.sep+'*.'+fileExt):
        image=dicom.read_file(image_filename)
        if 'PerFrameFunctionalGroupsSequence' not in image:
            continue
        if image.SeriesInstanceUID not in SeriesUID:
            if 'SeriesDescription' not in image:
                image.SeriesDescription = "N/A"
            if 'ProtocolName' not in image:
                image.ProtocolName=series.SeriesDescription
            print (image_filename)
            print(" " * 8 + "Series {0.SeriesNumber}: {0.ProtocolName} : {0.Modality}: {0.SeriesDescription}: ".format(image))
            print(" " * 10+ "{0.SeriesInstanceUID}".format(image))


def CalculateSpatialInfo(dataset):
    '''
    Calculate the basic spatial information of all images contained in the dataset list.
    The spatial coordinated of the Top-Left, Top-Right, Bottom-left, Bottom-right, and center of the images are calculated.
    A list of the same length as dataset is returned with dictionary entries

    Coordinates will be returned as RL,AP,FH

    TODO: We need a better way to discriminate between a stacked single 3D image and a collection of DICOM images
    '''
    SpatialInfo=[]
    MatPosOrientation=np.zeros((4,4))
    IndCol=np.zeros((4,1))

    if type(dataset)==list:
        for image in dataset:
            ImagePositionPatient=np.array(image.ImagePositionPatient)
            ImageOrientationPatient=np.array(image.ImageOrientationPatient)
            VoxelSize=np.array(image.PixelSpacing)

            MatPosOrientation[3,3]=1
            MatPosOrientation[0:3,0]=ImageOrientationPatient[0:3]*VoxelSize[0]
            MatPosOrientation[0:3,1]=ImageOrientationPatient[3:]*VoxelSize[1]
            MatPosOrientation[0:3,3]=ImagePositionPatient

            CenterRow=image.Rows/2
            CenterCol=image.Columns/2

            IndCol[0,0]=CenterCol
            IndCol[1,0]=CenterRow
            IndCol[2,0]=0
            IndCol[3,0]=1
            CenterImagePosition=np.dot(MatPosOrientation,IndCol).flatten()[0:3]

            IndCol[0,0]=0
            IndCol[1,0]=0
            IndCol[2,0]=0
            IndCol[3,0]=1
            TopLeft=np.dot(MatPosOrientation,IndCol).flatten()[0:3]

            IndCol[0,0]=image.Columns-1
            IndCol[1,0]=image.Rows-1
            IndCol[2,0]=0
            IndCol[3,0]=1
            BottomRight=np.dot(MatPosOrientation,IndCol).flatten()[0:3]

            IndCol[0,0]=0
            IndCol[1,0]=image.Rows-1
            IndCol[2,0]=0
            IndCol[3,0]=1
            BottomLeft=np.dot(MatPosOrientation,IndCol).flatten()[0:3]

            IndCol[0,0]=image.Columns-1
            IndCol[1,0]=0
            IndCol[2,0]=0
            IndCol[3,0]=1
            TopRight=np.dot(MatPosOrientation,IndCol).flatten()[0:3]

            SpatialInfo.append({'CenterImagePosition':CenterImagePosition,'TopLeft':TopLeft,'BottomRight':BottomRight,'BottomLeft':BottomLeft,'TopRight':TopRight})
    else:
        for SequenceInfo in dataset.PerFrameFunctionalGroupsSequence:
            ImagePositionPatient=np.array(SequenceInfo.PlanePositionSequence[0].ImagePositionPatient)
            ImageOrientationPatient=np.array(SequenceInfo.PlaneOrientationSequence[0].ImageOrientationPatient)
            VoxelSize=np.array(SequenceInfo.PixelMeasuresSequence[0].PixelSpacing)

            MatPosOrientation[3,3]=1
            MatPosOrientation[0:3,0]=ImageOrientationPatient[0:3]*VoxelSize[0]
            MatPosOrientation[0:3,1]=ImageOrientationPatient[3:]*VoxelSize[1]
            MatPosOrientation[0:3,3]=ImagePositionPatient

            CenterRow=dataset.Rows/2
            CenterCol=dataset.Columns/2

            IndCol[0,0]=CenterCol
            IndCol[1,0]=CenterRow
            IndCol[2,0]=0
            IndCol[3,0]=1
            CenterImagePosition=np.dot(MatPosOrientation,IndCol).flatten()[0:3]

            IndCol[0,0]=0
            IndCol[1,0]=0
            IndCol[2,0]=0
            IndCol[3,0]=1
            TopLeft=np.dot(MatPosOrientation,IndCol).flatten()[0:3]

            IndCol[0,0]=dataset.Columns-1
            IndCol[1,0]=dataset.Rows-1
            IndCol[2,0]=0
            IndCol[3,0]=1
            BottomRight=np.dot(MatPosOrientation,IndCol).flatten()[0:3]

            IndCol[0,0]=0
            IndCol[1,0]=dataset.Rows-1
            IndCol[2,0]=0
            IndCol[3,0]=1
            BottomLeft=np.dot(MatPosOrientation,IndCol).flatten()[0:3]

            IndCol[0,0]=dataset.Columns-1
            IndCol[1,0]=0
            IndCol[2,0]=0
            IndCol[3,0]=1
            TopRight=np.dot(MatPosOrientation,IndCol).flatten()[0:3]

            SpatialInfo.append({'CenterImagePosition':CenterImagePosition,'TopLeft':TopLeft,'BottomRight':BottomRight,'BottomLeft':BottomLeft,'TopRight':TopRight})
    return SpatialInfo

def CalculateOffset(image,xIndex,yIndex):
    '''
    Calculate the spatial information of a single location pixel xIndex, yIndexof the image

    Coordinates will be returned as RL,AP,FH
    '''
    if xIndex <0 or xIndex > image.Columns:
        raise ValueError("xIndex must be a value between 0 and " + str(image.Columns))
    if yIndex <0 or yIndex > image.Rows:
        raise ValueError("xIndex must be a value between 0 and " + str(image.Rows))

    MatPosOrientation=np.zeros((4,4))
    IndCol=np.zeros((4,1))
    ImagePositionPatient=np.array(image.ImagePositionPatient)
    ImageOrientationPatient=np.array(image.ImageOrientationPatient)
    VoxelSize=np.array(image.PixelSpacing)

    MatPosOrientation[3,3]=1
    MatPosOrientation[0:3,0]=ImageOrientationPatient[0:3]*VoxelSize[0]
    MatPosOrientation[0:3,1]=ImageOrientationPatient[3:]*VoxelSize[1]
    MatPosOrientation[0:3,3]=ImagePositionPatient

    IndCol[0,0]=xIndex
    IndCol[1,0]=yIndex
    IndCol[2,0]=0
    IndCol[3,0]=1
    ImagePosition=np.dot(MatPosOrientation,IndCol).flatten()[0:3]
    return ImagePosition

def CalculateOffsetList(image,xIndex,yIndex):
    '''
    Calculate the spatial information of all pixel location indicated in xIndex, yIndex.
    xIndex are assumed to be numpy array.

    Coordinates will be returned as RL,AP,FH
    '''
    if np.any(xIndex <0) or np.any(xIndex > image.Columns):
        raise ValueError("xIndex must be a value between 0 and " + str(image.Columns))
    if np.any(yIndex <0) or np.any(yIndex > image.Rows):
        raise ValueError("xIndex must be a value between 0 and " + str(image.Rows))

    MatPosOrientation=np.zeros((4,4))
    IndCol=np.zeros((4,xIndex.flatten().shape[0]))
    ImagePositionPatient=np.array(image.ImagePositionPatient)
    ImageOrientationPatient=np.array(image.ImageOrientationPatient)
    VoxelSize=np.array(image.PixelSpacing)

    MatPosOrientation[3,3]=1
    MatPosOrientation[0:3,0]=ImageOrientationPatient[0:3]*VoxelSize[0]
    MatPosOrientation[0:3,1]=ImageOrientationPatient[3:]*VoxelSize[1]
    MatPosOrientation[0:3,3]=ImagePositionPatient

    IndCol[0,:]=xIndex.flatten()
    IndCol[1,:]=yIndex.flatten()
    IndCol[2,:]=0
    IndCol[3,:]=1
    ImagePosition=np.dot(MatPosOrientation,IndCol)
    RLCoord=ImagePosition[0,:].reshape(xIndex.shape)
    APCoord=ImagePosition[1,:].reshape(xIndex.shape)
    HFCoord=ImagePosition[2,:].reshape(xIndex.shape)
    return RLCoord,APCoord,HFCoord

def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
    False for no.

    'resp' should be set to the default value assumed by the caller when
    user simply types ENTER.

    >>> confirm(prompt='Create Directory?', resp=True)
    Create Directory? [y]|n:
    True
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y:
    False
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: y
    True

    """

    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')

    while True:
        ans = raw_input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print ('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False

def autolim(img,tol = [.01 ,.99]):

    tol_low = tol[0]
    tol_high = tol[1]

    if img.dtype==np.dtype('uint8'):
        nbins = 256;
    else:
        nbins = 65536;

    if tol_low < tol_high:
        ilowhigh = np.zeros(2)
        N = np.histogram(img.flatten(),bins=nbins)[0]
        cdf = np.cumsum(N).astype(np.float)/np.sum(N)
        ilow = np.where(cdf > tol_low)[0][0]
        ihigh = np.where(cdf >= tol_high)[0][0]

        if ilow == ihigh:
            ilowhigh[0] = 1
            ilowhigh[1] = nbins
        else:
            ilowhigh[0] = ilow
            ilowhigh[1] = ihigh

        lowhigh = (ilowhigh - 1)/(nbins-1);

    else:
        raise ValueError('wrong tolerance')
    return lowhigh

def correctcontrast(img,lowhighin,lowhighout=[0,1],gamma=1.):
    lIn=lowhighin[0]
    hIn=lowhighin[1]
    lOut=lowhighout[0]
    hOut=lowhighout[1]
    g=gamma
    d=1.

    imgd=img.astype(np.float)/(2**16)

    imgd[imgd>hIn]=hIn
    imgd[imgd<lIn]=lIn

    imgd=( (imgd - lIn) / (hIn - lIn))**g
    imgd=imgd * (hOut-lOut) + lOut

    imgd=np.floor(imgd*(2**16))
    imgd[imgd<0]=0
    imgd[imgd>=2**16]=2**16-1

    return imgd.astype(np.uint16)


class SelectionMarkersUI():
    '''
    This class is to be used in a notebook to navigate and select a location in the image to recenter the dataset.

    The basics of operation is to load a dataset and apply modifcations to the images and the save them in a directory

    VERY IMPORTANT. The modifications to the images are accumulative.
    '''
    SpatialInfo=[]
    Images=[]
    Offset=np.zeros(3)
    TopContrast=1.0
    base_dir=''
    SeriesInstanceUID=''
    def __init__(self,base_dir,SeriesInstanceUID='',DICOMDIR=True):
        '''
        The default constructor reads the data from the DICOMDIR selected by its series UID
        '''
        self.base_dir=base_dir
        self.SeriesInstanceUID=SeriesInstanceUID
        if DICOMDIR:
            self.Images=ReadSpecificSeries(base_dir,SeriesInstanceUID)
        else:
            self.Images=ReadImagesInDirectory(base_dir)
        self.SpatialInfo=CalculateSpatialInfo(self.Images)

    def ShowData(self,TopContrast,nSlice,xCoord,yCoord,bShow3D=False):
        '''
        It shows a simple image with a marker showed in the pixel and spatial location (if the 3D opton is enabled)

        Very imporant, it stores in the object the Offset in spatial coordinates of the xCoord and yCoord. This offset can be used for spatial recenter.
        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.axes3d import Axes3D
        Offset=CalculateOffset(self.Images[nSlice],xCoord,yCoord)
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        sp=self.SpatialInfo[nSlice]
        IM=self.Images[nSlice].pixel_array-self.Images[nSlice].pixel_array.min()

        plt.imshow(IM,cmap=plt.cm.gray,vmax=IM.max()*TopContrast)
        plt.xlabel('x-pixels')
        plt.ylabel('y-pixels')
        plt.xlim([0,self.Images[nSlice].pixel_array.shape[1]-1])
        plt.ylim([self.Images[nSlice].pixel_array.shape[0]-1,0])
        plt.plot(xCoord,yCoord,'+r',ms=18,mew=3)

        if bShow3D:

            ax=plt.subplot(1,2,2,projection='3d')

            sp=self.SpatialInfo[nSlice]

            XX,YY=np.meshgrid(np.arange(self.Images[nSlice].Columns),np.arange(self.Images[nSlice].Rows))
            RLCoord,APCoord,HFCoord=CalculateOffsetList(self.Images[nSlice],XX,YY)
            ax.plot_surface(RLCoord,APCoord,HFCoord,  facecolors=plt.cm.gray(IM), shade=False,linewidth=0,rstride=5, cstride=5      )
            ax.set_xlabel('RL (mm)')
            ax.set_ylabel('AP (mm)')
            ax.set_zlabel('HF (mm)')
            ax.plot([Offset[0]],[Offset[1]],[Offset[2]],marker='+',color='r',ms=18,mew=3)

        plt.show()


        self.Offset=Offset
        self.TopContrast=TopContrast
        print(Offset)

    def SetOffset(self,nSlice,xCoord,yCoord):
        '''
        It calculates the reqiured offset indicating directly the number of slice , xCooord and yCoord in pixels.
        This is useful for batch processing or when it's already decided for the user to apply the same location for the recenter
        '''
        self.Offset=CalculateOffset(self.Images[nSlice],xCoord,yCoord)


    def InteractiveSel(self):
        '''
        This uses jupyter notebook widgets to show a minimalistic UI to navigate among the slices
        '''
        from ipywidgets import interact, IntSlider,FloatSlider

        Sz=self.Images[0].pixel_array.shape
        interact(self.ShowData,nSlice=IntSlider(value=len(self.Images)/2,min=0,max=len(self.Images)-1,step=1,width='600px',continuous_update=False),
                 xCoord=IntSlider(value=Sz[1]/2,min=0,max=Sz[1]-1,step=1,width='600px',continuous_update=False),
                 yCoord=IntSlider(value=Sz[0]/2,min=0,max=Sz[0]-1,step=1,width='600px',continuous_update=False),
                 TopContrast=FloatSlider(value=0.5,min=0.1,max=1.,width='600px',continuous_update=False))

    def Rotation(self,angleDeg,axis='HF'):
        '''
        It applies the indicated rotation in degrees along the specified axis (HF, AP, or LR) to all images in the dataset
        This is destructive step, the original orientation is lost unless data is reloaed (using a new instance of the class)
        '''
        if axis not in ['HF','LR','AP']:
            raise ValueError("axis must be either 'HF', 'LR', or 'AP'")
        B=np.zeros((3,3))
        RotMatrix=np.zeros((3,3))
        angleRad=angleDeg/180.*np.pi

        cosRad=np.around(np.cos(angleRad),3) # we round at 3 decimals to avoid some nasty rounding errors for rotations at 90,180 and 270 degrees
        sinRad=np.around(np.sin(angleRad),3) # we round at 3 decimals to avoid some nasty rounding errors for rotations at 90,180 and 270 degrees

        if axis=='HF':
            RotMatrix[0,0]= cosRad
            RotMatrix[0,1]=-sinRad
            RotMatrix[1,0]=sinRad
            RotMatrix[1,1]=cosRad
            RotMatrix[2,2]=1
        elif axis=='LR':
            RotMatrix[0,0]=cosRad
            RotMatrix[0,2]=sinRad
            RotMatrix[2,0]=-sinRad
            RotMatrix[2,2]=cosRad
            RotMatrix[1,1]=1
        else:
            RotMatrix[1,1]=cosRad
            RotMatrix[1,2]=-sinRad
            RotMatrix[2,1]=sinRad
            RotMatrix[2,2]=cosRad
            RotMatrix[0,0]=1

        for image in self.Images:
            RowCol=np.array(image.ImageOrientationPatient)
            B[0:3,0]=RowCol[0:3]
            B[0:3,1]=RowCol[3:]
            B[0:3,2]=np.cross(RowCol[0:3],RowCol[3:])
            B=np.dot(RotMatrix,B)
            image.ImageOrientationPatient[0]=B[0,0]
            image.ImageOrientationPatient[1]=B[1,0]
            image.ImageOrientationPatient[2]=B[2,0]
            image.ImageOrientationPatient[3]=B[0,1]
            image.ImageOrientationPatient[4]=B[1,1]
            image.ImageOrientationPatient[5]=B[2,1]

            Position=np.dot(RotMatrix,np.array(image.ImagePositionPatient))
            image.ImagePositionPatient[0]=Position[0]
            image.ImagePositionPatient[1]=Position[1]
            image.ImagePositionPatient[2]=Position[2]

    def ConvertCTtoMRI(self, refMRIImagePath,SkipImages=2,bAdjustContrastParam=False,lowhighin=[0.,1.],lowhighout=[0.,1.],gamma=0.4):
        '''
        We just hack the metadata to fool the Sonalleve software.
        CT Datasets often contain navigation images in the first 1 to 3 images that are almost impossible to detect using only metadata, so we'll skip them

        bAdjustContrastParam is used to indicate to correct contrast of CT image to make it more readable on the Sonalleve SW planning.
        If not corrected, some regions appear hyperintense.
        if bAdjustContrastParam is True, lowhighin, lowhighout and gamma are used to adjust the contrast. Default values seem to produce reasonable results
        good enough to visualize targets.
            lowhighin indicates the normalized range of the datatype dynamic to preseve from the input image. [0,1] means that values from 0 to 2^16-1 will be kept. It must have a values from 0 to 1, and lowhighin[1]>lowhighin[0]
            lowhighout indicates the normalized range of the datatype dynamic to strecth the data. [0,0.15] means that the input range will be stretched in values ranging from 0 to (0.15)*2^16-1.
            gamma is the exponent coefficient used to adjust background level. Values inferior to 1 increase the lumuninosity.
        '''

        bFirstMissingReconstructionDiameter=False
        bFirstMissingSpacingBetweenSlices=False
        bFirstMissingSliceLocation=False

        print ('Removing the first ', SkipImages, ' images ... remember this process is destructive')
        self.Images=self.Images[SkipImages:]

        ImagesInAcquisition=len(self.Images)

        for nImage in xrange(len(self.Images)):
            image = self.Images[nImage]
            refMRIImage=dicom.read_file(refMRIImagePath)

            refMRIImage.SliceThickness=image.SliceThickness

            if 'ReconstructionDiameter' in image:
                refMRIImage.ReconstructionDiameter=image.ReconstructionDiameter
            else:
                if bFirstMissingReconstructionDiameter==False:
                    print ("ReconstructionDiameter ( 0018,1100) no present in DICOMS, we'll reconstruct it using Rows, Columns and PixelSpacing")
                    bFirstMissingReconstructionDiameter=True
                    cornerLT=CalculateOffset(image,0,0)
                    cornerLB=CalculateOffset(image,0,int(image.Rows)-1)
                    cornerRT=CalculateOffset(image,int(image.Columns)-1,1)
                    ReconstructionDiameter1=np.around(euclidean(cornerLT,cornerLB),2)
                    ReconstructionDiameter2=np.around(euclidean(cornerLT,cornerRT),2)
                    if ReconstructionDiameter1!=ReconstructionDiameter2:
                        print('Non squared FOV =',ReconstructionDiameter2,ReconstructionDiameter2, ' using larger of those...' )
                        if ReconstructionDiameter2 > ReconstructionDiameter1:
                            ReconstructionDiameter=ReconstructionDiameter2
                        else:
                            ReconstructionDiameter=ReconstructionDiameter1
                    else:
                        ReconstructionDiameter=ReconstructionDiameter1
                    print ('ReconstructionDiameter',ReconstructionDiameter)
                refMRIImage.ReconstructionDiameter=ReconstructionDiameter

            if 'SpacingBetweenSlices' in image:
                refMRIImage.SpacingBetweenSlices=image.SpacingBetweenSlices
            else:
                if bFirstMissingSpacingBetweenSlices==False:
                    if nImage != 0:
                        raise ValueError('How come we are detecting SpacingBetweenSlices is not present from the very first image??? Are you sure this is a single dataset?')
                    print ("SpacingBetweenSlices (0018,0088) is not present in DICOMS, we'll reconstruct it using calculating euclidean distance from two ImagePositionPatient entries")
                    bFirstMissingSpacingBetweenSlices=True
                    SpacingBetweenSlices=np.around(euclidean(np.array(image.ImagePositionPatient),np.array(self.Images[nImage+1].ImagePositionPatient)),2)
                    print ('SpacingBetweenSlices',SpacingBetweenSlices)
                refMRIImage.SpacingBetweenSlices=SpacingBetweenSlices

            if 'SliceLocation' in image:
               refMRIImage.SliceLocation=image.SliceLocation
            else:
               if bFirstMissingSliceLocation==False:
                   if nImage != 0:
                       raise ValueError('How come we are detecting SliceLocation is not present from the very first image??? Are you sure this is a single dataset?')
                   print ("SliceLocation  (0020, 1041) is not present in DICOM, we'll reconstruct it using SpacingBetweenSlices")
                   SliceLocation=0.0
                   bFirstMissingSliceLocation=True
               refMRIImage.SliceLocation=SliceLocation
               SliceLocation+=float(refMRIImage.SpacingBetweenSlices)

            refMRIImage.ImagesInAcquisition=ImagesInAcquisition
            refMRIImage.PixelSpacing=image.PixelSpacing
            refMRIImage.Rows=image.Rows
            refMRIImage.Columns=image.Columns

            refMRIImage.ImagePositionPatient = image.ImagePositionPatient;
            refMRIImage.ImageOrientationPatient = image.ImageOrientationPatient;
            if 'SeriesDescription' is image:
                refMRIImage.SeriesDescription='From CT' + image.SeriesDescription
            elif [0x0033,0x1004] in image:
                refMRIImage.SeriesDescription='From CT' + str(image[0x0033,0x1004])
            else:
                refMRIImage.SeriesDescription='No Series Decription in CT'

            refMRIImage.PatientID = image.PatientID
            refMRIImage.PatientName=image.PatientName

            if bAdjustContrastParam:
                sImg=correctcontrast(image.pixel_array,lowhighin,lowhighout,gamma)
                refMRIImage.pixel_array=sImg
                refMRIImage.PixelData=sImg.tostring()
            else:
                refMRIImage.pixel_array=image.pixel_array
                refMRIImage.PixelData=image.PixelData

            refMRIImage.RescaleSlope=image.RescaleSlope
            refMRIImage.RescaleIntercept=image.RescaleIntercept


            self.Images[nImage] = refMRIImage

    def ApplyOffset(self):
        '''
        It applies the offset to the metadata. This is a destructive/accumulative process. It can't be reversed unless data is reloaded (using a new instance of the class)
        This is destructive step, the original position is lost unless data is reloaed (using a new instance of the class)
        '''
        for image in self.Images:
            for nin in xrange(3):
                image.ImagePositionPatient[nin]=float(image.ImagePositionPatient[nin])-self.Offset[nin]


    def ExportData(self,exportDir,bDeleteFilesFirst=False):
        '''
        Save the data in the specified directory base_dir. If the directory does not exists it will be created.
        To be sure the data can be easily visualized by any conformal DICOM processor, new UIDs are generated for the series and the images.
        This is destructive step, the original UIDs are lost unless data is reloaed (using a new instance of the class)

        '''
        if os.path.exists(exportDir)==False:
            os.makedirs(exportDir)
        else:
            if bDeleteFilesFirst:
                confirmDelete = confirm('Please confirm deleting files in ' + exportDir)
                if confirmDelete:
                    filelist = glob.glob(exportDir + os.sep + '*.dcm')
                    for f in filelist:
                        os.remove(f)
                else:
                    print ('Canceling operation, no export was done')
                    return
        NewSeriesInstanceUID=dicom.uid.generate_uid()

        NewSeriesNumber=int(self.Images[0].SeriesNumber)+np.random.randint(10000)
        time.sleep(0.01)
        NewImages=[]
        for image in self.Images:
            image.SeriesDescription=image.SeriesDescription+' offset (LR,AP,AH)= ' +np.array2string(self.Offset,formatter={'float_kind':'{0:2.1f}'.format})
            image.SeriesInstanceUID=NewSeriesInstanceUID
            image.SOPInstanceUID=dicom.uid.generate_uid()
            image.SeriesNumber=NewSeriesNumber
            if image.PatientID=='': #for anonmyzed data we need to put something
                image.PatientID = '12345';
            if image.PatientName=='': #for anonmyzed data we need to put something
                image.PatientName.FamilyName = 'test'

            time.sleep(0.01)
        for n in xrange(len(self.Images)):
            name=exportDir +os.sep+'IMG%03i.dcm' %(n)
            self.Images[n].save_as(name)

    def ExportDataToDicomNetwork(self,aec = "HIFU_CONSOLE",aet = "SCANNER",ip = "127.0.0.1",port = 4001):
        from netdicom import AE
        from netdicom.SOPclass import StorageSOPClass, MRImageStorageSOPClass, VerificationSOPClass
        from dicom.UID import ExplicitVRLittleEndian, ImplicitVRLittleEndian, \
            ExplicitVRBigEndian
        def OnAssociateResponse(association):
            print ("Association response received")
        # create application entity
        ts = [
            ExplicitVRLittleEndian,
            ImplicitVRLittleEndian,
            ExplicitVRBigEndian
            ]
        MyAE = AE(aet, 9999, [MRImageStorageSOPClass, VerificationSOPClass],[],ts)
        MyAE.OnAssociateResponse = OnAssociateResponse
        # remote application entity
        RemoteAE = dict(Address=ip, Port=port, AET=aec)
        # create association with remote AE

        print ("Request association")
        assoc = MyAE.RequestAssociation(RemoteAE)
        if not assoc:
            raise ValueError("Could not establish association")

        # perform a DICOM ECHO, just to make sure remote AE is listening
        print ("DICOM Echo ... ")
        st = assoc.VerificationSOPClass.SCU(1)
        print ('done with status "%s"' % st)

        NewSeriesInstanceUID=dicom.uid.generate_uid(prefix=self.Images[0].InstanceCreatorUID)
        NewSeriesNumber=int(self.Images[0].SeriesNumber)+np.random.randint(10000)
        time.sleep(0.01)

        for image in self.Images:
            image.SeriesDescription=image.SeriesDescription+' offset (LR,AP,AH)= ' +np.array2string(self.Offset,formatter={'float_kind':'{0:2.1f}'.format})
            image.SeriesInstanceUID=NewSeriesInstanceUID
            image.SOPInstanceUID=dicom.uid.generate_uid(prefix=image.InstanceCreatorUID)
            image.SeriesNumber=NewSeriesNumber
            time.sleep(0.01)

            try:
                print ('sending image...')
                st = assoc.SCU(image, 1)
                print ('done with status "%s"' % st)
            except:
                raise
                print ("problem", d.SOPClassUID)

        print ("Release association")

        assoc.Release(0)



        # done

        MyAE.Quit()
