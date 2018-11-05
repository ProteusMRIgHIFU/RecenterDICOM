# RecenterDICOM
Scripts to recenter and rotate DICOM data for treatment planning evaluation

## Disclaimer

This tool is only for ergonomic evaluation purposes. By all means, under no circumstance  this tool should be used for real-life treatment planning purposes, either pre-clinic or clinic. This is a tool only to be used for research purposes.

## Requisites

Any Python 2.7 distribution such as Canopy or Anaconda should be enough. Jupyter notebook with ipywdigets must be functional to use the provided notebooks. The following package must be installed with pip
```
  pip install pydicom
```

## Description
This is a simple tool that modifies the DICOM metadata from exported DICOM series to "virtually" translate-rotate medical images. This is only for testing purposes to have a first, very simplistic, ergonomic test for image-guided interventions.

The core of the functions are in the `RecenterDicom.py` file and two examples of use are provided in the `RecenterCT-To-MRI-BASE.ipynb` and `RecenterMRI-BASE.ipynb` notebooks. Both notebooks use Jupyter ipywdigets to have a simplistic graphical inspection of the location to be used to recenter the images. These notebooks are prepared to export data into the Profound Sonalleve MRI-guided Focused Ultrasound therapy system. However, the notebooks can be easily customized for any other application.  

The `RecenterCT-To-MRI-BASE.ipynb` includes some extra steps to be aware when dealing with CT data. In this example, the  Profound Sonalleve does not accept CT data in their treatment planning. `RecenterCT-To-MRI-BASE.ipynb` includes some extra *hacks* to make the CT dataset appears as an MRI dataset, then the treatment planning software can accept it.

## Demo data

A simple MRI imaging dataset is available in Zenodo at
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1478091.svg)](https://doi.org/10.5281/zenodo.1478091)
