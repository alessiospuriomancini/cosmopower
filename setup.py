#!/usr/bin/env python

from setuptools import setup, find_packages
import sys
import os
import shutil

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

def read_file(file):
   with open(file) as f:
        return f.read()

long_description = read_file("README.md")

# Determine whether the system is M1/M2 Mac
if 'arm' in os.uname().machine:
    tensorflow = 'tensorflow-metal'
else:
    tensorflow = 'tensorflow>2.0'

setup(classifiers=['Operating System :: OS Independent',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research'
                  ],
      name='cosmopower',
      version='v0.1.3',
      description='Machine Learning - accelerated Bayesian inference',
      long_description_content_type = "text/markdown",
      long_description = long_description,
      author='Alessio Spurio Mancini',
      author_email='a.spuriomancini@ucl.ac.uk',
      license='GNU General Public License v3 (GPLv3)',
      url='https://github.com/alessiospuriomancini/cosmopower',
      packages=find_packages(),
      install_requires=[tensorflow, install_requires],
     )

# cd to parent dir of setup.py
os.chdir(os.path.dirname(os.path.abspath(__file__)))
shutil.rmtree("dist", True)

# Clean up
shutil.rmtree("build", True)
shutil.rmtree("cosmopower.egg-info", True)
shutil.rmtree("__pycache__", True)
shutil.rmtree(".pytest_cache", True)
