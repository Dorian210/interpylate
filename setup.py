from setuptools import setup, find_packages

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='interpylate', 
    version='0.0.1', 
    url='https://github.com/Dorian210/interpylate', 
    author='Dorian Bichet', 
    author_email='dbichet@insa-toulouse.fr', 
    description='A package for N-linear regular grid interpolation', 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    packages=find_packages(), 
    install_requires=['numpy', 'numba'], 
    classifiers=['Programming Language :: Python :: 3', 
                 'Operating System :: OS Independent', 
                 'License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)'], 
    
)