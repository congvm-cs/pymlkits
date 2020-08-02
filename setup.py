from setuptools import setup, find_packages
from pymlkits import __version__
from os import path

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pymlkits',
    version=__version__,
    description='pymlkits',
    packages=find_packages(),
    package_data={'pymlkits': ['faces/*']},
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    author='congvm',
    author_email='congvm.it@gmail.com',
    license='MIT',
    zip_safe=True,
    install_requires=requirements,
    include_package_data=True,
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
)
