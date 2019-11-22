#!/usr/bin/env python

from setuptools import setup, find_packages

def _read(fname):
    try:
        with open(fname) as fobj:
            return fobj.read()

    except IOError:
        return ''


requirements = [
    'numpy',
    'scipy',
]

setup(
    name='ptfidf',
    version='0.1.1',
    description='A trainable variant of tf-idf matching',
    long_description=_read("Readme.md"),
    author='Matthias Ossadnik',
    author_email='ossadnik.matthias@gmail.com',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url="https://github.com/mossadnik/probabilistic-tf-idf.git",
    setup_requires=['pytest-runner'],
    install_requires=requirements,
    tests_require=['pytest'],
    test_suite='tests',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
