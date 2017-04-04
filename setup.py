# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='zbinner',
    version='0.1.1',
    description='significance based binning optimisation',
    long_description=readme,
    author='Yacine Haddad',
    author_email='yhaddad@cern.ch',
    url='https://github.com/yhaddad/zbinner.git',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
