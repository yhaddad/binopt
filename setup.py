#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'scipy',
    'numpy',
    'matplotlib',
    'pip>=8.1.2',
    'bumpversion>=0.5.3',
    'wheel>=0.29.0',
    'watchdog>=0.8.3',
    'flake8>=2.6.0',
    'coverage>=4.1',
    'Sphinx>=1.4.8',
    'PyYAML>=3.11'
]

# test_requirements = [
#     # TODO: put package test requirements here
# ]

test_requirements = requirements

setup(
    name='binopt',
    version='0.1.0',
    description="Categorisation of labeled data",
    long_description=readme + '\n\n' + history,
    author="Yacine Haddad",
    author_email='yhaddad@cern.ch',
    url='https://github.com/yhaddad/binopt',
    packages=[
        'binopt',
    ],
    package_dir={'binopt':
                 'binopt'},
    entry_points={
        'console_scripts': [
            'binopt=binopt.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='binopt',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
