import os
import sys
from setuptools import setup, find_packages

from bmds import __version__


if sys.argv[-1] == 'publish_test':
    os.system('python setup.py sdist upload -r https://testpypi.python.org/pypi')
    os.system('python setup.py bdist_wheel upload -r https://testpypi.python.org/pypi')
    sys.exit()

if sys.argv[-1] == 'publish_production':
    os.system('python setup.py sdist upload')
    os.system('python setup.py bdist_wheel upload')
    sys.exit()


def get_readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='bmds',
    version=__version__,
    description='BMDS',
    long_description=get_readme(),
    url='https://github.com/shapiromatron/bmds',
    author='Andy Shapiro',
    author_email='shapiromatron@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'pytest',
        'enum34',
    ],
    include_package_data=True,
    zip_safe=False
)
