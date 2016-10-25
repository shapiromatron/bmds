import os
import sys
from setuptools import setup, find_packages


if sys.argv[-1] == 'publish_test':
    os.system('python setup.py sdist upload -r https://testpypi.python.org/pypi')
    os.system('python setup.py bdist_wheel upload -r https://testpypi.python.org/pypi')
    sys.exit()

if sys.argv[-1] == 'publish_production':
    os.system('python setup.py sdist upload')
    os.system('python setup.py bdist_wheel upload')
    sys.exit()


def get_version():
    d = {}
    execfile('./bmds/__init__.py', d)
    return d['__version__']


def get_readme():
    with open('README.rst') as f:
        return f.read()

requirements = [
    'requests',
]

test_requirements = [
    'pytest',
]

setup(
    name='bmds',
    version=get_version(),
    description='Software development kit for US EPA\'s Benchmark dose modeling software (BMDS)',
    long_description=get_readme(),
    url='https://github.com/shapiromatron/bmds',
    author='Andy Shapiro',
    author_email='shapiromatron@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
    tests_require=test_requirements,
    include_package_data=True,
    zip_safe=False
)
