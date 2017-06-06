import re
from setuptools import setup, find_packages


def get_version():
    regex = r"""^__version__ = '(.*)'$"""
    with open('bmds/__init__.py', 'r') as f:
        text = f.read()
    return re.findall(regex, text, re.MULTILINE)[0]


def get_readme():
    with open('README.rst') as f:
        return f.read()


requirements = [
    'six',
    'requests',
    'numpy',
    'pandas',
    'openpyxl',
    'matplotlib',
    'scipy',
    'simple-settings',
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
    tests_require=['pytest-runner', 'pytest', 'pytest-mpl'],
    # List additional groups of dependencies here
    # (e.g. development dependencies).
    # You can install these using the following syntax, for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': [
            'wheel',
            'sphinx',
            'watchdog',
        ],
        'test': [
            'pytest',
            'pytest-runner',
            'pytest-mpl',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
