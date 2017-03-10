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
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
