from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='pytopics',
    version='0.1',
    description='Topic Models for short texts in Python',
    url='https://github.com/lambdaofgod/pytopics',
    author='Jakub Bartczuk',
    packages=find_packages(),
    install_requires=requirements
)
