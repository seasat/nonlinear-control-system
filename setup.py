
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='nonlinear_control_system',
    version='0.1.0',
    description='Package to simulate a nonlinear control system for a spacecraft',
    long_description=readme,
    author='Lars Blümler',
    author_email='lblumler@tudelft.nl',
    url='https://github.com/seasat/nonlinear-control-system',
    license=license,
    packages=find_packages(exclude=('test', 'doc'))
)
