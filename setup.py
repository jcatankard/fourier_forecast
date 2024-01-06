from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='fourier_forecast',
    version='0.0.3',
    packages=find_packages(),
    url='https://github.com/jcatankard/fourier_forecast',
    author='Josh Tankard',
    description='Time-series forecasting with Fourier series',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'plotly'],
)