import setuptools

setuptools.setup(
    name='PianoNet',
    version='0.0.1',
    author='Thomas Angsten',
    description="A deep neural network for generating piano compositions.",
    url="https://github.com/angsten/pianonet",
    packages=setuptools.find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
    python_requires='>=3.7',
)