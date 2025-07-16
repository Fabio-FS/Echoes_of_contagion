from setuptools import setup, find_packages

setup(
    name="echoes-of-contagion",
    version="0.1.0",
    author="Fabio-FS", 
    description="Social media opinion dynamics and disease spread simulation",
    url="https://github.com/Fabio-FS/Echoes_of_contagion",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "python-igraph>=0.9.0", 
        "numba>=0.56.0",
    ],
)