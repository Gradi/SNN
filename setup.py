from setuptools import setup, find_packages

setup(
    name="snn",

    version="0.1a0",

    description="Stupid Neural Network lib",

    long_description="SNN is my diploma project. Goal is to create simple library"
                     "which allows you to construct neural networks.",

    author="Maksim Kislyakov",
    author_email="Kislaykov.Maksim@ya.ru",

    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.5"
    ],
    packages=find_packages(),
    install_requires=["numpy"]
)