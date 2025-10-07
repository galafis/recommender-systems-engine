from setuptools import setup, find_packages

setup(
    name="recommender-systems-engine",
    version="1.0.0",
    author="Gabriel Demetrios Lafis",
    author_email="gabriel.lafis@example.com",
    description="Professional data science repository",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/galafis/recommender-systems-engine",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=open("requirements.txt").read().splitlines(),
)
