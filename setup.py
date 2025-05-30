from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="repairs_components",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A library of repair components for reinforcement learning environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RepairsComponents-v0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    package_data={"": ["*.stl"]},  # Include any data files if needed
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "genesis-world>=0.2.0",  # Genesis simulator
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "ruff",
            "isort>=5.10.1",
            "mypy>=0.910",
            "pylint>=2.12.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.15.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
