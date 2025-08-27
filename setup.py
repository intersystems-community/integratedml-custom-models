"""
IntegratedML Flexible Model Integration Framework Setup
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    """Read README.md for long description"""
    here = os.path.abspath(os.path.dirname(__file__))
    try:
        with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "IntegratedML Flexible Model Integration Framework"

# Read requirements.txt for dependencies
def read_requirements():
    """Read requirements.txt for dependencies"""
    here = os.path.abspath(os.path.dirname(__file__))
    try:
        with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    except FileNotFoundError:
        return []

setup(
    name="integratedml-demos",
    version="1.0.0",
    description="Comprehensive demo portfolio showcasing IntegratedML's flexible model integration capability",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="InterSystems Corporation",
    author_email="support@intersystems.com",
    url="https://github.com/intersystems/integratedml-demos",
    project_urls={
        "Bug Tracker": "https://github.com/intersystems/integratedml-demos/issues",
        "Documentation": "https://integratedml-demos.readthedocs.io/",
        "Source Code": "https://github.com/intersystems/integratedml-demos",
    },
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
            "pytest-cov>=3.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "integratedml-demo=shared.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
        "shared.data": ["*.csv", "*.json"],
        "demos": ["**/*.sql", "**/*.ipynb"],
    },
    keywords=[
        "integratedml", 
        "machine learning", 
        "database", 
        "sql", 
        "scikit-learn",
        "ensemble models",
        "flexible model integration",
        "intersystems"
    ],
    zip_safe=False,
)