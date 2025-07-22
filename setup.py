"""Setup configuration for ND2 Analysis Pipeline package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = this_directory / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="nd2-analysis-pipeline",
    version="1.0.0",
    author="ND2 Analysis Team",
    author_email="contact@nd2analysis.com",
    description="Professional-grade, cross-platform package for analyzing multi-channel ND2 microscopy images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clruan/nd2-analysis-pipeline",
    py_modules=[
        'main', 'visualize', 'config', 'data_models', 
        'image_processing', 'excel_output', 'visualization', 
        'processing_pipeline'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.812',
        ],
        'jupyter': [
            'jupyter>=1.0',
            'ipywidgets>=7.0',
            'plotly>=5.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'nd2-analysis=main:main',
            'nd2-visualize=visualize:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['examples/configs/*.json', 'examples/*.md'],
    },
    zip_safe=False,
    keywords="microscopy, image-analysis, nd2, biomedical, research, visualization",
    project_urls={
        "Bug Reports": "https://github.com/clruan/nd2-analysis-pipeline/issues",
        "Source": "https://github.com/clruan/nd2-analysis-pipeline",
        "Documentation": "https://github.com/clruan/nd2-analysis-pipeline/blob/main/README.md",
    },
)
