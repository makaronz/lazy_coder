"""
LazyCoder Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements-minimal.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="lazycoder",
    version="1.0.0",
    description="Autonomous AI Agent for Cursor IDE with God Mode capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LazyCoder Team",
    author_email="contact@lazycoder.dev",
    url="https://github.com/lazycoder/lazycoder",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lazycoder=cli:cli_main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Tools",
        "Topic :: Artificial Intelligence",
    ],
    keywords="ai, automation, cursor, ide, autonomous, agent, development",
    project_urls={
        "Bug Reports": "https://github.com/lazycoder/lazycoder/issues",
        "Source": "https://github.com/lazycoder/lazycoder",
        "Documentation": "https://docs.lazycoder.dev",
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
)