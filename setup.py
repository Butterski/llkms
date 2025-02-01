from setuptools import find_packages, setup

setup(
    name="llkms",
    version="0.1.0",
    description="A Language Learning Knowledge Management System",
    author="Milosz",
    url="https://github.com/Butterski/llkms",
    packages=find_packages(),
    install_requires=[
        "pytest",
        "langchain",
        "faiss-cpu",
        "python-docx",
        "beautifulsoup4",
        "boto3",
        "python-dotenv",
        "Pillow",
    ],
    entry_points={
        "console_scripts": [
            "llkms=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
