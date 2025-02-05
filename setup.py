from setuptools import find_packages, setup

setup(
    name="llkms",
    version="0.1.0",
    description="A Language Learning Knowledge Management System",
    author="Milosz",
    url="https://github.com/Butterski/llkms",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-openai",
        "python-dotenv",
        "boto3",
        "unstructured",
        "faiss-cpu",
        "pillow",
        "pypdf2",
        "python-docx",
        "bs4",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "llkms=llkms.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
