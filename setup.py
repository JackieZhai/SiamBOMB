# 
# GitHub: JackieZhai @ MiRA, CASIA, Beijing, China
# See Corresponding LICENSE, All Right Reserved.
# 
# setup pypi package based on
# pysot, pytracking, sots, stack and lighttrack.
# See Corresponding CONTRIBUTING, Please contact if any conflicts.
# 

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SiamBOMB",
    version="0.0.5",
    author="JackieZhai",
    author_email="jackieturing@gmail.com",
    description="SiamBOMB: Siamese network using Background information for real-time Online Multi-species home-cage animal Behavioral analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JackieZhai/SiamBOMB",
    project_urls={
        "Bug Tracker": "https://github.com/JackieZhai/SiamBOMB/issues",
        "Request Tracker": "https://github.com/JackieZhai/SiamBOMB/pulls",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    package_data={
        "": ["*.json", "*.qss", "*.yaml"]
    },
    install_requires=[
        "numpy",
        "opencv-python",
        "pyyaml",
        "yacs",
        "tqdm",
        "colorama",
        "matplotlib",
        "cython",
        "tensorboardX",
        "imutils",
        "pandas",
        "scikit-image",
        "visdom",
        "tb-nightly",
        "tikzplotlib", 
        "gdown", 
        "timm"
    ]
)
