from setuptools import setup, find_packages

setup(
    name="bmtrain_mindspore",
    version="0.1.0",
    author="Luo Yuqi",
    author_email="luo-yq23@mails.tsinghua.edu.cn",
    description="BMTrain for MindSpore",
    long_description=open("README.md").read(),  # 长描述，通常从 README.md 读取
    long_description_content_type="text/markdown",  # 长描述的内容类型
    url="https://github.com/demerzel-iv/bmtrain_mindspore",
    packages=find_packages(),
    install_requires=[
        "mindspore>=2.5.0",
        "mindnlp",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",  # Python 版本要求
)