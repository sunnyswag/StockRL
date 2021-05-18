from setuptools import setup, find_packages

try:
    REQUIRES = list()
    f = open("requirements.txt", "rb")
    for line in f.read().decode("utf-8").split("\n"):
        line = line.strip()
        if "#" not in line and line != '':
            REQUIRES.append(line)
except:
    print("未找到 'requirements.txt'!")
    REQUIRES = list()

setup(
    name="RL_in_Finance",
    version="0.1.0",
    include_package_data=True,
    author="Huiqing Huang",
    author_email="hhq126152@gmail.com",
    url="https://github.com/sunnyswag/RL_in_Finance",
    license="GNU",
    packages=find_packages(),
    install_requires=REQUIRES,
    description="使用DRL，交易股票",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Reinforcment Learning",
    platform=["any"],
    python_requires=">=3.7",
)