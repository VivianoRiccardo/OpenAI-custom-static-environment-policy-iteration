import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gym_staticinvader",
    version="0.0.2",
    install_requires=['gym', 'numpy'],
    author="Riccardo Viviano",
    author_email="riciric83@gmail.com",
    description="A small example package",
    long_description="description",
    long_description_content_type="text/markdown",
    url="https://github.com/VivianoRiccardo/gym_environment_test",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5.2',
)
