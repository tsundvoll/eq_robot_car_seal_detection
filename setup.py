from setuptools import find_packages, setup

setup(
    name="car_seal",
    description="Package to find detect car seals from images using custom vision",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version="1.0.5",
    author="Equinor ASA",
    author_email="fg_robots@equinor.com",
    url="https://github.com/equinor/eq_robot_car_seal_detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)
