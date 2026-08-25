from setuptools import find_packages, setup

package_name = "vln_ai_module_dino"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/vln_ai_module_dino.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Keerthi Shetty",
    maintainer_email="keerthi@example.com",
    description="CMU VLN Challenge 2026 AI module (Grounding DINO perception backend)",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vln_ai_module_dino_node = vln_ai_module_dino.main_node:main",
        ],
    },
)
