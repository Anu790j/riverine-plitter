import setuptools

packages=setuptools.find_packages(),

setuptools.setup(
	name="plitter",
	version="0.1.0",
	author="Anubhav Jain",
	author_email="anubhavj280@gmail.com",
	description="plastic litter detection tool",
	long_description_content_type="text/markdown",
	packages=["plitter"],
	install_requires=[
		"opencv-python",
		"pandas",
		"exif",
		"gpxpy",
		 "torch>=1.7.0",
        "torchvision>=0.8.0",
        "opencv-python",
        "numpy",
        "tqdm",
	],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: ",
		"Operating System :: Linux",
	],
)

