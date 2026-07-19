from setuptools import find_packages, setup
package_name = 'camera_calib_pkg'
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Camera calibration ROS2 wrapper (publisher + undistort node)',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_publisher = camera_calib_pkg.image_publisher:main',
            'undistort_node = camera_calib_pkg.undistort_node:main',
            'segmentation_bridge_node = camera_calib_pkg.segmentation_bridge_node:main',
        ],
    },
)
