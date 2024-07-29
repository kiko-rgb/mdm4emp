import copy
import importlib

import numpy as np

from kiss_icp.tools.visualizer import (
    RegistrationVisualizer,
    StubVisualizer,
    SPHERE_SIZE,
    BLUE,
    GRAY,
    YELLOW,
    BLACK,
    RED,
)


# TODO: override the RegistrationVisualizer methods if necessary. Sadly the interface of the OffscreenRenderer is different to the VisualizerWithKeyCallback. Probably you can't register key callbacks with the OffscreenRenderer. I put in a class that uses the OffscreenRenderer to visualize a mesh. It uses the render to image function to return numpy arrays of the rendered images. You can use this as a reference to implement the AFMRegistrationVisualizer. However, things such as the camera update fn needs to be adapted to your needs.
class AFMRegistrationVisualizer(StubVisualizer):
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError as err:
            print(f'open3d is not installed on your system, run "pip install open3d"')
            exit(1)

        # Initialize GUI controls
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True

        # Create data
        self.source = self.o3d.geometry.PointCloud()
        self.keypoints = self.o3d.geometry.PointCloud()
        self.target = self.o3d.geometry.PointCloud()
        self.frames = []

        # Visualization options
        self.render_map = True
        self.render_source = True
        self.render_keypoints = False
        self.global_view = False
        self.render_trajectory = True
        # Cache the state of the visualizer
        self.state = (
            self.render_map,
            self.render_keypoints,
            self.render_source,
        )

        self.vis = self.o3d.visualization.rendering.OffscreenRenderer(1920, 1080)

    def _initialize_visualizer(self):
        self.vis.scene.add_geometry("source", self.source)
        self.vis.scene.add_geometry("keypoints", self.keypoints)
        self.vis.scene.add_geometry("target", self.target)

    def _register_key_callbacks(self):
        pass

    # def camera_update_fn(self, upward, lookat, viewpoint):
    #     """Update the camera position and orientation.
    #     Args:
    #         camera: The camera object to update.
    #         upward: The upward axis of the camera.
    #         lookat: The point the camera is looking at.
    #         viewpoint: The position of the camera.
    #     """

    #     self.vis.scene.camera.look_at(lookat, viewpoint, upward)

    def _update_geometries(self, source, keypoints, target, pose):
        self.vis.scene.remove_geometry("source")
        self.vis.scene.remove_geometry("keypoints")
        self.vis.scene.remove_geometry("target")

        # Source hot frame
        if self.render_source:
            self.source.points = self.o3d.utility.Vector3dVector(source)
            self.source.paint_uniform_color(YELLOW)
            if self.global_view:
                self.source.transform(pose)
        else:
            self.source.points = self.o3d.utility.Vector3dVector()

        # Keypoints
        if self.render_keypoints:
            self.keypoints.points = self.o3d.utility.Vector3dVector(keypoints)
            self.keypoints.paint_uniform_color(YELLOW)
            if self.global_view:
                self.keypoints.transform(pose)
        else:
            self.keypoints.points = self.o3d.utility.Vector3dVector()

        # Target Map
        if self.render_map:
            target = copy.deepcopy(target)
            self.target.points = self.o3d.utility.Vector3dVector(target)
            if self.global_view:
                self.target.paint_uniform_color(GRAY)
            else:
                self.target.transform(np.linalg.inv(pose))
        else:
            self.target.points = self.o3d.utility.Vector3dVector()

        # Update always a list with all the trajectories
        new_frame = self.o3d.geometry.TriangleMesh.create_sphere(SPHERE_SIZE)
        new_frame.paint_uniform_color(BLUE)
        new_frame.compute_vertex_normals()
        new_frame.transform(pose)
        self.frames.append(new_frame)
        # Render trajectory, only if it make sense (global view)
        if self.render_trajectory and self.global_view:
            self.vis.scene.add_geometry(self.frames[-1], reset_bounding_box=False)

        self.vis.scene.add_geometry("source", self.source)
        self.vis.scene.add_geometry("keypoints", self.keypoints)
        self.vis.scene.add_geometry("target", self.target)
        # self.vis.update_geometry(self.keypoints)
        # self.vis.update_geometry(self.source)
        # self.vis.update_geometry(self.target)
        # if self.reset_bounding_box:
        #     self.vis.reset_view_point(True)
        #     self.reset_bounding_box = False

        # set the camera perspective
        lookat = pose[:3, 3]
        viewpoint = lookat - np.array([0, 0, 1])
        upward = np.array([0, 1, 0])
        self.vis.scene.camera.look_at(lookat, viewpoint, upward)

        # numpy image!!! do with it what you want
        np_img = np.asarray(self.vis.render_to_image())


# class MeshRenderer:
#     def __init__(self, options: dict[str, Any]) -> None:
#         # Set up matplotlib figure optios&parameters
#         self.single_axes = options.get("single_axes", True)
#         self.window_size = options.get("window_size", [480, 540])
#         self.figure_size = options.get("figure_size", [2, 2])
#         self.left_top = options.get("left_top", [0, 0])
#         self.radius, self.viewpoint = options.get(
#             "trajectory", np.array([[0, 0]]).repeat(2, axis=0)
#         )
#         self.background_color = options.get(
#             "background_color", [255.0, 255.0, 255.0, 1.0]
#         )
#         self.index = 0
#         self.vis = None

#     def _offscreen_render(
#         self,
#         mesh,
#         timestamp,
#         camera_update_fn: Callable[[int, o3d.visualization.Visualizer], None],
#         **kwargs,
#     ):
#         window_size = kwargs.get("window_size", self.window_size)

#         vis = o3d.visualization.rendering.OffscreenRenderer(
#             window_size[0], window_size[1]
#         )
#         vis.scene.camera.set_projection(
#             o3d.visualization.rendering.Camera.Projection.Ortho, -1, 1, -1, 1, 0, 10
#         )
#         vis.scene.set_background(
#             np.array(kwargs.get("background_color", self.background_color))
#         )

#         camera_update_fn(timestamp, vis.scene.camera, **kwargs)

#         material = o3d.visualization.rendering.MaterialRecord()

#         geometry = copy.deepcopy(mesh)
#         geometry.compute_vertex_normals()
#         vis.scene.add_geometry("skeleton", geometry, material)
#         if kwargs.get("axis", False):
#             vis.scene.add_geometry("axis", custom_axis(0.5), material)

#         return np.asarray(vis.render_to_image())

#     def render(
#         self,
#         timed_meshes: list[int, o3d.geometry.TriangleMesh],
#         camera_update_fn: Callable[[int, o3d.visualization.Visualizer], None],
#         **kwargs,
#     ):
#         sequence_length = len(timed_meshes)

#         imgs = []
#         for idx in range(sequence_length):
#             imgs.append(
#                 self._offscreen_render(
#                     timed_meshes[idx][1],
#                     timed_meshes[idx][0],
#                     camera_update_fn,
#                     **kwargs,
#                 )
#             )

#         return imgs
