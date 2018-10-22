'''
Created on Oct 17, 2018

@author: dao
'''
# Python packages.
import numpy as np

# Meshpy modules.
from .. import mpy, Rotation, Mesh
from . import create_beam_mesh_arc_segment, create_beam_mesh_line

def create_stent_cell(mesh, beam_object, material, width, bottom_width, neck_width, height,
    alpha, radius, is_bottom_cell=False, is_top_cell=False, add_sets=False):
    def add_line(pointa, pointb, n_el=1):
        return create_beam_mesh_line(
            mesh,
            beam_object,
            material,
            pointa,
            pointb,
            n_el=n_el
            )

    def add_segment(center, axis_rotation, radius, angle, n_el=1):
        return create_beam_mesh_arc_segment(mesh, beam_object, material, center,
                                            axis_rotation, radius, angle, n_el=n_el)

    # Create S1 curve
    neck_point = np.array([-neck_width, height * 0.5, 0])
    d = (height * 0.5 / np.tan(alpha) + bottom_width - neck_width) / np.sin(alpha)
    CM = np.array([-np.sin(alpha), -np.cos(alpha), 0]) * (d - radius)
    MO = np.array([np.cos(alpha), -np.sin(alpha), 0]) * np.sqrt(radius ** 2 - (d - radius) ** 2)
    S1_angle = np.pi / 2 + np.arcsin((d - radius) / radius)
    S1_center1 = CM + MO + neck_point
    S1_axis_rotation1 = Rotation([0, 0, 1], 2 * np.pi - S1_angle - alpha)
    add_segment(S1_center1, S1_axis_rotation1, radius, S1_angle)
    add_line([-bottom_width, 0, 0], mesh.nodes[-1].coordinates)

    S1_center2 = 2 * neck_point - S1_center1
    S1_axis_rotation2 = Rotation([0, 0, 1], np.pi - alpha - S1_angle)

    add_segment(S1_center2, S1_axis_rotation2, radius, S1_angle)
    add_line(mesh.nodes[-1].coordinates, 2 * neck_point - [-bottom_width, 0, 0])
    top_width = - mesh.nodes[-1].coordinates[0]

    if is_bottom_cell:
        S3_radius = (width - bottom_width + height * 0.5 / np.tan(alpha)) * np.tan(alpha / 2)
        S3_center = [-width, height * 0.5, 0] + S3_radius * np.array([0, 1, 0])
        S3_axis_rotation = Rotation()
        S3_angle = np.pi - alpha
        add_segment(S3_center, S3_axis_rotation,
            S3_radius, S3_angle)
        add_line(mesh.nodes[-1].coordinates, [-bottom_width, height, 0])

        # Create S3 curve with translation down
        S3_radius = (width - bottom_width + height * 0.5 / np.tan(alpha)) * np.tan(alpha / 2)
        S3_center = [-width, -height * 0.5, 0] + S3_radius * np.array([0, 1, 0])
        S3_axis_rotation = Rotation()
        S3_angle = np.pi - alpha
        add_segment(S3_center, S3_axis_rotation,
            S3_radius, S3_angle)
        add_line(mesh.nodes[-1].coordinates, [-bottom_width, 0, 0])

    else:
        if is_top_cell:
            S2_radius = (height * 0.5 / np.tan(alpha) + top_width) * np.tan(alpha * 0.5)
            S2_center = [0, height * 0.5, 0] - S2_radius * np.array([0, 1, 0])
            S2_angle = np.pi - alpha
            S2_axis_rotation = Rotation([0, 0, 1], np.pi)
            add_segment(S2_center, S2_axis_rotation,
                S2_radius, S2_angle)
            add_line(mesh.nodes[-1].coordinates, [-top_width, 0, 0])

            # Create S2 curve with translation up
            S2_radius = (height * 0.5 / np.tan(alpha) + top_width) * np.tan(alpha * 0.5)
            S2_center = [0, height * 1.5, 0] - S2_radius * np.array([0, 1, 0])
            S2_angle = np.pi - alpha
            S2_axis_rotation = Rotation([0, 0, 1], np.pi)
            add_segment(S2_center, S2_axis_rotation,
                S2_radius, S2_angle)
            add_line(mesh.nodes[-1].coordinates, [-top_width, height, 0])
        else:
            # Create S2 curve
            S2_radius = (height * 0.5 / np.tan(alpha) + top_width) * np.tan(alpha * 0.5)
            S2_center = [0, height * 0.5, 0] - S2_radius * np.array([0, 1, 0])
            S2_angle = np.pi - alpha
            S2_axis_rotation = Rotation([0, 0, 1], np.pi)
            add_segment(S2_center, S2_axis_rotation,
                S2_radius, S2_angle)
            add_line(mesh.nodes[-1].coordinates, [-top_width, 0, 0])

            # Create S3 curve
            S3_radius = (width - bottom_width + height * 0.5 / np.tan(alpha)) * np.tan(alpha / 2)
            S3_center = [-width, height * 0.5, 0] + S3_radius * np.array([0, 1, 0])
            S3_axis_rotation = Rotation()
            S3_angle = np.pi - alpha
            add_segment(S3_center, S3_axis_rotation,
                S3_radius, S3_angle)
            add_line(mesh.nodes[-1].coordinates, [-bottom_width, height, 0])


def create_stent_cell_column(mesh, beam_object, material, row_number, width,
            bottom_width, neck_width, height, alpha, radius):
    def create_bottom_cell(bottom):
        mesh_bottom = Mesh()
        create_stent_cell(mesh_bottom, beam_object, material, width, bottom_width,
            neck_width, height, alpha, radius, is_bottom_cell=True)
        mesh_bottom.translate([0, -height, 0])
        bottom.add_mesh(mesh_bottom)

    def create_top_cell(top):
        mesh_top = Mesh()
        create_stent_cell(mesh_top, beam_object, material, width, bottom_width,
                neck_width, height, alpha, radius, is_top_cell=True)
        mesh_top.translate([0, (row_number - 2) * height, 0])
        top.add_mesh(mesh_top)

    mesh1 = Mesh()
    mesh2 = Mesh()
    for i in range(row_number - 2):
        mesh1.translate([0, height, 0])
        mesh2.translate([0, height, 0])
        create_stent_cell(mesh1, beam_object, material, width, bottom_width,
            neck_width, height, alpha, radius)
        create_stent_cell(mesh2, beam_object, material, width, bottom_width,
            neck_width, height, alpha, radius)

    create_bottom_cell(mesh1)
    create_top_cell(mesh1)
    create_bottom_cell(mesh2)
    create_top_cell(mesh2)
    mesh2.reflect([-1, 0, 0], [0, 0, 0])
    mesh.add_mesh(mesh1)
    mesh.add_mesh(mesh2)

def create_beam_mesh_stent(mesh, beam_object, material, row_number, column_number,
                        width, bottom_width, neck_width, height, alpha, radius):
    for i in range(column_number):
        create_stent_cell_column(mesh, beam_object, material, row_number, width,
                        bottom_width, neck_width, height, alpha, radius)
        mesh.translate([width * 2, 0, 0])
    for i in range(column_number // 2):
        create_beam_mesh_line(mesh, beam_object, material,
                    [4 * (i + 1) * width - width, -1.5 * height, 0], [4 * (i + 1) * width - width, 9.5 * height, 0])
        create_beam_mesh_line(mesh, beam_object, material, [(4 * i + 1) * width, -1.5 * height, 0],
                    [(4 * i + 1) * width, -0.5 * height, 0])
    mesh.rotate(Rotation([1, 0, 0], np.pi / 2))
    mesh.rotate(Rotation([0, 0, 1], np.pi / 2))
    mesh.translate([column_number * width / np.pi, 0, 0])
    mesh.wrap_around_cylinder()

