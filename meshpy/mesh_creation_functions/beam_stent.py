'''
Created on Oct 17, 2018

@author: dao
'''
# Python packages.
import numpy as np

# Meshpy modules.
from .. import mpy, Rotation, Mesh, GeometryName, GeometrySet
from . import create_beam_mesh_arc_segment, create_beam_mesh_line

def create_stent_cell(mesh, beam_object, material, width, bottom_width, neck_width, height,
    alpha, radius, n_el, is_bottom_cell=False, is_top_cell=False, add_sets=False):
    """
    Create a cell of the stent. This cell is on the x-y plane.

    Args
    ----
    mesh: Mesh
        Mesh that the stent will be added to.
    beam_object: Beam
        Object that will be used to create the beam elements.
    material: Material
        Material for the beam.
    width: float
        Width of the total cell.
    bottom_width: float
        Width of the cell's bottom.
    neck_width: float
        Width of the cell's neck.
    height: float
        Height of the total cell.
    alpha: radiant
        The angle between
    radius: float
        The radius of the two neck curve in S1.
    n_el: int
        Number of elements per beam line.
    is_bottom_cell: bool
        This check weather the cell is on bottom of the stent flat. If it is True S2-curve isn't created,
        S3-curve will be twice created and one of them will be moved down.
    is_top_cell: bool
        This check weather the cell is on top of the stent flat. If it is True S3-curve isn't created,
        S2-curve will be twice created and one of them will be down moved.
    ( these variables are described in a file )
    """
    def add_line(pointa, pointb, n_el_line):
        """ Shortcut to add line."""
        return create_beam_mesh_line(
            mesh,
            beam_object,
            material,
            pointa,
            pointb,
            n_el=n_el_line
            )

    def add_segment(center, axis_rotation, radius, angle, n_el_segment):
        """ Shortcut to add arc segment."""
        return create_beam_mesh_arc_segment(mesh, beam_object, material, center,
                                            axis_rotation, radius, angle, n_el=n_el_segment)

    # Create S1 curve
    neck_point = np.array([-neck_width, height * 0.5, 0])
    d = (height * 0.5 / np.tan(alpha) + bottom_width - neck_width) / np.sin(alpha)
    CM = np.array([-np.sin(alpha), -np.cos(alpha), 0]) * (d - radius)
    MO = np.array([np.cos(alpha), -np.sin(alpha), 0]) * np.sqrt(radius ** 2 - (d - radius) ** 2)
    S1_angle = np.pi / 2 + np.arcsin((d - radius) / radius)
    S1_center1 = CM + MO + neck_point
    S1_axis_rotation1 = Rotation([0, 0, 1], 2 * np.pi - S1_angle - alpha)
    add_segment(S1_center1, S1_axis_rotation1, radius, S1_angle, n_el)
    add_line([-bottom_width, 0, 0], mesh.nodes[-1].coordinates, 2 * n_el)

    S1_center2 = 2 * neck_point - S1_center1
    S1_axis_rotation2 = Rotation([0, 0, 1], np.pi - alpha - S1_angle)

    add_segment(S1_center2, S1_axis_rotation2, radius, S1_angle, n_el)
    add_line(mesh.nodes[-1].coordinates, 2 * neck_point - [-bottom_width, 0, 0], 2 * n_el)
    top_width = - mesh.nodes[-1].coordinates[0]

    if is_bottom_cell:
        S3_radius = (width - bottom_width + height * 0.5 / np.tan(alpha)) * np.tan(alpha / 2)
        S3_center = [-width, height * 0.5, 0] + S3_radius * np.array([0, 1, 0])
        S3_axis_rotation = Rotation()
        S3_angle = np.pi - alpha
        add_segment(S3_center, S3_axis_rotation,
            S3_radius, S3_angle, n_el)
        add_line(mesh.nodes[-1].coordinates, [-bottom_width, height, 0], 2 * n_el)

        # Create S3 curve with translation down
        S3_radius = (width - bottom_width + height * 0.5 / np.tan(alpha)) * np.tan(alpha / 2)
        S3_center = [-width, -height * 0.5, 0] + S3_radius * np.array([0, 1, 0])
        S3_axis_rotation = Rotation()
        S3_angle = np.pi - alpha
        add_segment(S3_center, S3_axis_rotation,
            S3_radius, S3_angle, n_el)
        add_line(mesh.nodes[-1].coordinates, [-bottom_width, 0, 0], 2 * n_el)

    else:
        if is_top_cell:
            S2_radius = (height * 0.5 / np.tan(alpha) + top_width) * np.tan(alpha * 0.5)
            S2_center = [0, height * 0.5, 0] - S2_radius * np.array([0, 1, 0])
            S2_angle = np.pi - alpha
            S2_axis_rotation = Rotation([0, 0, 1], np.pi)
            add_segment(S2_center, S2_axis_rotation,
                S2_radius, S2_angle, 2 * n_el)
            add_line(mesh.nodes[-1].coordinates, [-top_width, 0, 0], 2 * n_el)

            # Create S2 curve with translation up
            S2_radius = (height * 0.5 / np.tan(alpha) + top_width) * np.tan(alpha * 0.5)
            S2_center = [0, height * 1.5, 0] - S2_radius * np.array([0, 1, 0])
            S2_angle = np.pi - alpha
            S2_axis_rotation = Rotation([0, 0, 1], np.pi)
            add_segment(S2_center, S2_axis_rotation,
                S2_radius, S2_angle, 2 * n_el)
            add_line(mesh.nodes[-1].coordinates, [-top_width, height, 0], 2 * n_el)
        else:
            # Create S2 curve
            S2_radius = (height * 0.5 / np.tan(alpha) + top_width) * np.tan(alpha * 0.5)
            S2_center = [0, height * 0.5, 0] - S2_radius * np.array([0, 1, 0])
            S2_angle = np.pi - alpha
            S2_axis_rotation = Rotation([0, 0, 1], np.pi)
            add_segment(S2_center, S2_axis_rotation,
                S2_radius, S2_angle, 2 * n_el)
            add_line(mesh.nodes[-1].coordinates, [-top_width, 0, 0], 2 * n_el)

            # Create S3 curve
            S3_radius = (width - bottom_width + height * 0.5 / np.tan(alpha)) * np.tan(alpha / 2)
            S3_center = [-width, height * 0.5, 0] + S3_radius * np.array([0, 1, 0])
            S3_axis_rotation = Rotation()
            S3_angle = np.pi - alpha
            add_segment(S3_center, S3_axis_rotation,
                S3_radius, S3_angle, n_el)
            add_line(mesh.nodes[-1].coordinates, [-bottom_width, height, 0], 2 * n_el)


def create_stent_cell_column(mesh, beam_object, material, row_number, width,
            bottom_width, neck_width, height, alpha, radius, n_el):
    """ Create a column of completed cells. A completed cell consists of one cell, that is created
    with the create cell function and it's reflection.

    Args
    ----
    mesh: Mesh
        Mesh that the stent will be added to.
    beam_object: Beam
        Object that will be used to create the beam elements.
    material: Material
        Material for the beam.
    row_number: int
        The cell number on the column.
    width: float
        Width of the total cell.
    bottom_width: float
        Width of the cell's bottom.
    neck_width: float
        Width of the cell's neck.
    height: float
        Height of the total cell.
    alpha: radiant
        The angle between
    radius: float
        The radius of the two neck curve in S1.
    n_el: int
        Number of elements per beam line.
    ( these variables are described in a file )
    """
    def create_bottom_cell(bottom):
        """ Create a completed cell in bottom."""
        mesh_bottom = Mesh()
        create_stent_cell(mesh_bottom, beam_object, material, width, bottom_width,
            neck_width, height, alpha, radius, n_el, is_bottom_cell=True)
        mesh_bottom.translate([0, -height, 0])
        bottom.add_mesh(mesh_bottom)

    def create_top_cell(top):
        """ Create a completed cell on top."""
        mesh_top = Mesh()
        create_stent_cell(mesh_top, beam_object, material, width, bottom_width,
                neck_width, height, alpha, radius, n_el, is_top_cell=True)
        mesh_top.translate([0, (row_number - 2) * height, 0])
        top.add_mesh(mesh_top)

    mesh1 = Mesh()
    mesh2 = Mesh()
    for i in range(row_number - 2):
        mesh1.translate([0, height, 0])
        mesh2.translate([0, height, 0])
        create_stent_cell(mesh1, beam_object, material, width, bottom_width,
            neck_width, height, alpha, radius, n_el)
        create_stent_cell(mesh2, beam_object, material, width, bottom_width,
            neck_width, height, alpha, radius, n_el)

    create_bottom_cell(mesh1)
    create_top_cell(mesh1)
    create_bottom_cell(mesh2)
    create_top_cell(mesh2)
    mesh2.reflect([-1, 0, 0], [0, 0, 0])
    mesh.add_mesh(mesh1)
    mesh.add_mesh(mesh2)

def create_beam_mesh_stent(mesh, beam_object, material, row_number, column_number,
                        width, bottom_width, neck_width, height, alpha, radius, n_el=1, add_sets=False):
    """
    Create a stent structure around cylinder, The cylinder axis will be the z-axis.

    Args
    ----
    mesh: Mesh
        Mesh that the stent will be added to.
    beam_object: Beam
        Object that will be used to create the beam elements.
    material: Material
        Material for the beam.
    row_number: int
        The cell number on a column.
    row_number: int
        The cell number on a row.
    width: float
        Width of the total cell.
    bottom_width: float
        Width of the cell's bottom.
    neck_width: float
        Width of the cell's neck.
    height: float
        Height of the total cell.
    alpha: radiant
        The angle between
    radius: float
        The radius of the two neck curve in S1.
    n_el: int
        Number of elements per beam line.
    ( these variables are described in a file )
    add_sets: bool
    If this is true the sets are added to the mesh and then displayed
    n eventual VTK output, even if they are not used for a boundary
    condition or coupling.

    Return
    ----
    return_set: GeometryName
        Set with nodes on the top, bottom boundaries. Those
        sets only contains end nodes of lines, not the middle ones. The set
        'all' contains all nodes.
    """
    i_node_start = len(mesh.nodes)
    for i in range(column_number):
        create_stent_cell_column(mesh, beam_object, material, row_number, width,
                        bottom_width, neck_width, height, alpha, radius, n_el)
        mesh.translate([width * 2, 0, 0])
    for i in range(column_number // 2):
        for j in range(row_number + 1):
            create_beam_mesh_line(mesh, beam_object, material,
                        [4 * (i + 1) * width - width, (j - 1.5) * height, 0],
                        [4 * (i + 1) * width - width, (j - 0.5) * height, 0], n_el=2 * n_el)
        create_beam_mesh_line(mesh, beam_object, material, [(4 * i + 1) * width, -1.5 * height, 0],
                    [(4 * i + 1) * width, -0.5 * height, 0], n_el=2 * n_el)
    mesh.rotate(Rotation([1, 0, 0], np.pi / 2))
    mesh.rotate(Rotation([0, 0, 1], np.pi / 2))
    mesh.translate([column_number * width / np.pi, 0, 0])
    mesh.translate([0, width, 0])
    mesh.wrap_around_cylinder()


    # List of nodes from the stent that are candidates for connections.
    stent_nodes_all = [
        mesh.nodes[i] for i in range(i_node_start, len(mesh.nodes))
        ]
    stent_nodes = [
        node for node in stent_nodes_all if node.is_end_node
        ]

    # Add connections for the nodes with same positions.

    mesh.couple_nodes(nodes=stent_nodes)

    # Get min and max nodes of the honeycomb.
    min_max_nodes = mesh.get_min_max_nodes(nodes=stent_nodes)

    # Return the geometry set.
    return_set = GeometryName()
    return_set['top'] = min_max_nodes['z_max']
    return_set['bottom'] = min_max_nodes['z_min']
    return_set['all'] = GeometrySet(mpy.line, stent_nodes_all)
    if add_sets:
        mesh.add(return_set)
    return return_set


