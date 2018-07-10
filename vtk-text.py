# -*- coding: utf-8 -*-
"""
This script is used to test the functionality of the meshpy module.
"""

import os

# Meshpy imports.
from meshpy import *
from meshpy import BaseMeshItem

import numpy as np
import autograd.numpy as npAD

def test_curve():
    """Create a helix from a parametric curve."""
    
    input_file = InputFile(maintainer='Ivo Steinbrecher')
    
    
    input_file.add('''
--------------------------------------------------------------------PROBLEM SIZE
DIM                                   3
---------------------------------------------------------------------PROBLEM TYP
PROBLEMTYP                            Structure
RESTART                               0
------------------------------------------------------------------DISCRETISATION
NUMFLUIDDIS                           0
NUMSTRUCDIS                           1
NUMALEDIS                             0
NUMTHERMDIS                           0
-----------------------------------------------------------IO/RUNTIME VTK OUTPUT
OUTPUT_DATA_FORMAT                    ascii
INTERVAL_STEPS                        1
EVERY_ITERATION                       No
-----------------------------------------------------IO/RUNTIME VTK OUTPUT/BEAMS
OUTPUT_BEAMS                          Yes
DISPLACEMENT                          Yes
USE_ABSOLUTE_POSITIONS                Yes
TRIAD_VISUALIZATIONPOINT              Yes
STRAINS_GAUSSPOINT                    Yes
MATERIAL_FORCES_GAUSSPOINT            Yes
--------------------------------------------------------------STRUCTURAL DYNAMIC
INT_STRATEGY                          Standard
TIMESTEP                              1
NUMSTEP                               2
MAXTIME                               2.0
LINEAR_SOLVER                         1
DYNAMICTYP                            Statics
RESULTSEVRY                           1
NLNSOL                                fullnewton
TOLRES                                1.0E-7
TOLDISP                               1.0E-10
NORM_RESF                             Abs
NORM_DISP                             Abs
NORMCOMBI_RESFDISP                    And
MAXITER                               15
PREDICT                               TangDis
------------------------------------------------------------------------SOLVER 1
NAME                                  Structure_Solver
SOLVER                                UMFPACK
''')
    
    # Add material and functions.
    mat = Material('MAT_BeamReissnerElastHyper', 2.07e2, 0.3, 1e-3, 0.5,
        shear_correction=0.75
        )
    
    # Set parameters for the helix.
    R = 4.
    tz = 1. # incline
    n = 2 # number of turns
    n_el = 5
    
    # Create a helix with a parametric curve.
    def helix(t):
        #return np.array([R*np.cos(t), R*np.sin(t), t*tz/(2*np.pi)])
        return npAD.array([npAD.cos(t),npAD.sin(t),t])
    
    print(1)
    helix_set = input_file.create_beam_mesh_line(Beam3rHerm2Lin3, mat,
        [1,0,0], [1., 10, 2.*np.pi*n], n_el=n_el)
    print(2)
    input_file.wrap_around_cylinder()
    print(3)
    
    # Apply boundary conditions.
    input_file.add(BoundaryCondition(helix_set['start'], 'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
        bc_type=mpy.dirichlet))
    input_file.add(BoundaryCondition(helix_set['end'], 'NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0',
        bc_type=mpy.neumann))
    
    
    #input_file.preview_python()
    input_file.write_input_file('/home/ivo/temp/beam.dat')
    
    # Check the output.
    #ref_file = os.path.join(testing_input, 'curve_3d_helix_ref.dat')
    #string2 = input_file.get_string(header=False).strip()
    #with open(ref_file, 'r') as r_file:
    #    string1 = r_file.read().strip()
    #self.compare_strings('test_curve_3d_helix', string1, string2)


test_curve()




























def indent(elem, level=0):
  i = "\n" + level*"  "
  if len(elem):
    if not elem.text or not elem.text.strip():
      elem.text = i + "  "
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
    for elem in elem:
      indent(elem, level+1)
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
  else:
    if level and (not elem.tail or not elem.tail.strip()):
      elem.tail = i







import xml.etree.cElementTree as ET

# build a tree structure
root = ET.Element("html")

head = ET.SubElement(root, "head")

title = ET.SubElement(head, "title")
title.text = "Page Title"

body = ET.SubElement(root, "body")
body.set("bgcolor", "#ffffff")

body.text = "Hello, World!"


indent(root)

# wrap it in an ElementTree instance, and save as XML
tree = ET.ElementTree(root)




tree.write("/home/ivo/temp/page.xhtml",  xml_declaration=True, encoding='utf-8', method="xml")












def print_arg(obj):
        
    list = (
        [method_name for method_name in dir(obj)
     if callable(getattr(obj, method_name))]
        )
     
    for item in list:
        print(item)
      










# 
# 
import vtk
# from vtk import *
#  
# #setup points and vertices
# Points = vtk.vtkPoints()
# Triangles = vtk.vtkCellArray()
#  
# Points.InsertNextPoint(1.0, 0.0, 0.0)
# Points.InsertNextPoint(0.0, 0.0, 0.0)
# Points.InsertNextPoint(0.0, 1.0, 0.0)
#  
# Triangle = vtk.vtkTriangle();
# Triangle.GetPointIds().SetId(0, 0);
# Triangle.GetPointIds().SetId(1, 1);
# Triangle.GetPointIds().SetId(2, 2);
# Triangles.InsertNextCell(Triangle);
#  
# #setup colors (setting the name to "Colors" is nice but not necessary)
# Colors = vtk.vtkUnsignedCharArray();
# Colors.SetNumberOfComponents(3);
# Colors.SetName("Colors");
# Colors.InsertNextTuple3(255,50,0);
#  
# polydata = vtk.vtkPolyData()
# polydata.SetPoints(Points)
# polydata.SetPolys(Triangles)
#  
# polydata.GetCellData().SetScalars(Colors);
# polydata.Modified()
# if vtk.VTK_MAJOR_VERSION <= 5:
#     polydata.Update()

# 
# writer = vtk.vtkXMLPolyDataWriter();
# writer.SetFileName("TriangleSolidColor.vtp");
# if vtk.VTK_MAJOR_VERSION <= 5:
#     writer.SetInput(polydata)
# else:
#     writer.SetInputData(polydata)
# writer.Write()




 
points = vtk.vtkPoints()
points.InsertNextPoint(0.2, 0, 0)
points.InsertNextPoint(1, 0, 0)
points.InsertNextPoint(1, 1, 0)
points.InsertNextPoint(0, 1, 1)
points.InsertNextPoint(5, 5, 5)
points.InsertNextPoint(6, 5, 5)
points.InsertNextPoint(6, 6, 5)
points.InsertNextPoint(5, 6, 6)
 
# The first tetrahedron
unstructuredGrid1 = vtk.vtkUnstructuredGrid()
unstructuredGrid1.SetPoints(points)
 
tetra = vtk.vtkTetra()
 
tetra.GetPointIds().SetId(0, 0)
tetra.GetPointIds().SetId(1, 1)
tetra.GetPointIds().SetId(2, 2)
tetra.GetPointIds().SetId(3, 3)
 
cellArray = vtk.vtkCellArray()
cellArray.InsertNextCell(tetra)


tetra2 = vtk.vtkTetra()
 
tetra2.GetPointIds().SetId(0, 4)
tetra2.GetPointIds().SetId(1, 5)
tetra2.GetPointIds().SetId(2, 6)
tetra2.GetPointIds().SetId(3, 7)


unstructuredGrid1.SetCells(vtk.VTK_TETRA, cellArray)

cellArray.InsertNextCell(tetra2)

unstructuredGrid1.SetCells(vtk.VTK_TETRA, cellArray)
 
 















points3 = vtk.vtkPoints()
points3.InsertNextPoint(0.2, 0, 0)
points3.InsertNextPoint(1, 0, 2)
points3.InsertNextPoint(1, 1, 0)

points3.InsertNextPoint(5, 5, 5)
points3.InsertNextPoint(6, 5, 5)
points3.InsertNextPoint(6, 6, 5)
points3.InsertNextPoint(5, 6, 6)

unstructuredGrid3 = vtk.vtkUnstructuredGrid()
unstructuredGrid3.SetPoints(points3)

cellArray3 = vtk.vtkCellArray()

 
poly = vtk.vtkPolyLine()
poly.GetPointIds().SetNumberOfIds(3)
poly.GetPointIds().SetId(0, 0)
poly.GetPointIds().SetId(1, 1)
poly.GetPointIds().SetId(2, 2)
 
cellArray3.InsertNextCell(poly)





poly2 = vtk.vtkPolyLine()
poly2.GetPointIds().SetNumberOfIds(4)
poly2.GetPointIds().SetId(0, 3)
poly2.GetPointIds().SetId(1, 0)
poly2.GetPointIds().SetId(2, 5)
poly2.GetPointIds().SetId(3, 6)

cellArray3.InsertNextCell(poly2)


unstructuredGrid3.SetCells(vtk.VTK_POLY_LINE, cellArray3)


#unstructuredGrid3.InsertNextCell(poly.GetCellType(),poly.GetPointIds())
#unstructuredGrid3.InsertNextCell(poly2.GetCellType(),poly2.GetPointIds())



 
Colors1 = vtk.vtkUnsignedCharArray();
Colors1.SetNumberOfComponents(3);
Colors1.SetName("Colors1");
Colors1.InsertNextTuple3(255,50,0);
Colors1.InsertNextTuple3(255,50,0);

Colors2 = vtk.vtkUnsignedCharArray();
Colors2.SetNumberOfComponents(4);
Colors2.SetName("Colors2");
Colors2.InsertNextTuple4(255,50,0,3);
Colors2.InsertNextTuple4(255,50,0,0.34444444);

Colors3 = vtk.vtkUnsignedCharArray();
Colors3.SetNumberOfComponents(3);
Colors3.SetName("Colors3");
Colors3.InsertNextTuple3(255,50,0);
Colors3.InsertNextTuple3(255,50,0);
Colors3.InsertNextTuple3(255,50,0);
Colors3.InsertNextTuple3(255,50,0);
Colors3.InsertNextTuple3(255,50,0);
Colors3.InsertNextTuple3(255,50,0);
Colors3.InsertNextTuple3(255,50,0);



print_arg(Colors2)


unstructuredGrid3.GetCellData().SetVectors(Colors1)
unstructuredGrid3.GetCellData().SetScalars(Colors2)
#unstructuredGrid3.GetCellData().SetTensors(Colors3)


unstructuredGrid3.GetPointData().SetVectors(Colors3)

















# write cell data to file
writer = vtk.vtkXMLUnstructuredGridWriter();
# wirte human readable data
writer.SetDataModeToAscii()
writer.SetFileName("TriangleSolidColor2.vtu");
writer.SetInputData(unstructuredGrid3)
writer.Write()

 
 
 


 

  
#  
#  
#  
#  
#  
# # Create a mapper and actor
# mapper1 = vtk.vtkDataSetMapper()
# if vtk.VTK_MAJOR_VERSION <= 5:
#     mapper1.SetInputConnection(unstructuredGrid1.GetProducerPort())
# else:
#     mapper1.SetInputData(unstructuredGrid1)
#  
# actor1 = vtk.vtkActor()
# actor1.SetMapper(mapper1)
#  
# # Create a mapper and actor
# mapper2 = vtk.vtkDataSetMapper()
# if vtk.VTK_MAJOR_VERSION <= 5:
#     mapper2.SetInputConnection(unstructuredGrid2.GetProducerPort())
# else:
#     mapper2.SetInputData(unstructuredGrid2)
#  
# actor2 = vtk.vtkActor()
# actor2.SetMapper(mapper2)
#  
# # Create a renderer, render window, and interactor
# renderer = vtk.vtkRenderer()
# renderWindow = vtk.vtkRenderWindow()
# renderWindow.AddRenderer(renderer)
# renderWindowInteractor = vtk.vtkRenderWindowInteractor()
# renderWindowInteractor.SetRenderWindow(renderWindow)
#  
# # Add the actor to the scene
# renderer.AddActor(actor1)
# renderer.AddActor(actor2)
# renderer.SetBackground(.3, .6, .3) # Background color green
#  
# # Render and interact
# renderWindow.Render()
# renderWindowInteractor.Start()
# 


