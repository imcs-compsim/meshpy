---
layout: page
#title: About MeshPy
permalink: index.html
---


# MeshPy: A general purpose 3D beam finite element input generator

MeshPy is a general purpose 3D beam finite element input generator written in python3.
It contains basic geometry creation and manipulation functions to create complex beam geometries, including rotational degrees of freedom for the beam nodes.
It is currently used in combination with the academic finite element solver [**BACI**](https://baci.pages.gitlab.lrz.de/website), but can easily be adapted to create input files for other solvers as well.
MeshPy is developed at the [Institute for Mathematics and Computer-Based Simulation (IMCS)](https://www.unibw.de/imcs-en) at the Universität der Bundeswehr München.


## Download MeshPy

A clone of the current MeshPy `master` branch is hosted on [GitHub](https://github.com/imcs-compsim/meshpy).


## How to cite MeshPy?

Whenever you use or mention MeshPy in some sort of scientific document/publication/presentation, please cite MeshPy as follows:

```
Steinbrecher, I., Popp, A.: MeshPy -- A general purpose 3D beam finite element input generator, https://compsim.gitlab.io/codes/meshpy
```

With the following BiBTeX entry:
{% raw %}
```TeX
@Misc{MeshPyWebsite,
  author       = {Steinbrecher, I. and Popp, A.},
  howpublished = {\url{https://compsim.gitlab.io/codes/meshpy}},
  title        = {{M}esh{P}y -- {A} general purpose {3D} beam finite element input generator},
  year         = {2021},
  key          = {MeshPyWebsite},
  url          = {https://compsim.gitlab.io/codes/meshpy},
}
```
{% endraw %}


## Contributors

**Main developer**

Ivo Steinbrecher

**Contributors** (in alphabetical order)

Dao Viet Anh

Nora Hagmeyer

Matthias Mayr


## Publications

Journal publications in which MeshPy has been used:

### 2021
- Steinbrecher, I., Popp, A., Meier, C.: Consistent coupling of positions and rotations for embedding 1D Cosserat beams into 3D solid volumes. Comput Mech (2021). doi: [10.1007/s00466-021-02111-4](https://doi.org/10.1007/s00466-021-02111-4)
- Hagmeyer, N., Mayr, M., Steinbrecher, I., Popp, A.: Fluid-beam interaction: Capturing the effect of embedded slender bodies on global fluid flow and vice versa (2021), submitted, [arXiv:2104.09844](https://arxiv.org/abs/2104.09844)

### 2020
- Steinbrecher, I., Mayr, M., Grill, M. J., Kremheller, J., Meier, C., Popp, A.: A mortar-type finite element approach for embedding 1D beams into 3D solid volumes, Comput Mech 66 (6) (2020) 1377–1398. doi: [10.1007/s00466-020-01907-0](https://doi.org/10.1007/s00466-020-01907-0)


## Examples created using MeshPy

Fiber reinforced composite plate:

<img src="figures/composite_plate.png" alt="drawing" width="400"/>

Fiber reinforced pipe under pressure:

<img src="figures/pressure_pipe.png" alt="drawing" width="400"/>

Fiber reinforcements of a twisted plate:

<img src="figures/twisted_plate.png" alt="drawing" width="400"/>
