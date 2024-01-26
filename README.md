Repositorio con los codigos para calcular ejecuciones de VQE para moleculas y fermi-Hubbard. Para ejercutar estos archivos, es necesario tener, de forma local o descargar la libreria VqePy y luego importarla al codigo.


# Ejecutar codigo
Para ejecutar los codigos, se tiene que estar en el directiorio base del proyecto.

`
python3 <archivo> <learning rate>
`

El archivo es uno de los .py dentro de las carpetas de los optimizadores (distinto al base.py). El otro parametro es el learning rate que usara el optimizadore en la convergencia.

# Estructuras de los archivos

Existen 2 carpetas de archivos, la primera es FH (modelo de Fermi-Hubbard) y la segunda es Mol (estructuras moleculares).

En ambas carpetas se pueden encontrar los mismos tipos de archivos, cada carpeta representa un optimizadores distinto. En cada una de estas carpetas, se encuentran diferentes codigos, el archivo central es base.py, en este se definen diferentes funciones, el primer tipo de funciones son las que definen los parametros (varios de los parametros estan hardcodeados, asi que cualquier cambio se tiene que hacer a mano, o en el script que se usan en esas funciones) y luego estan las que definen los flujos de trabajo, que reciben los diccionarios de parametros y ejecutan el VQE con las condiciones definidas ahi. Las otras carpetas, como se dijo, representan optimizadores distintos, por lo tanto, hay valores hardcodeados distintos.

Los otros archivos representan diferentes combinaciones de sistemas y ansatz, en las cuales se calculan cosas diferentes y por lo tanto se trabajan con parametros distintos del hamiltoniano.


# Editar archivos
La idea de estos archivos es que sirvan como guia para la creacion de nuevos scripts que permitan calcular el estado de minima energia para diferentes combinaciones de hamiltonianos y ansatz. Para esto se tiene que modificar el archivo de base.py, para ajustar al nuevo sistema, ansatz y sus parametros respectivamente.