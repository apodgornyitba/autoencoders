# Trabajo Práctico 5 - Implementaciones de Autoencoders

## Dependencias

- Python **>= 3.11**
- Pipenv

## Set up

Primero se deben descargar las dependencias a usar en el programa. Para ello podemos hacer uso de los archivos _Pipfile_ y _Pipfile.lock_ provistos, que ya las tienen detalladas. Para usarlos se debe correr en la carpeta del TP4:

```bash
$> pipenv shell
$> pipenv install
```

Esto creará un nuevo entorno virtual, en el que se instalarán las dependencias a usar, que luego se borrarán una vez se cierre el entorno.

**NOTA:** Previo a la instalación se debe tener descargado **python** y **pipenv**, pero se omite dicho paso en esta instalación.

## Cómo Correr \TODO

```bash
python ej1.py
python ej1b.py
python ej2.py
```


## Archivo de Configuración:

### Configuraciones Basicas


```json5
{
    "multilayer": {
        "number_of_inputs": 35,
        "hidden_layers": [15,2,15],
        "number_of_outputs": 35,
        "epochs" : 40000,
        "learning_rate" : 0.005,
        "beta": 0.001,
        "momentum": 0.8
    }
}
```

El campo de *hidden_layers* define la estrutura de la red, donde cada elemento del array representa una capa oculta, y el valor de cada elemento representa la cantidad de neuronas en dicha capa.

### Archivos de salida
Los archivos de salida son imagenes png generados por matplotlib y se muestran al finalizar la ejecución del programa.