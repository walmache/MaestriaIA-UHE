class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

    def saludar(self):
        print(f"Hola, mi nombre es {self.nombre} y tengo {self.edad} años.")

    def es_mayor_de_edad(self):
        return self.edad >= 18

# Crear instancias de la clase Persona
persona1 = Persona("Juan", 28)
persona2 = Persona("Ana", 16)

# Usar métodos de la clase Persona
persona1.saludar()
persona2.saludar()

# Verificar si las personas son mayores de edad
if persona1.es_mayor_de_edad():
    print(f"{persona1.nombre} es mayor de edad.")
else:
    print(f"{persona1.nombre} no es mayor de edad.")

if persona2.es_mayor_de_edad():
    print(f"{persona2.nombre} es mayor de edad.")
else:
    print(f"{persona2.nombre} no es mayor de edad.")
    
    asdasd
