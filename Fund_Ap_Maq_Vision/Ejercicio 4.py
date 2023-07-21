print("Para calcular la distancia focal precione 1", 
      "para la distancia de la imagen al lente 2", 
      "y para la distancia del objeto al lente 3")
s=int(input("engrese la opcion: "))

if s==1:
    u=float(input("ingrese la distancia del objeto al lente: "))
    v=float(input("ingrese la distancia de la imagen al lente: "))

    f=1/((1/u)+(1/v))

    print("la distancia focal es: ", f)

elif s==2:
    f=float(input("ingrese la distancia focal: "))
    v=float(input("ingrese la distancia de la imagen al lente: "))
    u=1/((1/f)-(1/v))

    
    print("la distancia del objeto es: ", u)
elif s==3:

    u=float(input("ingrese la distancia focal: "))
    f=float(input("ingrese la distancia del objeto al lente: "))
    v=1/((1/f)-(1/u))

    print("la distancia de la imagen es: ",v)
else:

    print("elija una opcion valida")

