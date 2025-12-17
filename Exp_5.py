# To write a Python script containing four functions Addition, Subtraction, Multiply and Divide functions.

def add(a,b):
    return a + b

def subtract(a,b):
    return a - b

def multiply(a,b):
    return a * b

def divide(a,b):
    if b != 0:
       return a/b 
    else:
        return "Error! Division by zero."
    
if __name__ == "__main__":
    x = float(input("Enter first number: "))
    y = float(input("Enter second number: "))

    print("Addition:", add(x,y))
    print("Subtraction:",subtract(x,y)) 
    print("Multiplication:",multiply(x,y))
    print("Division:",divide(x,y))

