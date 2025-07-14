# making a calculator app
while True:
    print ("This is simple Calculator app")
    print("This app is created by Govardhan raj")
    print("To exit from calculator type: exit ")
    
    try:    
        num1 = input("Enter first number: ")
        if num1 == 'exit':
            break
        elif  num1 == "":
            print("Please input number or Exit")
            continue
        num1 = float(num1)

        num2 = input("Enter second number: ")
        if num2 == 'exit':
            break
        elif  num2 == "":
            print("Please input number or Exit")
            continue
        num2 = float(num2)
        
        op = input("Choose operation (+, -, *, /): ")
        if op == 'exit':
            break
        elif  num1 == "":
            print("Please input number or Exit")
            continue
        elif op == "+":
            result = num1 + num2
        elif op == "-":
            result = num1-num2
        
        elif op == "*":
            result = num1*num2
        
        elif op == "/" :
            if num2 != 0 :
                result = num1/num2
            else :
                print ("Python cannot substract by zero")
                continue
        else :
            print ("Invalid operator")
        print (f"result of {num1} {op} {num2} is = {result}")   


    except Exception as e:
        print(e)