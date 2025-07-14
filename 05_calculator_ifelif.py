# making a calculator app
while True:
    print("-" * 30)
    print ("This is simple Calculator app")
    print("This app is created by Govardhan raj")
    print("To exit from calculator type: exit ")
    
    try:    
        num1 = input("Enter first number: ")
        if num1.lower == 'exit':
            break
        elif  num1 == "":
            print("Please input a number or type 'exit'")
            continue
        num1 = float(num1)

        num2 = input("Enter second number: ")
        if num2.lower == 'exit':
            break
        elif  num2 == "":
            print("Please input a number or type 'exit'")
            continue
        num2 = float(num2)
        
        op = input("Choose operation (+, -, *, /): ")
        if op.lower == 'exit':
            break
        elif  op == "":
            print("Choose operation (+, -, *, /): ")
            continue

        if op == "+":
            result = num1 + num2
        elif op == "-":
            result = num1-num2
        
        elif op == "*":
            result = num1*num2
        
        elif op == "/" :
            if num2 != 0 :
                result = num1/num2
            else :
                print ("❌ Cannot divide by zero")
                continue
        else :
            print ("❌ Invalid operator. Please choose +, -, *, or /.")
            continue
        print(f"✅ Result of {num1} {op} {num2} is: {result}")   


    except Exception as e:
        print(f"❌ Error: {e}")
