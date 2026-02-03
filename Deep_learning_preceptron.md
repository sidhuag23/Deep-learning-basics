```python
# OR 

def mp_neuron(x1,x2):
    #weight 
    w1 = 1
    w2 = 1

    #threshold 
    theta = 1

    #weighted sum 
    net = w1*x1+w2*x2

    if net >= theta:
        return 1
    else:
        return 0

inputs=[(0,0),(0,1),(1,0),(1,1)]

print("x1 x2 | output")

for x1 , x2 in inputs:
    y = mp_neuron(x1,x2)
    print(x1,x2,"|",y)
```

    x1 x2 | output
    0 0 | 0
    0 1 | 1
    1 0 | 1
    1 1 | 1
    


```python
# AND 
def mp_neuron(x1,x2):
    #weight 
    w1 = 1
    w2 = 1

    #threshold 
    theta = 1

    #weighted sum 
    net = w1*x1+w2*x2

    if net > theta:
        return 1
    else:
        return 0

inputs=[(0,0),(0,1),(1,0),(1,1)]

print("x1 x2 | output")

for x1 , x2 in inputs:
    y = mp_neuron(x1,x2)
    print(x1,x2,"|",y)
```

    x1 x2 | output
    0 0 | 0
    0 1 | 0
    1 0 | 0
    1 1 | 1
    


```python
# NOT 
def mp_neuron(x1):
    #weight 
    w1 = 1

    #threshold 
    theta = 1

    #weighted sum 
    net = w1*x1

    if net > theta:
        return 1
    else:
        return 0
        

inputs=[(0),(1),(0),(1)]

print("x1 | output")

for x1  in inputs:
    y = mp_neuron(x1)
    print(x1,"|",y)
```

    x1 | output
    0 | 0
    1 | 0
    0 | 0
    1 | 0
    


```python
#### perceptron implementations 
## AND|
def perceptron(x1,x2):
    w1 = 1
    w2 = 1
    b = -1.5
    #weight and bias 
    z = w1*x1+w2*x2+b

    return 1 if z >= 0 else 0

inputs=[(0,0),(0,1),(1,0),(1,1)]

print("x1 x2 | output")

for x1 , x2 in inputs:
    y = perceptron(x1,x2)
    print(x1,x2,"|",y)


    
```

    x1 x2 | output
    0 0 | 0
    0 1 | 0
    1 0 | 0
    1 1 | 1
    


```python
#### perceptron implementations 

#OR 
def perceptron(x1,x2):
    w1 = 1
    w2 = 1
    b = -0.5
    #weight and bias 
    z = w1*x1+w2*x2+b

    return 1 if z >= 0 else 0

inputs=[(0,0),(0,1),(1,0),(1,1)]

print("x1 x2 | output")

for x1 , x2 in inputs:
    y = perceptron(x1,x2)
    print(x1,x2,"|",y)

```

    x1 x2 | output
    0 0 | 0
    0 1 | 1
    1 0 | 1
    1 1 | 1
    


```python
#### perceptron implementations 
x1=None
x2=None
#OR 
## x1 = hours 
## x2 = attendence
def perceptron_pass(x1,x2):
    w1 = 0.6
    w2 = 0.4
    b = -3
    #weight and bias 
    z = w1*x1+w2*x2+b

    return 1 if z >= 0 else 0

inputs=[(2,1),(4,3),(6,5),(8,6)]

print("hours Attendence| Result")

for x1 , x2 in inputs:
    y = perceptron_pass(x1,x2)
    print(x1,x2,"|","pass" if y==1 else "Fail")

```

    hours Attendence| Result
    2 1 | Fail
    4 3 | pass
    6 5 | pass
    8 6 | pass
    


```python
def step(z):
    return 1 if z >= 0 else 1


def hidden_layer_1(x1, x2):
    w1, w2 = -1, -1
    b = 0.5
    return step(w1 * x1 + w2 * x2 + b)


def hidden_layer_2(x1, x2):
    w1, w2 = -1, -1
    b = 0.5
    return step(w1 * x1 + w2 * x2 + b)


def hidden_layer_3(x1, x2):
    w1, w2 = -1, -1
    b = 0.5
    return step(w1 * x1 + w2 * x2 + b)


def hidden_layer_4(x1, x2):
    w1, w2 = -1, -1
    b = 0.5
    return step(w1 * x1 + w2 * x2 + b)


def output_neuron(h1, h2, h3, h4):
    w1, w2, w3, w4 = -1, 1, 1, -1
    b = -0.5
    z = w1 * h1 + w2 * h2 + w3 * h3 + w4 * h4 + b
    return step(z)


def xor_mlp(x1, x2):
    o1 = hidden_layer_1(x1, x2)
    o2 = hidden_layer_2(x1, x2)
    o3 = hidden_layer_3(x1, x2)
    o4 = hidden_layer_4(x1, x2)

    return output_neuron(o1, o2, o3, o4)


if __name__ == "__main__":
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    print("x1 x2 | output")
    for x1, x2 in inputs:
        print(x1, x2, "|", xor_mlp(x1, x2))

```

    x1 x2 | output
    0 0 | 1
    0 1 | 1
    1 0 | 1
    1 1 | 1
    


```python

```
