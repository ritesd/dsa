from dataclasses import dataclass, asdict, astuple
import inspect


@dataclass(frozen=True, order=True)
class A:
    a:int
    b:int

    def __iter__(self):
        return {f''}

class NormalA:
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b


a1 = NormalA(5,6)
b1 = NormalA(5,6)

print(a1==b1) # will print False

a2 = A(5,6)
b2 = A(5,6)

print(a2==b2) # will print True , here A has __eq__ funtion 

print(inspect.getmembers(A, inspect.isfunction)) 
"""
[('__delattr__', <function A.__delattr__ at 0x1030c7ce0>), ('__eq__', <function A.__eq__ at 0x1030c4fe0>), ('__ge__', <function A.__ge__ at 0x1030c54e0>), ('__gt__', <function A.__gt__ at 0x1030c40e0>), ('__hash__', <function A.__hash__ at 0x102b8b6a0>), ('__init__', <function A.__init__ at 0x1030c4400>), ('__le__', <function A.__le__ at 0x1030c5a80>), ('__lt__', <function A.__lt__ at 0x1030c5580>), ('__repr__', <function A.__repr__ at 0x1030c5e40>), ('__setattr__', <function A.__setattr__ at 0x1030c4b80>)]
"""