import random
class Robot:  
    def __init__(self, name):
        self.name = name
        self.health_level = random.random() 
        
    def say_hi(self):
        print("Hi, I am " + self.name)

class PhysicianRobot(Robot):
    pass
x = Robot("Marvin")
y = PhysicianRobot("James")
print(x, type(x))
print(y, type(y))
y.say_hi()

class PhysicianRobot(Robot):
	def say_hi(self):
		super().say_hi()
		print("i love physics");

class sahil:
	pass
amin = sahil
print(amin)