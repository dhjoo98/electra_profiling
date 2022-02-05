from one import Base

class two(Base):
    def __init__(self):
        self.name = two

    def printing(self):
        print('2')
        return 2

testing = two()
a = testing.printing()
print('returned value = ' , a)

'''
2
returned value = 2

즉, 가장 밑 단의 함수를 사용, return 
'''
