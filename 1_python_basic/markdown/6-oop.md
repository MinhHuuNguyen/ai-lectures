---
time: 03/04/2022
title: Lập trình hướng đối tượng trong Python
description: Lập trình hướng đối tượng (OOP) là một mô hình lập trình tổ chức thiết kế phổ biến được sử dụng trong nhiều lĩnh vực khác nhau như phát triển phần mềm, lập trình game, lập trình web, quản lý dữ liệu, và trí tuệ nhân tạo. Cho đến nay, đã có nhiều ngôn ngữ lập trình hỗ trợ OOP, trong đó Python là một trong những ngôn ngữ phổ biến và dễ học nhất.
banner_url: https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/1-introduction/python_logo.jpeg
tags: [python]
is_highlight: false
is_published: true
---

## 1. Lý thuyết chung về lập trình hướng đối tượng (OOP)

Lập trình hướng đối tượng (Object-Oriented Programming, OOP) là một mô hình lập trình tổ chức thiết kế phần mềm xung quanh các đối tượng (objects), thay vì chỉ tập trung vào hàm hay logic.
Mỗi đối tượng là một thực thể phần mềm đóng gói thông tin và các hàm xử lý thông tin đó.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/6-oop/banner.jpeg" style="width: 700px;"/>

Ví dụ:
- Trong lập trình game, một đối tượng có thể là một nhân vật (character) với các thuộc tính như tên, sức khỏe, vị trí, và các phương thức như di chuyển, tấn công, né đòn.
- Trong lập trình web, một đối tượng có thể là một nút bấm (button) trên giao diện người dùng với các thuộc tính như màu sắc, kích thước, và các phương thức như nhấn (click), thay đổi màu sắc (change color).
- Trong lập trình quản lý dữ liệu, một đối tượng có thể là một bản ghi (record) trong cơ sở dữ liệu với các thuộc tính như tên, tuổi, địa chỉ, và các phương thức như lưu (save), xóa (delete), cập nhật (update).
- Trong lập trình trí tuệ nhân tạo, một đối tượng có thể là một mô hình học máy (machine learning model) với các thuộc tính như trọng số (weights), cấu trúc mạng (architecture), và các phương thức như huấn luyện (train), dự đoán (predict).


OOP là một phương pháp để mô hình hóa các sự vật cụ thể như ô tô, cũng như các mối quan hệ giữa các sự vật như mối quan hệ giữa công ty và nhân viên, học sinh và giáo viên...
OOP mô hình hóa các thực thể trong thế giới thực thành các đối tượng phần mềm, có một số dữ liệu được liên kết với
chúng và có thể thực hiện các chức năng nhất định.

### 1.1. Các khái niệm cơ bản trong OOP

#### Class (Lớp) và Instance (Thể hiện)

Class (Lớp) đóng vai trò như một khuôn mẫu (blueprint) để tạo ra các đối tượng - “Class defines a set of attributes and methods that the created objects (instances) can have.”

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/6-oop/class_instance.jpeg" style="width: 700px;"/>

Mỗi đối tượng được tạo ra từ một lớp được gọi là một instance (thể hiện) của lớp đó.
Đối tượng được xem là thực thể cụ thể của lớp, có trạng thái và hành vi riêng biệt.

#### Property (Thuộc tính) và Method (Phương thức)

OOP tập trung vào biểu diễn các thực thể thế giới thực (ví dụ: Xe hơi, Người dùng, Animal...) dưới dạng các đối tượng có thuộc tính (data/fields) và phương thức (hàm).
Ví dụ, trong một lớp Dog ta có các thuộc tính như name (tên chó), age (tuổi chó) và phương thức như bark() (hành
động chó sủa).

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/6-oop/property_method.jpeg" style="width: 700px;"/>

Trong lớp, thuộc tính (attribute) là các biến dữ liệu, còn phương thức (method) là hàm được định nghĩa bên trong lớp.

### 1.2. Các tính chất của OOP

OOP mang lại nhiều lợi ích cho phát triển phần mềm: mã trở nên modular hơn, dễ tái sử dụng và mở rộng (scalability) hơn.
Các module (đối tượng) riêng biệt có thể được phát triển, kiểm thử và bảo trì độc lập, giúp tăng năng suất và giảm chi phí phát triển.

Phương pháp OOP cũng hỗ trợ cộng tác nhóm tốt hơn, vì mỗi lập trình viên có thể đảm nhiệm một số lớp (đối tượng) nhất định. 

Các tính chất của OOP như **Tính đóng gói (Encapsulation)** giúp bảo vệ mã nguồn, **Tính kế thừa (Inheritance)** giúp tái sử dụng logic và **Tính đa hình (Polymorphism)** và **Tính trừu tượng (Abstraction)** giúp linh hoạt trong thiết kế, tất cả đều làm cho hệ thống ổn định, dễ bảo trì và phát triển lâu dài.

#### Tính đóng gói (Encapsulation)

Tính đóng gói (Encapsulation) là việc đóng gói dữ liệu và phương thức trong một lớp, bảo vệ trạng thái bên trong của đối tượng và hạn chế truy cập trực tiếp từ bên ngoài.
Nhờ đóng gói, ta chỉ cho phép thao tác lên dữ liệu thông qua các phương thức công khai, giúp tăng tính an toàn và ổn định cho mã nguồn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/6-oop/encapsulation.jpeg" style="width: 700px;"/>

Ví dụ:
Trong một lớp `BankAccount`, ta có thể đóng gói các thuộc tính như `account_number` (số tài khoản), `balance` (số dư) và phương thức như `deposit()` (gửi tiền), `withdraw()` (rút tiền).

Ta chỉ cho phép truy cập và thay đổi thuộc tính `balance` (số dư) thông qua phương thức `deposit()` (gửi tiền) và `withdraw()` (rút tiền), giúp bảo vệ thuộc tính `balance` (số dư) khỏi việc bị thay đổi trực tiếp từ bên ngoài.

Trong các phương thức như `deposit()` (gửi tiền) và `withdraw()` (rút tiền), ta có thể có thêm các kiểm tra để đảm bảo rằng số tiền gửi hoặc rút là hợp lệ (ví dụ: không thể rút nhiều hơn số dư hiện có trong tài khoản).

Từ đó, ta có thể đảm bảo rằng trạng thái của đối tượng `BankAccount` luôn hợp lệ và không bị lỗi do các thao tác không đúng từ bên ngoài.

Trong các ngôn ngữ lập trình khác, ta có thể sử dụng các từ khóa như `private`, `protected` hoặc `public` để kiểm soát mức độ truy cập của các thuộc tính và phương thức trong lớp.
- `private`: Chỉ có thể truy cập từ bên trong lớp, không thể truy cập từ bên ngoài hoặc từ các lớp con kế thừa.
- `protected`: Có thể truy cập từ bên trong lớp và từ các lớp con kế thừa, nhưng không thể truy cập từ bên ngoài.
- `public`: Có thể truy cập từ bất kỳ đâu, bao gồm cả bên ngoài lớp.

#### Tính đa hình (Polymorphism)

Tính đa hình (Polymorphism) là khả năng cho phép cùng một tên phương thức nhưng thực hiện khác nhau tùy theo ngữ cảnh hoặc lớp đối tượng.
Polymorphism cho phép đối xử với đối tượng qua giao diện chung mà không cần quan tâm chi tiết lớp cụ thể.
Điều này giúp tăng tính linh hoạt và khả năng mở rộng của mã nguồn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/6-oop/polymorphism.jpeg" style="width: 700px;"/>

Ví dụ: Cả lớp `Cat` và `Dog` đều có phương thức `speak()`, nhưng `Cat().speak()` có thể trả về `"Meow"` trong khi `Dog().speak()` trả về `"Woof"`.

Trong các ngôn ngữ lập trình khác, ta có thể sử dụng tính đa hình thông qua các khái niệm như:
- `Method Overriding`: Khi một lớp con định nghĩa lại phương thức của lớp cha với cùng tên và tham số, phương thức của lớp con sẽ được gọi thay vì phương thức của lớp cha.
- `Method Overloading`: Khi một lớp có nhiều phương thức cùng tên nhưng khác tham số, phương thức phù hợp sẽ được gọi dựa trên số lượng và kiểu tham số truyền vào.

#### Tính kế thừa (Inheritance)

Tính kế thừa (Inheritance) cho phép một lớp con (subclass) kế thừa thuộc tính và phương thức từ một lớp cha (superclass).
Lớp con có thể tái sử dụng, mở rộng hoặc ghi đè (override) chức năng của lớp cha mà không cần viết lại mã nguồn từ đầu.
Điều này giúp tăng khả năng mở rộng và tái sử dụng mã nguồn.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/6-oop/inheritance.jpeg" style="width: 700px;"/>

Ví dụ: Lớp `Dog` và `Cat` có thể kế thừa từ lớp `Animal` chung, giúp chia sẻ các thuộc tính và phương thức cơ bản như `name` và `speak()`.

Trong các ngôn ngữ lập trình khác, ta có thể sử dụng các từ khóa như `extends` hoặc `implements` để định nghĩa mối quan hệ kế thừa giữa các lớp.
- `extends`: Được sử dụng để chỉ ra rằng một lớp con kế thừa từ một lớp cha.
- `implements`: Được sử dụng để chỉ ra rằng một lớp con thực hiện một giao diện (interface) nào đó.

#### Tính trừu tượng (Abstraction)

Tính trừu tượng (Abstraction) là việc ẩn chi tiết cài đặt bên trong đối tượng, chỉ giữ lại các giao diện cần thiết để sử dụng.
Lớp trừu tượng (abstract class) có thể định nghĩa các phương thức chưa triển khai, buộc các lớp con triển khai chúng.
Nhờ trừu tượng, lập trình viên chỉ cần quan tâm đến cái “làm được gì” của đối tượng, chứ không cần lo “thao tác như thế nào”.

<img src="https://raw.githubusercontent.com/MinhHuuNguyen/ai-lectures/refs/heads/master/1_python_basic/images/6-oop/abstraction.jpeg" style="width: 700px;"/>

Ví dụ: Lớp `Animal` có thể là một lớp trừu tượng với phương thức `speak()` chưa được triển khai, buộc các lớp con như `Dog` và `Cat` phải định nghĩa phương thức này để có thể kế thừa từ `Animal`.

## 2. Lập trình hướng đối tượng trong Python

### 2.1. Định nghĩa Class và tạo Object

#### Class và Instance

Ví dụ về class và instance trong Python:

```python
class Dog:
    pass

dog_lucky = Dog()
dog_bobby = Dog()
```

Ví dụ về class attribute trong Python:

```python
class Dog:
    animal_type = 'Canine'  # Class attribute

dog_lucky = Dog()
print(dog_lucky.animal_type)

# Output: Canine

dog_bobby = Dog()
print(dog_bobby.animal_type)
# Output: Canine
```

#### Constructor __init__() và self.

Ví dụ về constructor __init__() của class trong Python:

```python
class Dog:
    def __init__(self, name, age):
        print("A dog is created.")
    
dog_lucky = Dog("Lucky", 3)
# Output: A dog is created.
```

Ví dụ về self. sử dụng trong constructor __init__() trong Python:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute
        print(f"A dog is created.")
        print(f"Name: {self.name}, Age: {self.age}")

dog_lucky = Dog("Lucky", 3)
# Output: A dog is created.
#         Name: Lucky, Age: 3

dog_bobby = Dog("Bobby", 5)
# Output: A dog is created.
#         Name: Bobby, Age: 5
```

Ví dụ về class attribute và instance attribute trong Python:

```python
class Dog:

    animal_type = 'Canine'  # Class attribute

    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute
        print(f"A dog is created.")
        print(f"Name: {self.name}, Age: {self.age}")

dog_lucky = Dog("Lucky", 3)
# Output: A dog is created.
#         Name: Lucky, Age: 3

print(dog_lucky.name)
print(dog_lucky.age)
print(dog_lucky.animal_type)
# Output: Lucky
#         3
#         Canine

dog_bobby = Dog("Bobby", 5)
# Output: A dog is created.
#         Name: Bobby, Age: 5

print(dog_bobby.name)
print(dog_bobby.age)
print(dog_bobby.animal_type)
# Output: Bobby
#         5
#         Canine
```

### 2.2. Phương thức (Method)

#### Method thông thường

Method thông thường được định nghĩa như một hàm bên trong class và được sử dụng khi instance của class gọi trực tiếp đến nó.

Ví dụ về method trong Python:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

    def bark(self):
        print(f"{self.name} is barking.")

dog_lucky = Dog("Lucky", 3)
dog_lucky.bark()
# Output: Lucky is barking.

dog_bobby = Dog("Bobby", 5)
dog_bobby.bark()
# Output: Bobby is barking.
```

Ví dụ về method với tham số trong Python:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

    def bark(self, sound):
        print(f"{self.name} is barking {sound}")

dog_lucky = Dog("Lucky", 3)
dog_lucky.bark("Woof Woof")
# Output: Lucky is barking Woof Woof

dog_bobby = Dog("Bobby", 5)
dog_bobby.bark("Woof")
# Output: Bobby is barking Woof
```

#### Static Method

Static method là phương thức không phụ thuộc vào instance của class và không thể truy cập hoặc thay đổi trạng thái của instance.
Static method được định nghĩa bằng cách sử dụng decorator @staticmethod.

Ví dụ về static method trong Python:

```python
class Math:

    @staticmethod
    def add(x, y):
        return x + y

result = Math.add(5, 3)
print(result)
# Output: 8
```

Ví dụ về lỗi khi thay đổi trạng thái của instance trong static method:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

    @staticmethod
    def change_name(new_name):
        self.name = new_name  # This will raise an error

dog_lucky = Dog("Lucky", 3)
dog_lucky.change_name("Buddy")
# Output: NameError: name 'self' is not defined
```

#### Một số phương thức đặc biệt (Magic Methods)

Phương thức đặc biệt (magic methods) trong Python là các phương thức có tên bắt đầu và kết thúc bằng hai dấu gạch dưới (__).
Chúng được sử dụng để định nghĩa các hành vi đặc biệt của các đối tượng, chẳng hạn như cách chúng được khởi tạo, so sánh, hoặc biểu diễn dưới dạng chuỗi.
__init__() cũng là một phương thức đặc biệt.

Ví dụ về phương thức đặc biệt __str__() trong Python:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

dog_lucky = Dog("Lucky", 3)
print(dog_lucky)
# Output: <__main__.Dog object at 0x7f9b8c0c0d30>

class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

    def __str__(self):
        return f"Dog(Name: {self.name}, Age: {self.age})"

dog_lucky = Dog("Lucky", 3)
print(dog_lucky)
# Output: Dog(Name: Lucky, Age: 3)
```

Ví dụ về phương thức đặc biệt __len__() trong Python:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

dog_lucky = Dog("Lucky", 3)
print(len(dog_lucky))
# Output: TypeError: object of type 'Dog' has no len()

class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute
    
    def __len__(self):
        return self.age

dog_lucky = Dog("Lucky", 3)
print(len(dog_lucky))
# Output: 3
```

Ví dụ về phương thức đặc biệt __add__() trong Python:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

dog_lucky = Dog("Lucky", 3)
dog_bobby = Dog("Bobby", 5)
print(dog_lucky + dog_bobby)
# Output: TypeError: unsupported operand type(s) for +: 'Dog' and 'Dog'

class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute
    
    def __add__(self, other):
        return self.age + other.age

dog_lucky = Dog("Lucky", 3)
dog_bobby = Dog("Bobby", 5)
print(dog_lucky + dog_bobby)
# Output: 8
```

### 2.3. Tính đóng gói và Private Membership

#### Private Membership

Trong Python, các thành viên (thuộc tính và phương thức) của một lớp có thể được định nghĩa là private bằng cách đặt tên của chúng bắt đầu bằng hai dấu gạch dưới (__).

Ví dụ về private membership trong Python:

```python
class Dog:
    def __init__(self, name, age):
        self.__name = name  # Private instance attribute
        self.__age = age    # Private instance attribute
    
    def get_name(self):
        return self.__name
    
    def get_age(self):
        return self.__age
    
    def set_age(self, age):
        if age > 0:
            self.__age = age
        else:
            print("Age must be positive.")

dog_lucky = Dog("Lucky", 3)
print(dog_lucky.get_name())
print(dog_lucky.get_age())
# Output: Lucky
#         3

dog_lucky.set_age(5)
print(dog_lucky.get_age())
# Output: 5

dog_lucky.set_age(-2)
# Output: Age must be positive.

print(dog_lucky.get_age())
# Output: 5

print(dog_lucky.__name)
# Output: AttributeError: 'Dog' object has no attribute '__name'

print(dog_lucky.__age)
# Output: AttributeError: 'Dog' object has no attribute '__age'
```

#### Property Decorator
Trong Python, Property Decorator (@property) được sử dụng để tạo các thuộc tính (properties) trong một lớp mà không cần phải gọi các phương thức getter và setter một cách trực tiếp.

Ví dụ về Property Decorator trong Python:

```python
class Dog:
    def __init__(self, name, age):
        self.__name = name  # Private instance attribute
        self.__age = age    # Private instance attribute

    @property
    def name(self):
        return self.__name
    
    @property
    def age(self):
        return self.__age
    
    @age.setter
    def age(self, age):
        if age > 0:
            self.__age = age
        else:
            print("Age must be positive.")

dog_lucky = Dog("Lucky", 3)
print(dog_lucky.name)
print(dog_lucky.age)

# Output: Lucky
#         3

dog_lucky.age = 5
print(dog_lucky.age)
# Output: 5

dog_lucky.age = -2
# Output: Age must be positive.
print(dog_lucky.age)
# Output: 5
```

### 2.4. Tính đa hình và ghi đè phương thức (Method Overriding)

Ví dụ về polymorphism và method overriding trong Python:

```python
class Animal:
    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

dog = Dog()
print(dog.speak())
# Output: Woof!

cat = Cat()
print(cat.speak())
# Output: Meow!
```

### 2.5.Tính kế thừa và super()

Trong phần này, ta sẽ lấy ví dụ có một lớp cha (parent class) là Animal và hai lớp con (child classes) là Dog và Cat kế thừa từ lớp cha Animal.

#### Kế thừa cơ bản

Ví dụ về inheritance trong Python.
Trong ví dụ này, lớp con Dog và Cat không có constructor riêng nên chúng sẽ sử dụng constructor của lớp cha Animal.

```python
class Animal:
    def __init__(self, name):
        self.name = name  # Instance attribute
    
    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")

print(dog.speak())
print(cat.speak())
# Output: Buddy says Woof!
#         Whiskers says Meow!
```

Ví dụ về super() trong Python.
Với việc sử dụng super(), ta có thể gọi constructor của lớp cha trong constructor của lớp con để khởi tạo các thuộc tính của lớp cha.
Điều này cho phép các lớp con kế thừa và mở rộng chức năng của lớp cha một cách hiệu quả.

```python
class Animal:
    def __init__(self, name):
        self.name = name  # Instance attribute
    
    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Call the constructor of the parent class
        self.breed = breed      # Additional instance attribute
    
    def speak(self):
        return f"{self.name} the {self.breed} says Woof!"

dog = Dog("Buddy", "Golden Retriever")
print(dog.speak())
# Output: Buddy the Golden Retriever says Woof!
```

#### Kế thừa đa cấp (Multilevel Inheritance)

```python
class Animal:
    def __init__(self, name):
        self.name = name  # Instance attribute
    
    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")


class Canine(Animal):
    def bark(self):
        return f"{self.name} is barking."

class Dog(Canine):
    def speak(self):
        return f"{self.name} says Woof!"

dog = Dog("Buddy")
print(dog.speak())
print(dog.bark())
# Output: Buddy says Woof!
#         Buddy is barking.
```

#### Kế thừa đa hình (Multiple Inheritance)

Ví dụ về multiple inheritance trong Python:

```python
class Flyer:
    def fly(self):
        return "Flying"

class Swimmer:
    def swim(self):
        return "Swimming"

class Duck(Flyer, Swimmer):
    def quack(self):
        return "Quack!"

class Fish(Swimmer):
    pass

class Bird(Flyer):
    pass

duck = Duck()
print(duck.fly())
print(duck.swim())
print(duck.quack())
# Output: Flying
#         Swimming
#         Quack!

fish = Fish()
print(fish.swim())
# Output: Swimming

bird = Bird()
print(bird.fly())
# Output: Flying
```

Ví dụ về thứ tự kế thừa trong Python và method resolution order (MRO):

```python
class A:
    def introduce(self):
        return "I am A"

class B():
    def introduce(self):
        return "I am B"

class C(A, B):
    pass

c = C()
print(c.introduce())
# Output: I am A
print(C.__mro__)
# Output: (<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class 'object'>)
```
Trong ví dụ trên, lớp C kế thừa từ cả lớp A và lớp B.
Khi ta gọi phương thức introduce() trên instance của lớp C, Python sẽ tìm kiếm phương thức này theo thứ tự kế thừa (MRO).
Vì lớp A được liệt kê trước lớp B trong định nghĩa của lớp C, nên phương thức introduce() của lớp A sẽ được gọi.

#### So sánh Inheritance và Composition

Điểm khác biệt chính giữa inheritance và composition là:
- Inheritance (kế thừa) là một mối quan hệ "là một" (is-a relationship), trong đó một lớp con kế thừa các thuộc tính và phương thức từ một lớp cha.
- Composition (hợp thành) là một mối quan hệ "có một" (has-a relationship), trong đó một lớp chứa các đối tượng của các lớp khác như các thành phần của nó.

Ví dụ:
- Ta có lớp cha `Animal` và lớp con `Dog`. Trong trường hợp này, `Dog` là một loại `Animal` (`Dog` is an `Animal`), vì vậy ta sử dụng inheritance.
- Ta có lớp `Car` và lớp `Engine`. Trong trường hợp này, một `Car` có một `Engine` (`Car` has an `Engine`), vì vậy ta sử dụng composition.

Ví dụ về composition trong Python:

```python
class Engine:
    def start(self):
        return "Engine started"

class Car:
    def __init__(self):
        self.engine = Engine()  # Car has an Engine

    def start(self):
        return self.engine.start()  # Delegating start to Engine

car = Car()
print(car.start())
# Output: Engine started
```

### 2.6. Tính trừu tượng và ABC

Trong Python, ta có thể sử dụng module `ABC` (Abstract Base Classes) để định nghĩa các lớp trừu tượng và các phương thức trừu tượng.
Ví dụ về lớp trừu tượng và phương thức trừu tượng trong Python:

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

dog = Dog()
cat = Cat()

dog.speak()  # Output: Woof!
cat.speak()  # Output: Meow!
```

Nếu ta cố gắng tạo một instance của lớp trừu tượng hoặc không triển khai tất cả các phương thức trừu tượng trong lớp con, Python sẽ ném ra lỗi.

```python

from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass

class Dog(Animal):
    pass

dog = Dog()  # This will raise an error
# Output: TypeError: Can't instantiate abstract class Dog with abstract methods speak
```

So sánh với việc không sử dụng lớp trừu tượng:

```python
class Animal:
    def speak(self):
        raise NotImplementedError("Subclasses must implement this method")

class Dog(Animal):
    pass

dog = Dog() # This will not raise an error
dog.speak()  # This will raise an error
# Output: NotImplementedError: Subclasses must implement this method
```

---

## Luyện tập

<details>
<summary>Câu hỏi trắc nghiệm</summary>

1. Trong Python, phương thức khởi tạo của lớp được định nghĩa bằng tên nào?

A. `__start__`
B. `__init__`
C. `__construct__`
D. `__new__`

2. Cho đoạn code:

```python
class A:
    x = 10

a1 = A()
a2 = A()
a1.x = 20
print(a2.x)
```

Kết quả in ra là gì?

A. 10
B. 20
C. None
D. Lỗi

3. Khái niệm nào sau đây thể hiện Polymorphism?

A. Giấu chi tiết cài đặt, chỉ để lộ giao diện cần thiết
B. Nhiều lớp có thể định nghĩa cùng một phương thức với hành vi khác nhau
C. Khả năng kế thừa thuộc tính và phương thức từ lớp cha
D. Sử dụng property để bảo vệ dữ liệu nội bộ

4. Trong Python, để tạo lớp trừu tượng, ta cần import module nào?

A. `abstract`
B. `abc`
C. `abstractbase`
D. `oop`

5. Đoạn code sau in ra gì?

```python
class A:
    def greet(self):
        return "Hello from A"

class B(A):
    def greet(self):
        return "Hello from B"

class C(A):
    pass

class D(B, C):
    pass

d = D()
print(d.greet())
```

A. Hello from A
B. Hello from B
C. Hello from C
D. Lỗi do đa kế thừa

6. Cho lớp:

```python
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name
```

Câu nào đúng?

A. Có thể gán trực tiếp `p.name = "Nam"` để đổi tên
B. `name` chỉ đọc, không thể gán trực tiếp
C. `name` bị ẩn hoàn toàn, không thể đọc
D. Sử dụng `p.get_name()` để đọc

7. `@staticmethod` khác `@classmethod` ở điểm nào?

A. `staticmethod` nhận cls làm tham số đầu tiên
B. `classmethod` không cần đối tượng instance để gọi
C. `staticmethod` không truy cập được cả class-level và instance-level
D. `classmethod` chỉ dùng trong abstract class

8. Trong Python, `__str__()` và `__repr__()` khác nhau thế nào?

A. `__str__()` dành cho người dùng, `__repr__()` dành cho lập trình viên
B. `__str__()` luôn ưu tiên gọi trước `__repr__()`
C. `__repr__()` không thể override
D. Không có sự khác biệt

9. Khi nào nên dùng composition thay vì inheritance?

A. Khi lớp con cần tái sử dụng logic của lớp cha nhưng không phải là một kiểu của lớp cha
B. Khi muốn override toàn bộ hành vi lớp cha
C. Khi cần chia sẻ static method giữa nhiều lớp
D. Khi muốn thay đổi MRO

10. Cho đoạn code:

```python
class Engine:
    def start(self): return "Engine started"

class Car:
    def __init__(self):
        self.engine = Engine()
    def start(self):
        return self.engine.start()

c = Car()
print(c.start())
```

Quan hệ giữa Car và Engine là gì?

A. Inheritance (Kế thừa)
B. Polymorphism (Đa hình)
C. Composition (Thành phần)
D. Encapsulation (Đóng gói)

</details>

<details>
<summary>Đáp án</summary>

1. B
2. A
3. B
4. B
5. B
6. B
7. C
8. A
9. A
10. C

</details>

<details>
<summary>Bài tập thực hành</summary>

**1. Lớp đơn giản: `Rectangle`** Viết lớp `Rectangle` có thuộc tính width và height. Cài đặt: phương thức khởi tạo `__init__()`, phương thức `area()` trả về diện tích, phương thức `perimeter()` trả về chu vi, phương thức `__str__()` để in dạng `"Rectangle(width=..., height=...)"`.

**2. Tính đóng gói (Encapsulation) & thuộc tính (property)** Viết lớp `BankAccount` với: thuộc tính riêng `_balance` (không truy cập trực tiếp từ ngoài), phương thức `deposit(amount)` và `withdraw(amount)` (nếu rút vượt quá, `raise ValueError`), `@property balance` để đọc số dư (không cho phép ghi trực tiếp).

**3. Lớp trừu tượng (`ABC`) và `interface`** Sử dụng `ABC` để định nghĩa `Shape` trừu tượng với phương thức `abstract area()` và `perimeter()`. Cài đặt `Circle` và `Square` kế thừa `Shape`.

**4. Operator overloading (`__add__`, `__repr__`)** Viết lớp `Vector2D` biểu diễn vectơ 2 chiều với x, y. Cài đặt `__add__` để cộng hai vectơ, `__repr__` để in đối tượng.

**5. Classmethod & Staticmethod** Viết lớp `Temperature` với: thuộc tính `instance celsius`, `@classmethod from_fahrenheit(cls, f)` trả về instance từ độ F, `@staticmethod c_to_f(c)` trả về giá trị Fahrenheit từ Celsius.

**6. Polymorphism nâng cao: danh sách Shape** Cho `Shape (abstract)` có `area()`. Viết hàm `total_area(shapes: list[Shape])` trả về tổng diện tích của các hình khác nhau (Circle, Rectangle, Square). Thử nghiệm với danh sách hỗn hợp.

**7. Thiết kế bài toán thực tế: TodoList với serialization** Xây dựng hệ `TodoItem` và `TodoList`: `TodoItem` có title, done (bool), toggle() để đổi trạng thái, `TodoList` quản lý danh sách `TodoItem` với phương thức `add(item)`, `remove(title)`, `list_all()` trả danh sách, thêm phương thức `to_dict()` và `from_dict()` để serialize/deserialze (ví dụ lưu/đọc JSON).

</details>

<details>
<summary>Lời giải</summary>

1. Lớp đơn giản: `Rectangle`

```python
class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def __str__(self) -> str:
        return f"Rectangle(width={self.width}, height={self.height})"

# Ví dụ sử dụng
r = Rectangle(3, 4)
print(r)               # Rectangle(width=3, height=4)
print(r.area())        # 12
print(r.perimeter())   # 14
```

2. Tính đóng gói (Encapsulation) & thuộc tính (property)

```python
class BankAccount:
    def __init__(self, initial: float = 0.0):
        if initial < 0:
            raise ValueError("Initial balance cannot be negative")
        self._balance = float(initial)

    @property
    def balance(self) -> float:
        return self._balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Withdraw amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount

# Ví dụ
acc = BankAccount(100)
acc.deposit(50)
try:
    acc.withdraw(200)
except ValueError as e:
    print("Lỗi:", e)   # Lỗi: Insufficient funds
print(acc.balance)     # 150.0
```

3. Lớp trừu tượng (`ABC`) và `interface`

```python
class Animal:
    def speak(self) -> str:
        raise NotImplementedError("Subclasses must implement speak()")

class Dog(Animal):
    def speak(self) -> str:
        return "Woof"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow"

def animal_says(animal: Animal) -> None:
    print(animal.speak())

# Ví dụ
animal_says(Dog())  # Woof
animal_says(Cat())  # Meow
```

4. Operator overloading (`__add__`, `__repr__`)

```python
class Vector2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other):
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"Vector2D(x={self.x}, y={self.y})"

# Ví dụ
v1 = Vector2D(1, 2)
v2 = Vector2D(3, 4)
print(v1 + v2)  # Vector2D(x=4, y=6)
```

5. Classmethod & Staticmethod

```python
class Temperature:
    def __init__(self, celsius: float):
        self.celsius = float(celsius)

    @classmethod
    def from_fahrenheit(cls, f: float):
        c = (f - 32) * 5.0 / 9.0
        return cls(c)

    @staticmethod
    def c_to_f(c: float) -> float:
        return c * 9.0 / 5.0 + 32

# Ví dụ
t = Temperature.from_fahrenheit(98.6)
print(t.celsius)           # 37.0
print(Temperature.c_to_f(0))  # 32.0
```

6. Polymorphism nâng cao: danh sách Shape

```python
# Sử dụng Shape, Circle, Rectangle từ bài trước; nếu chưa có, định nghĩa nhanh:
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        ...

class Circle(Shape):
    def __init__(self, radius: float): self.radius = radius
    def area(self): return math.pi * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, w, h): self.w = w; self.h = h
    def area(self): return self.w * self.h

class Square(Shape):
    def __init__(self, side): self.side = side
    def area(self): return self.side * self.side

def total_area(shapes: list[Shape]) -> float:
    return sum(s.area() for s in shapes)

# Ví dụ
shapes = [Circle(1), Rectangle(2, 3), Square(4)]
print(total_area(shapes))  # pi*1 + 6 + 16
```

7. Thiết kế bài toán thực tế: TodoList với serialization

```python
import json
from typing import List, Dict

class TodoItem:
    def __init__(self, title: str, done: bool = False):
        self.title = title
        self.done = bool(done)

    def toggle(self):
        self.done = not self.done

    def to_dict(self) -> Dict:
        return {"title": self.title, "done": self.done}

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(d["title"], d.get("done", False))

    def __repr__(self):
        return f"TodoItem(title={self.title!r}, done={self.done})"

class TodoList:
    def __init__(self):
        self._items: List[TodoItem] = []

    def add(self, item: TodoItem):
        self._items.append(item)

    def remove(self, title: str):
        self._items = [it for it in self._items if it.title != title]

    def list_all(self) -> List[TodoItem]:
        return list(self._items)

    def to_json(self) -> str:
        return json.dumps([it.to_dict() for it in self._items])

    @classmethod
    def from_json(cls, s: str):
        data = json.loads(s)
        tl = cls()
        for d in data:
            tl.add(TodoItem.from_dict(d))
        return tl

# Ví dụ
tl = TodoList()
tl.add(TodoItem("Write report"))
tl.add(TodoItem("Read paper", True))
js = tl.to_json()
print(js)
tl2 = TodoList.from_json(js)
print(tl2.list_all())
```

</details>
