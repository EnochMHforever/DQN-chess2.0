class Restaurant():
    def __int__(self, restaurant_name='just eat', cuisine_type='stir'):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type
    def describe_restaurant(self):
        print("Our restaurant is called: " + self.restaurant_name.title())
        print("Our cuisine type is: " + self.cuisine_type.title())
    def open_restaurant(self):
        print("We are OPEN!")

my_restaurant = Restaurant('just eat','stir')

print("My re na is" + my_restaurant.restaurant_name.title() + ".")
print("My re cu is" + my_restaurant.cuisine_type.title() + ".")