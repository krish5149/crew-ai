from crewai.flow.flow import Flow, listen, start

class ShoppingCartFlow(Flow):

    @start()
    def initialize_kart(self):
        print("Kart_ID : ",self.state['id'])
        self.state['items'] = []
        self.state['total'] = 0.0
        self.state['discount'] = 0.0

    @listen(initialize_kart)
    def add_items(self):
        self.state['items'].append({"name":"Laptop","price":500})
        self.state['items'].append({"name":"mouse","price":40})
        self.state['customer'] = "Jhon"

    @listen(add_items)
    def apply_discount(self):
        self.state['discount'] = 10.0
        subtotal = sum(item['price'] for item in self.state['items'])
        self.state['total'] = subtotal * (1-self.state['discount']/100)

    @listen(apply_discount)
    def checkout(self):
        print(f"Customer: {self.state['customer']}")
        print(f"Items: {self.state['items']}")
        print(f"Discount: {self.state['discount']}%")
        print(f"Total: ${self.state['total']:.2f}")
        print(f"Full state: {self.state}")


flow = ShoppingCartFlow()

flow.plot("shopping_cart_flow")

flow.kickoff()