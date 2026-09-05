from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel
import random

class ExampleState(BaseModel):
    success_state: bool = True

class RouterFlow(Flow[ExampleState]):

    @start()
    def start_method(self):
        print("Started First Method....")
        self.state.success_state = random.choice([True,False])

    @router(start_method)
    def router_method(self):
        if self.state.success_state:
            return "success"
        else:
            return "failure"

    @listen("success")
    def success_method(self):
        print("Success_Method Running.....")

    @listen("failure")
    def failure_method(self):
       print("Failure_Method Running.....")


flow = RouterFlow()
flow.plot("RouterFlow2")
flow.kickoff()