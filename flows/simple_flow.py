from crewai.flow.flow import Flow, listen, start

class FinalOutputExample(Flow):
    @start()
    def first_method(self):
        return "First method Completed"

    @listen(first_method)
    def second_method(self):
        return "Second method Completed"


flow = FinalOutputExample()
flow.plot("my_flow")
final_output = flow.kickoff()

print(final_output)
print(flow.plot("my_flow"))
