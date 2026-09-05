from crewai.flow.flow import Flow, listen, start, or_

class OrExampleFlow(Flow):

    @start()
    def first_method(self):
        return "Output from first method"

    @listen(first_method)
    def second_method(self):
        return "Output from second method"

    @listen(or_(first_method,second_method))
    def logger(self,result):
        print("Logger contains is ",result)


flow = OrExampleFlow()
path = flow.plot("./flow/OrExampleFlow")
print(path)
final = flow.kickoff()

