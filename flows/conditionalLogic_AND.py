from crewai.flow.flow import listen, start, Flow, and_

class ANDExampleFlow(Flow):

    @start()
    def first_method(self):
        return "Output from first method"
    
    @listen(first_method)
    def second_method(self):
        return "Output from second method"
    
    @listen(and_(first_method,second_method))
    def logger(self,result):
        print("Logger contains is ",result)
    
    
flow = ANDExampleFlow()
flow.plot("ANDExampleFlow")
final = flow.kickoff()