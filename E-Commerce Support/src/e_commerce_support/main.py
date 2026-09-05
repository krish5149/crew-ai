import os

from dotenv import load_dotenv
from pydantic import BaseModel

from crewai import LLM
from crewai.flow.flow import Flow, listen, router, start
from crewai.flow.human_feedback import human_feedback

from e_commerce_support.LLMClassifier import classify_intent
from e_commerce_support.placing_order import create_order

from e_commerce_support.crews.ProductSearch.ProductSearchCrew import ProductSearchCrew
from e_commerce_support.crews.OrderStatus.OrderStatusCrew import OrderStatusCrew
from e_commerce_support.crews.EsclationManager.EsclationManagerCrew import EsclationManagerCrew
from e_commerce_support.crews.ReturnRefund.ReturnRefundCrew import ReturnRefundCrew

# Load environment variables
load_dotenv("/workspaces/bles/.env")


openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is not available. "
        "Add it to /workspaces/bles/.env."
    )


open_router = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)


class ECommerceState(BaseModel):
    query: str = ""
    intent: str = ""

    product_name: str = ""
    product_quantity: int = 0
    product_price: float = 0.0

    suggestion: str = ""
    ticket_id: int = 0
    human_handoff: bool = False

    order_id: str = ""
    order_status: str = ""

    review_result: str = ""


class ECommerceFlow(Flow[ECommerceState]):

    @start()
    def user_query(self):
        query = input("How can I assist you? ").strip()

        if not query:
            raise ValueError("Customer query cannot be empty.")

        self.state.query = query
        return query

    @router(user_query)
    def query_classifier(self):
        intent = classify_intent(self.state.query)

        if not intent:
            raise ValueError("Intent classifier returned an empty result.")

        self.state.intent = str(intent).strip()

        print(f"Detected intent: {self.state.intent}")

        return self.state.intent

    # ---------------------------------------------------------
    # ORDER STATUS FLOW
    # ---------------------------------------------------------

    @listen("Order Status")
    def track_order_status(self):
        inputs = {
            "query": self.state.query,
        }

        result = (
            OrderStatusCrew()
            .Crew()     #type:ignore
            .kickoff(inputs=inputs)
        )

        self.state.order_status = str(result)

        print("\nOrder status result:")
        print(self.state.order_status)

        return result

    # ---------------------------------------------------------
    # PRODUCT SEARCH AND ORDER PLACEMENT FLOW
    # ---------------------------------------------------------

    @listen("Product Search")
    def product_search(self):
        inputs = {
            "Product": self.state.query,
        }

        result = (
            ProductSearchCrew()
            .product_search_crew()      #type:ignore
            .kickoff(inputs=inputs)
        )

        if result.pydantic is None:
            raise ValueError(
                "ProductSearchCrew did not return Pydantic output. "
                "Verify that output_pydantic is configured on the final task."
            )

        self.state.product_name = result.pydantic.product_name
        self.state.product_quantity = result.pydantic.product_quantity
        self.state.product_price = result.pydantic.product_price

        if hasattr(result.pydantic, "suggestion"):
            self.state.suggestion = result.pydantic.suggestion

        print("\nProduct selected:")
        print(f"Product: {self.state.product_name}")
        print(f"Quantity: {self.state.product_quantity}")
        print(f"Unit price: {self.state.product_price}")

        return result.pydantic

    @listen(product_search)
    def place_order(self, product_result):
        if not self.state.product_name:
            raise ValueError("Product name is empty.")

        if self.state.product_quantity <= 0:
            raise ValueError(
                "Product quantity must be greater than zero."
            )

        if self.state.product_price < 0:
            raise ValueError(
                "Product price cannot be negative."
            )

        total_price = (
            self.state.product_price
            * self.state.product_quantity
        )

        order_id = create_order(
            product_name=self.state.product_name,
            quantity=self.state.product_quantity,
            price=total_price,
        )

        self.state.order_id = str(order_id)

        print("\nOrder created successfully.")
        print(f"Order ID: {self.state.order_id}")
        print(f"Product: {self.state.product_name}")
        print(f"Quantity: {self.state.product_quantity}")
        print(f"Total price: {total_price:.2f}")

        return self.state.order_id

    # ---------------------------------------------------------
    # COMPLAINT AND ESCALATION FLOW
    # ---------------------------------------------------------

    @listen("Complaint / Escalation")
    def complaint_escalation(self):
        inputs = {
            "query": self.state.query,
        }

        result = (
            EsclationManagerCrew()
            .esclation_crew()       #type:ignore
            .kickoff(inputs=inputs)
        )

        if result.pydantic is None:
            raise ValueError(
                "EsclationManagerCrew did not return Pydantic output. "
                "Verify that output_pydantic is configured on the final task."
            )

        escalation_output = result.pydantic

        self.state.human_handoff = (
            escalation_output.human_handoff
        )
        self.state.suggestion = (
            escalation_output.suggestion
        )
        self.state.ticket_id = (
            escalation_output.ticket_id
        )

        print("\nEscalation analysis completed.")
        print(f"Ticket ID: {self.state.ticket_id}")
        print(f"Human handoff: {self.state.human_handoff}")
        print(f"Suggestion: {self.state.suggestion}")

        return escalation_output

    @router(complaint_escalation)
    def human_router(self):
        if self.state.human_handoff:
            return "Human Confirmation"

        return "Human Confirmation No Need"

    # ---------------------------------------------------------
    # HUMAN-IN-THE-LOOP PATH
    # ---------------------------------------------------------

    @listen("Human Confirmation")
    @human_feedback(
        message=(
            "Please review the escalated customer-support ticket. "
            "Enter 'resolved' to approve the resolution or "
            "'rejected' when further action is required."
        ),
        emit=["resolved", "rejected"],
        llm=open_router,
        default_outcome="rejected",
        metadata={
            "Reviewer": "Ramki",
            "Workflow": "E-Commerce Support Escalation",
        },
    )
    def human_review(self):
        print("\nHUMAN REVIEW REQUIRED")
        print(f"Customer query: {self.state.query}")
        print(f"Ticket ID: {self.state.ticket_id}")
        print(f"Suggestion: {self.state.suggestion}")

        
        return {
            "customer_query": self.state.query,
            "ticket_id": self.state.ticket_id,
            "suggestion": self.state.suggestion,
        }

    @listen("resolved")
    def ticket_resolved(self, feedback_result):
        self.state.review_result = "resolved"

        print(
            f"\nTicket {self.state.ticket_id} "
            "was marked as resolved by the reviewer."
        )

        if feedback_result is not None:
            print(
                f"Reviewer feedback: "
                f"{feedback_result.feedback}"
            )

        return "resolved"

    @listen("rejected")
    def ticket_rejected(self, feedback_result):
        self.state.review_result = "rejected"

        print(
            f"\nTicket {self.state.ticket_id} was rejected. "
            "Further investigation is required."
        )

        if feedback_result is not None:
            print(
                f"Reviewer feedback: "
                f"{feedback_result.feedback}"
            )

        return "rejected"

    # ---------------------------------------------------------
    # NO HUMAN REVIEW PATH
    # ---------------------------------------------------------

    @listen("Human Confirmation No Need")
    def no_human_needed(self):
        self.state.review_result = "automatic"

        print(
            f"\nTicket {self.state.ticket_id} created successfully."
        )
        print(
            f"Suggested resolution: {self.state.suggestion}"
        )
        print("Human review was not required.")

        return "handled"

    # ---------------------------------------------------------
    # RETURN & REFUND
    # ---------------------------------------------------------   

    @listen("Returns & Refunds")
    def return_refund(self):

        inputs = {"customer_request":self.state.query}

        result = ReturnRefundCrew().return_refund_crew().kickoff(inputs=inputs) #type: ignore
        return result




if __name__ == "__main__":
    flow = ECommerceFlow()
    final_result = flow.kickoff()

    print("\nFinal flow result:")
    print(final_result)

    print("\nFinal flow state:")
    print(flow.state)