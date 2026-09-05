from pydantic import BaseModel,Field
from crewai.tools import BaseTool
import sqlite3
import os

DB_PATH = "/workspaces/bles/E-Commerce Support/src/e_commerce_support/data/orders.db"

class OrderDetailInput(BaseModel):
    order_id: str = Field(
        ...,
        description="The order ID to look up"
    )

class FetchOrderDetails(BaseTool):
    name: str = "Fetch Order Details"
    description: str = "Looks up an order's status, shipping, and delivery info from the orders database."
    args_schema: type[BaseModel] = OrderDetailInput

    def _run(self, order_id: str) -> str:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return f"No order found with ID '{order_id}'."

        return (
            f"Order ID: {row['order_id']}\n"
            f"Product Name: {row['product_name']}\n"
            f"Quantity: {row['quantity']}\n"
            f"Status: {row['status']}\n"
            f"Order Date: {row['order_date']}\n"
            f"Shipped Date: {row['shipped_date'] or 'Not shipped yet'}\n"
            f"Expected Delivery: {row['expected_delivery'] or 'TBD'}\n"
            f"Tracking Number: {row['tracking_number'] or 'N/A'}"
        )