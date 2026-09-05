import sqlite3
import random
import os
from datetime import date

DB_DIR = "/workspaces/bles/E-Commerce Support/src/e_commerce_support/data"
DB_PATH = os.path.join(DB_DIR, "orders.db")


def generate_order_id(cursor) -> str:
    """Generate a unique 4-digit order ID, retrying if it already exists."""
    while True:
        order_id = str(random.randint(1000, 9999))
        cursor.execute("SELECT 1 FROM orders WHERE order_id = ?", (order_id,))
        if cursor.fetchone() is None:
            return order_id


def create_order(
    product_name: str,
    quantity: int,
    price: float,
    order_date: str = "",
    customer_name: str = "Kinsey",
    customer_email: str = "kinsey@latvaria.com",                                                               
    status: str = "Processing",
    shipped_date: str = "10/11/2027",
    expected_delivery: str = "10/12/2027", 
    tracking_number: str = None, # type: ignore
) -> str:
    """Insert a new order into the DB and return the generated order_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    order_id = generate_order_id(cursor)
    order_date = order_date or date.today().isoformat()

    cursor.execute("""
        INSERT INTO orders
        (order_id, customer_name, customer_email, product_name, quantity, price,
         status, order_date, shipped_date, expected_delivery, tracking_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        order_id, customer_name, customer_email, product_name, quantity, price,
        status, order_date, shipped_date, expected_delivery, tracking_number
    ))

    conn.commit()
    conn.close()
    return order_id


# if __name__ == "__main__":
#     new_id = create_order(
#         customer_name="Alice Brown",
#         customer_email="alice@example.com",
#         product_name="Wireless Earbuds",
#         quantity=2,
#         price=2499.00
#     )
#     print(f"Order created with ID: {new_id}")