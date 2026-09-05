from pydantic import BaseModel, Field

class ProductSearchOutput(BaseModel):
    product_name: str = Field(...,description="Name of the matched product")
    product_quantity: int = Field(...,description="Quantity requested by the customer, default 1 if not mentioned")
    product_price: float = Field(...,description="Price of the product as a plain number, no $ or commas")