import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class ProductSearchInput(BaseModel):
    query: str = Field(..., description="Product name, category, or keyword to search for")

class ProductSearchTool(BaseTool):
    name: str = "search_product"
    description: str = "Searches the product catalog (CSV) and returns matching products with price and description."
    args_schema: type[BaseModel] = ProductSearchInput

    def _run(self, query: str) -> str:
        df = pd.read_csv("/workspaces/bles/E-Commerce Support/products.csv")

        mask = (
            df["product_name"].str.contains(query, case=False, na=False) |
            df["category"].str.contains(query, case=False, na=False) |
            df["description"].str.contains(query, case=False, na=False)
        )
        results = df[mask]

        if results.empty:
            return f"No products found matching '{query}'."

        output = []
        for _, row in results.iterrows():
            output.append(
                f"{row['product_name']} ({row['category']}) - {row['price'].strip()}\n"
                f"  {row['description']}"
            )
        return "\n".join(output)