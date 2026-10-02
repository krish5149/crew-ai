# E-Commerce Support Crew
A command-line customer-support assistant built with [crewAI](https://docs.crewai.com/). It classifies a customer message and routes it to a specialist crew for product search, order tracking, return/refund requests, or complaint escalation.
## Features
- Classifies messages into Product Search, Order Status, Returns & Refunds, Complaint / Escalation, or FAQ / General.
- Searches a CSV product catalog and can create an order in a local SQLite database.
- Looks up shipping and delivery information from the orders database.
- Evaluates return/refund requests against the return policy knowledge source.
- Creates a support ticket for complaints and can pause for human review when escalation is required.
## How it works
`src/e_commerce_support/main.py` defines the crewAI flow. An LLM classifies the user's message, and the flow routes it to the corresponding crew. Crews, agents, and tasks are configured in YAML files under `src/e_commerce_support/crews/`.
| Intent | Crew / action |
| --- | --- |
| Product Search | Searches `products.csv`, extracts the requested product and quantity, then creates an order in `orders.db`. |
| Order Status | Searches `orders.db` for order and delivery details. |
| Returns & Refunds | Parses the request, checks order and policy details, and drafts a decision and customer response. |
| Complaint / Escalation | Creates a support ticket and routes cases requiring human attention to a review step. |
| FAQ / General | The classifier supports this category, but the current flow does not yet define a handler for it. |
## Project structure
```text
E-Commerce Support/
├── knowledge/
│   └── returnpolicy.txt
├── products.csv
├── src/e_commerce_support/
│   ├── crews/
│   │   ├── EsclationManager/
│   │   ├── OrderStatus/
│   │   ├── ProductSearch/
│   │   └── ReturnRefund/
│   ├── data/
│   │   └── orders.db
│   ├── tools/
│   ├── LLMClassifier.py
│   ├── main.py
│   └── placing_order.py
└── pyproject.toml
```
## Requirements
- Python `>=3.10,<3.14`
- [UV](https://docs.astral.sh/uv/) (recommended)
- An OpenRouter API key for the configured `openai/gpt-4o-mini` model
## Setup and run
From this directory:
```bash
uv sync
```
Set `OPENROUTER_API_KEY` in your environment, then start the flow:
```bash
uv run python -m e_commerce_support.main
```
The program prompts for a customer-support message in the terminal.
## Configuration
- Edit `src/e_commerce_support/crews/*/config/agents.yaml` to adjust agent roles and behavior.
- Edit `src/e_commerce_support/crews/*/config/tasks.yaml` to adjust task instructions.
- Update `products.csv` to change the searchable product catalog.
- Update `knowledge/returnpolicy.txt` to change the return-policy knowledge source.
- `orders.db` stores order information used by the order lookup and order-placement tools.
## Important security and setup notes
This repository needs a security and portability cleanup before it is safe or ready to run in a fresh environment. Some crew modules currently contain a hard-coded API credential, and application files use machine-specific absolute paths for `.env`, product, policy, and database files. **Do not publish or use any exposed credential. Revoke and rotate it, move credentials to environment variables, and replace absolute paths with paths derived from the project location before running or deploying this project.**
The returns/refunds flow currently uses LLM task instructions to describe refund processing; it does not integrate with a real payment processor. Product Search currently proceeds to create an order, so use care when testing it with a live database.
## License
No license is currently included. Add a `LICENSE` file if you intend to grant others permission to use, modify, or distribute this project.
