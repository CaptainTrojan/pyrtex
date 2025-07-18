#!/usr/bin/env python3
"""
Example 3: PDF Invoice Data Extraction

This example demonstrates extracting structured data from PDF invoice documents.
It parses invoice information and organizes it into a structured format.
"""

from datetime import date
from pathlib import Path

from pydantic import BaseModel

from pyrtex import Job


# Define the input schema for PDF data
class PDFInvoiceInput(BaseModel):
    file_path: Path  # PDF file path
    parsing_focus: str  # What specific aspects to focus on during parsing
    detail_level: str  # Level of detail required (basic/comprehensive)


# Define the output schema for invoice information
class InvoiceItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float


class CustomerInfo(BaseModel):
    name: str
    address: str


class CompanyInfo(BaseModel):
    name: str
    address: str
    phone: str


class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    due_date: str
    customer_id: str
    company: CompanyInfo
    customer: CustomerInfo
    items: list[InvoiceItem]
    subtotal: float
    tax_rate: float
    tax_amount: float
    total: float
    payment_terms: str


def main():
    # Set up the job
    job = Job[InvoiceData](
        model="gemini-2.0-flash-lite-001",
        output_schema=InvoiceData,
        prompt_template="""
        Extract {{ detail_level }} information from this PDF invoice with focus on {{ parsing_focus }}:
        
        Please parse all the invoice details including:
        - Invoice number, date, due date, and customer ID
        - Company information (name, address, phone)
        - Customer details (name and full address)
        - All line items with descriptions, quantities, unit prices, and totals
        - Financial totals (subtotal, tax rate, tax amount, total)
        - Payment terms
        
        Convert all monetary amounts to float values (remove $ signs and commas).
        Be precise and extract exactly what appears on the invoice.
        """,
    )

    # Read invoice PDF
    data_dir = Path(__file__).parent / "data"
    pdf_path = data_dir / "sample_invoice.pdf"

    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_path}")
        print(
            "Please run 'python generate_sample_data.py' first to create sample files."
        )
        return

    # Add the extraction request using Path object directly
    job.add_request(
        "invoice_extraction",
        PDFInvoiceInput(
            file_path=pdf_path,  # Pass Path object directly
            parsing_focus="financial accuracy and line item details",
            detail_level="comprehensive",
        ),
    )

    # Process and get results
    print("Extracting invoice data from PDF...")
    print(f"Processing: {pdf_path}")

    # Submit job, wait for completion, then get results
    results = list(job.submit().wait().results())

    # Display results - note: we use request key to ensure we get the right result
    for result in results:
        print(f"\nProcessing result for request key: '{result.request_key}'")
        if result.was_successful:
            invoice = result.output
            print(f"\n--- Invoice Data Extraction ---")
            print(f"Invoice: {invoice.invoice_number}")
            print(f"Date: {invoice.date}")
            print(f"Due Date: {invoice.due_date}")
            print(f"Customer ID: {invoice.customer_id}")

            print(f"\nCompany:")
            print(f"  Name: {invoice.company.name}")
            print(f"  Address: {invoice.company.address}")
            print(f"  Phone: {invoice.company.phone}")

            print(f"\nCustomer:")
            print(f"  Name: {invoice.customer.name}")
            print(f"  Address: {invoice.customer.address}")

            print(f"\nItems:")
            for item in invoice.items:
                print(f"  • {item.description}")
                print(
                    f"    Qty: {item.quantity} × ${item.unit_price:.2f} = ${item.total:.2f}"
                )

            print(f"\nFinancials:")
            print(f"  Subtotal: ${invoice.subtotal:.2f}")
            print(f"  Tax ({invoice.tax_rate:.2f}%): ${invoice.tax_amount:.2f}")
            print(f"  Total: ${invoice.total:.2f}")
            print(f"  Payment Terms: {invoice.payment_terms}")

            # Calculate some additional insights
            print(f"\n--- Invoice Analysis ---")
            print(f"Number of line items: {len(invoice.items)}")
            print(
                f"Average item value: ${sum(item.total for item in invoice.items) / len(invoice.items):.2f}"
            )
            print(
                f"Effective tax rate: {(invoice.tax_amount / invoice.subtotal * 100):.2f}%"
            )
        else:
            print(f"Error processing '{result.request_key}': {result.error}")

    print(f"\n--- Processing Stats ---")
    print(f"Input: PDF invoice ({pdf_path.name})")
    print(f"File size: {pdf_path.stat().st_size:,} bytes")
    print(
        f"Tokens used: {results[0].usage_metadata.get('totalTokenCount', 'N/A') if results else 'N/A'}"
    )
    print(
        f"💡 Note: Always use request keys to map results back to inputs when processing multiple items."
    )


if __name__ == "__main__":
    main()
