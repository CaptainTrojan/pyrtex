#!/usr/bin/env python3
"""
Example 4: PDF Processing

Demonstrates extracting structured data from PDF documents.
"""

from pathlib import Path
from pydantic import BaseModel
from pyrtex import Job


class InvoiceItem(BaseModel):
    """Individual line item from invoice."""
    description: str
    quantity: int
    unit_price: float
    total: float


class InvoiceData(BaseModel):
    """Invoice information extracted from PDF."""
    invoice_number: str
    date: str
    customer_name: str
    items: list[InvoiceItem]
    total_amount: float


class PDFInput(BaseModel):
    """Input schema for PDF processing."""
    file_path: Path


def main():
    data_dir = Path(__file__).parent / "data"
    pdf_path = data_dir / "sample_invoice.pdf"
    
    if not pdf_path.exists():
        print("Sample PDF not found. Run generate_sample_data.py first.")
        return
    
    job = Job(
        model="gemini-2.0-flash-lite-001",
        output_schema=InvoiceData,
        prompt_template="Extract invoice data from this PDF: {{ file_path }}",
    )
    
    job.add_request("invoice", PDFInput(file_path=pdf_path))
    
    for result in job.submit().wait().results():
        if result.was_successful:
            invoice = result.output
            print(f"Invoice: {invoice.invoice_number}")
            print(f"Date: {invoice.date}")
            print(f"Customer: {invoice.customer_name}")
            print(f"Total: ${invoice.total_amount}")
            print(f"Items: {len(invoice.items)}")


if __name__ == "__main__":
    main()
