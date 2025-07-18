#!/usr/bin/env python3
"""
Generate sample PDF and image files for PyRTex examples.
This script creates realistic sample data to demonstrate PDF and image processing capabilities.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import base64

def create_sample_pdf():
    """Create a sample PDF with invoice data using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
    except ImportError:
        print("reportlab not installed. Installing...")
        os.system("pip install reportlab")
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors

    # Create PDF file path
    data_dir = Path(__file__).parent / "data"
    pdf_path = data_dir / "sample_invoice.pdf"
    
    # Create the PDF document
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    story.append(Paragraph("INVOICE", title_style))
    story.append(Spacer(1, 20))
    
    # Company info
    company_style = ParagraphStyle(
        'Company',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6
    )
    story.append(Paragraph("<b>Tech Solutions Inc.</b>", company_style))
    story.append(Paragraph("123 Business Ave", company_style))
    story.append(Paragraph("San Francisco, CA 94102", company_style))
    story.append(Paragraph("Phone: (555) 123-4567", company_style))
    story.append(Spacer(1, 20))
    
    # Invoice details
    invoice_data = [
        ['Invoice #:', 'INV-2025-001'],
        ['Date:', 'July 18, 2025'],
        ['Due Date:', 'August 17, 2025'],
        ['Customer ID:', 'CUST-001']
    ]
    
    invoice_table = Table(invoice_data, colWidths=[1.5*inch, 2*inch])
    invoice_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(invoice_table)
    story.append(Spacer(1, 20))
    
    # Bill to section
    story.append(Paragraph("<b>Bill To:</b>", styles['Normal']))
    story.append(Paragraph("John Doe", styles['Normal']))
    story.append(Paragraph("456 Customer St", styles['Normal']))
    story.append(Paragraph("Austin, TX 78701", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Items table
    items_data = [
        ['Description', 'Quantity', 'Unit Price', 'Total'],
        ['Software License (Annual)', '1', '$1,200.00', '$1,200.00'],
        ['Premium Support Package', '1', '$300.00', '$300.00'],
        ['Training Session (4 hours)', '1', '$500.00', '$500.00'],
        ['Setup and Configuration', '1', '$150.00', '$150.00'],
    ]
    
    items_table = Table(items_data, colWidths=[3*inch, 0.8*inch, 1*inch, 1*inch])
    items_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(items_table)
    story.append(Spacer(1, 20))
    
    # Totals
    totals_data = [
        ['Subtotal:', '$2,150.00'],
        ['Tax (8.25%):', '$177.38'],
        ['Total:', '$2,327.38']
    ]
    
    totals_table = Table(totals_data, colWidths=[1.5*inch, 1*inch])
    totals_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, -1), (-1, -1), 2, colors.black),
    ]))
    
    # Right-align the totals table
    story.append(Spacer(1, 10))
    story.append(Paragraph('<para align="right"><b>Payment Terms: Net 30 days</b></para>', styles['Normal']))
    story.append(Spacer(1, 10))
    
    # Create a container for right-aligned table
    totals_container = Table([[totals_table]], colWidths=[6.5*inch])
    totals_container.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
    ]))
    story.append(totals_container)
    
    # Build PDF
    doc.build(story)
    print(f"Created sample PDF: {pdf_path}")
    return pdf_path

def create_product_catalog_image():
    """Create a sample product catalog image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed. Installing...")
        os.system("pip install Pillow")
        from PIL import Image, ImageDraw, ImageFont
    
    # Create image
    width, height = 800, 1000
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a better font
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        price_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        price_font = ImageFont.load_default()
    
    # Title
    draw.text((50, 30), "PRODUCT CATALOG", fill='black', font=title_font)
    draw.text((50, 70), "Electronics & Accessories", fill='gray', font=header_font)
    
    # Draw a line
    draw.line([(50, 100), (750, 100)], fill='black', width=2)
    
    # Product 1
    y_pos = 130
    draw.rectangle([(50, y_pos), (750, y_pos + 180)], outline='gray', width=1)
    draw.text((70, y_pos + 20), "Wireless Headphones Pro", fill='black', font=header_font)
    draw.text((70, y_pos + 50), "SKU: WHP-001", fill='gray', font=text_font)
    draw.text((70, y_pos + 75), "• Noise cancellation technology", fill='black', font=text_font)
    draw.text((70, y_pos + 95), "• 30-hour battery life", fill='black', font=text_font)
    draw.text((70, y_pos + 115), "• Bluetooth 5.0 connectivity", fill='black', font=text_font)
    draw.text((70, y_pos + 135), "• Premium leather headband", fill='black', font=text_font)
    draw.text((580, y_pos + 40), "$199.99", fill='red', font=price_font)
    draw.text((580, y_pos + 65), "In Stock: 150", fill='green', font=text_font)
    
    # Product 2
    y_pos = 330
    draw.rectangle([(50, y_pos), (750, y_pos + 180)], outline='gray', width=1)
    draw.text((70, y_pos + 20), "Smart Watch Series X", fill='black', font=header_font)
    draw.text((70, y_pos + 50), "SKU: SWX-002", fill='gray', font=text_font)
    draw.text((70, y_pos + 75), "• Heart rate & fitness tracking", fill='black', font=text_font)
    draw.text((70, y_pos + 95), "• GPS navigation", fill='black', font=text_font)
    draw.text((70, y_pos + 115), "• Waterproof design (IP68)", fill='black', font=text_font)
    draw.text((70, y_pos + 135), "• 5-day battery life", fill='black', font=text_font)
    draw.text((580, y_pos + 40), "$299.99", fill='red', font=price_font)
    draw.text((580, y_pos + 65), "In Stock: 75", fill='green', font=text_font)
    
    # Product 3
    y_pos = 530
    draw.rectangle([(50, y_pos), (750, y_pos + 180)], outline='gray', width=1)
    draw.text((70, y_pos + 20), "Laptop Stand Aluminum", fill='black', font=header_font)
    draw.text((70, y_pos + 50), "SKU: LSA-003", fill='gray', font=text_font)
    draw.text((70, y_pos + 75), "• Adjustable height (6 positions)", fill='black', font=text_font)
    draw.text((70, y_pos + 95), "• Cooling vents for airflow", fill='black', font=text_font)
    draw.text((70, y_pos + 115), "• Universal laptop compatibility", fill='black', font=text_font)
    draw.text((70, y_pos + 135), "• Premium aluminum construction", fill='black', font=text_font)
    draw.text((580, y_pos + 40), "$49.99", fill='red', font=price_font)
    draw.text((580, y_pos + 65), "In Stock: 200", fill='green', font=text_font)
    
    # Product 4
    y_pos = 730
    draw.rectangle([(50, y_pos), (750, y_pos + 180)], outline='gray', width=1)
    draw.text((70, y_pos + 20), "USB-C Charging Hub", fill='black', font=header_font)
    draw.text((70, y_pos + 50), "SKU: UCH-004", fill='gray', font=text_font)
    draw.text((70, y_pos + 75), "• 6-port fast charging", fill='black', font=text_font)
    draw.text((70, y_pos + 95), "• 100W power delivery", fill='black', font=text_font)
    draw.text((70, y_pos + 115), "• Compact travel design", fill='black', font=text_font)
    draw.text((70, y_pos + 135), "• LED power indicators", fill='black', font=text_font)
    draw.text((580, y_pos + 40), "$79.99", fill='red', font=price_font)
    draw.text((580, y_pos + 65), "In Stock: 120", fill='green', font=text_font)
    
    # Footer
    draw.line([(50, 930), (750, 930)], fill='black', width=1)
    draw.text((50, 950), "Contact: sales@techsolutions.com | Phone: (555) 123-4567", fill='gray', font=text_font)
    
    # Save image
    data_dir = Path(__file__).parent / "data"
    image_path = data_dir / "product_catalog.png"
    image.save(image_path)
    print(f"Created product catalog image: {image_path}")
    return image_path

def create_business_card_image():
    """Create a sample business card image for text extraction."""
    # Create image
    width, height = 600, 350
    image = Image.new('RGB', (width, height), '#f0f0f0')
    draw = ImageDraw.Draw(image)
    
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        name_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        name_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    
    # Background
    draw.rectangle([(0, 0), (width, height)], fill='white', outline='#cccccc', width=2)
    
    # Company logo area (simple rectangle)
    draw.rectangle([(30, 30), (120, 80)], fill='#4a90e2', outline='#4a90e2')
    draw.text((35, 45), "LOGO", fill='white', font=name_font)
    
    # Company name
    draw.text((140, 35), "TechSolutions Inc.", fill='#2c3e50', font=title_font)
    draw.text((140, 65), "Digital Innovation Partners", fill='#7f8c8d', font=text_font)
    
    # Contact person
    draw.text((30, 130), "Sarah Johnson", fill='#2c3e50', font=name_font)
    draw.text((30, 155), "Senior Solutions Architect", fill='#34495e', font=text_font)
    
    # Contact details
    draw.text((30, 200), "📧 sarah.johnson@techsolutions.com", fill='#2c3e50', font=text_font)
    draw.text((30, 225), "📱 +1 (555) 987-6543", fill='#2c3e50', font=text_font)
    draw.text((30, 250), "🌐 www.techsolutions.com", fill='#2c3e50', font=text_font)
    draw.text((30, 275), "📍 456 Innovation Drive, Austin, TX 78701", fill='#2c3e50', font=text_font)
    
    # Right side - services
    draw.text((350, 130), "Services:", fill='#2c3e50', font=name_font)
    draw.text((350, 160), "• Cloud Architecture", fill='#34495e', font=text_font)
    draw.text((350, 180), "• AI/ML Solutions", fill='#34495e', font=text_font)
    draw.text((350, 200), "• Digital Transformation", fill='#34495e', font=text_font)
    draw.text((350, 220), "• 24/7 Technical Support", fill='#34495e', font=text_font)
    
    # Bottom accent
    draw.rectangle([(0, height-20), (width, height)], fill='#4a90e2')
    
    # Save image
    data_dir = Path(__file__).parent / "data"
    image_path = data_dir / "business_card.png"
    image.save(image_path)
    print(f"Created business card image: {image_path}")
    return image_path

def main():
    """Generate all sample files."""
    print("Generating sample files for PyRTex examples...")
    
    # Ensure data directory exists
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Generate files
    pdf_path = create_sample_pdf()
    catalog_image = create_product_catalog_image()
    card_image = create_business_card_image()
    
    print(f"\nGenerated sample files:")
    print(f"  📄 PDF Invoice: {pdf_path}")
    print(f"  🖼️  Product Catalog: {catalog_image}")
    print(f"  💳 Business Card: {card_image}")
    print(f"\nFiles are ready for use in PyRTex examples!")

if __name__ == "__main__":
    main()
