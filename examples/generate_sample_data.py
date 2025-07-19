#!/usr/bin/env python3
"""
Generate sample PDF and image files for PyRTex examples.
This script creates realistic sample data to demonstrate PDF and image
processing capabilities.
"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_sample_pdf():
    """Create a sample PDF with invoice data using reportlab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        print("reportlab not installed. Installing...")
        os.system("pip install reportlab")
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

    # Create PDF file path
    data_dir = Path(__file__).parent / "data"
    pdf_path = data_dir / "sample_invoice.pdf"

    # Create the PDF document
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center alignment
    )
    story.append(Paragraph("INVOICE", title_style))
    story.append(Spacer(1, 20))

    # Company info
    company_style = ParagraphStyle(
        "Company", parent=styles["Normal"], fontSize=12, spaceAfter=6
    )
    story.append(Paragraph("<b>Tech Solutions Inc.</b>", company_style))
    story.append(Paragraph("123 Business Ave", company_style))
    story.append(Paragraph("San Francisco, CA 94102", company_style))
    story.append(Paragraph("Phone: (555) 123-4567", company_style))
    story.append(Spacer(1, 20))

    # Invoice details
    invoice_data = [
        ["Invoice #:", "INV-2025-001"],
        ["Date:", "July 18, 2025"],
        ["Due Date:", "August 17, 2025"],
        ["Customer ID:", "CUST-001"],
    ]

    invoice_table = Table(invoice_data, colWidths=[1.5 * inch, 2 * inch])
    invoice_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(invoice_table)
    story.append(Spacer(1, 20))

    # Bill to section
    story.append(Paragraph("<b>Bill To:</b>", styles["Normal"]))
    story.append(Paragraph("John Doe", styles["Normal"]))
    story.append(Paragraph("456 Customer St", styles["Normal"]))
    story.append(Paragraph("Austin, TX 78701", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Items table
    items_data = [
        ["Description", "Quantity", "Unit Price", "Total"],
        ["Software License (Annual)", "1", "$1,200.00", "$1,200.00"],
        ["Premium Support Package", "1", "$300.00", "$300.00"],
        ["Training Session (4 hours)", "1", "$500.00", "$500.00"],
        ["Setup and Configuration", "1", "$150.00", "$150.00"],
    ]

    items_table = Table(
        items_data, colWidths=[3 * inch, 0.8 * inch, 1 * inch, 1 * inch]
    )
    items_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(items_table)
    story.append(Spacer(1, 20))

    # Totals
    totals_data = [
        ["Subtotal:", "$2,150.00"],
        ["Tax (8.25%):", "$177.38"],
        ["Total:", "$2,327.38"],
    ]

    totals_table = Table(totals_data, colWidths=[1.5 * inch, 1 * inch])
    totals_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, -1), (-1, -1), 2, colors.black),
            ]
        )
    )

    # Right-align the totals table
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            '<para align="right"><b>Payment Terms: Net 30 days</b></para>',
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    # Create a container for right-aligned table
    totals_container = Table([[totals_table]], colWidths=[6.5 * inch])
    totals_container.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (0, 0), "RIGHT"),
            ]
        )
    )
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
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        # Try to use a better font
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32
        )
        header_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
        text_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
        price_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except (OSError, IOError):
        # Fallback to default font
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        price_font = ImageFont.load_default()

    # Title
    draw.text((50, 30), "PRODUCT CATALOG", fill="black", font=title_font)
    draw.text((50, 70), "Electronics & Accessories", fill="gray", font=header_font)

    # Draw a line
    draw.line([(50, 100), (750, 100)], fill="black", width=2)

    # Product 1
    y_pos = 130
    draw.rectangle([(50, y_pos), (750, y_pos + 180)], outline="gray", width=1)
    draw.text(
        (70, y_pos + 20), "Wireless Headphones Pro", fill="black", font=header_font
    )
    draw.text((70, y_pos + 50), "SKU: WHP-001", fill="gray", font=text_font)
    draw.text(
        (70, y_pos + 75),
        "• Noise cancellation technology",
        fill="black",
        font=text_font,
    )
    draw.text((70, y_pos + 95), "• 30-hour battery life", fill="black", font=text_font)
    draw.text(
        (70, y_pos + 115), "• Bluetooth 5.0 connectivity", fill="black", font=text_font
    )
    draw.text(
        (70, y_pos + 135), "• Premium leather headband", fill="black", font=text_font
    )
    draw.text((580, y_pos + 40), "$199.99", fill="red", font=price_font)
    draw.text((580, y_pos + 65), "In Stock: 150", fill="green", font=text_font)

    # Product 2
    y_pos = 330
    draw.rectangle([(50, y_pos), (750, y_pos + 180)], outline="gray", width=1)
    draw.text((70, y_pos + 20), "Smart Watch Series X", fill="black", font=header_font)
    draw.text((70, y_pos + 50), "SKU: SWX-002", fill="gray", font=text_font)
    draw.text(
        (70, y_pos + 75),
        "• Heart rate & fitness tracking",
        fill="black",
        font=text_font,
    )
    draw.text((70, y_pos + 95), "• GPS navigation", fill="black", font=text_font)
    draw.text(
        (70, y_pos + 115), "• Waterproof design (IP68)", fill="black", font=text_font
    )
    draw.text((70, y_pos + 135), "• 5-day battery life", fill="black", font=text_font)
    draw.text((580, y_pos + 40), "$299.99", fill="red", font=price_font)
    draw.text((580, y_pos + 65), "In Stock: 75", fill="green", font=text_font)

    # Product 3
    y_pos = 530
    draw.rectangle([(50, y_pos), (750, y_pos + 180)], outline="gray", width=1)
    draw.text((70, y_pos + 20), "Laptop Stand Aluminum", fill="black", font=header_font)
    draw.text((70, y_pos + 50), "SKU: LSA-003", fill="gray", font=text_font)
    draw.text(
        (70, y_pos + 75),
        "• Adjustable height (6 positions)",
        fill="black",
        font=text_font,
    )
    draw.text(
        (70, y_pos + 95), "• Cooling vents for airflow", fill="black", font=text_font
    )
    draw.text(
        (70, y_pos + 115),
        "• Universal laptop compatibility",
        fill="black",
        font=text_font,
    )
    draw.text(
        (70, y_pos + 135),
        "• Premium aluminum construction",
        fill="black",
        font=text_font,
    )
    draw.text((580, y_pos + 40), "$49.99", fill="red", font=price_font)
    draw.text((580, y_pos + 65), "In Stock: 200", fill="green", font=text_font)

    # Product 4
    y_pos = 730
    draw.rectangle([(50, y_pos), (750, y_pos + 180)], outline="gray", width=1)
    draw.text((70, y_pos + 20), "USB-C Charging Hub", fill="black", font=header_font)
    draw.text((70, y_pos + 50), "SKU: UCH-004", fill="gray", font=text_font)
    draw.text((70, y_pos + 75), "• 6-port fast charging", fill="black", font=text_font)
    draw.text((70, y_pos + 95), "• 100W power delivery", fill="black", font=text_font)
    draw.text(
        (70, y_pos + 115), "• Compact travel design", fill="black", font=text_font
    )
    draw.text((70, y_pos + 135), "• LED power indicators", fill="black", font=text_font)
    draw.text((580, y_pos + 40), "$79.99", fill="red", font=price_font)
    draw.text((580, y_pos + 65), "In Stock: 120", fill="green", font=text_font)

    # Footer
    draw.line([(50, 930), (750, 930)], fill="black", width=1)
    draw.text(
        (50, 950),
        "Contact: sales@techsolutions.com | Phone: (555) 123-4567",
        fill="gray",
        font=text_font,
    )

    # Save image
    data_dir = Path(__file__).parent / "data"
    image_path = data_dir / "product_catalog.png"
    image.save(image_path)
    print(f"Created product catalog image: {image_path}")
    return image_path


def create_business_card_1():
    """Create a corporate-style business card (horizontal layout)."""
    # Create image
    width, height = 600, 350
    image = Image.new("RGB", (width, height), "#f0f0f0")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
        name_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
        text_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        name_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    # Background
    draw.rectangle([(0, 0), (width, height)], fill="white", outline="#cccccc", width=2)

    # Company logo area (simple rectangle)
    draw.rectangle([(30, 30), (120, 80)], fill="#4a90e2", outline="#4a90e2")
    draw.text((35, 45), "LOGO", fill="white", font=name_font)

    # Company name
    draw.text((140, 35), "TechSolutions Inc.", fill="#2c3e50", font=title_font)
    draw.text((140, 65), "Digital Innovation Partners", fill="#7f8c8d", font=text_font)

    # Contact person
    draw.text((30, 130), "Sarah Johnson", fill="#2c3e50", font=name_font)
    draw.text((30, 155), "Senior Solutions Architect", fill="#34495e", font=text_font)

    # Contact details
    draw.text(
        (30, 200), "📧 sarah.johnson@techsolutions.com", fill="#2c3e50", font=text_font
    )
    draw.text((30, 225), "📱 +1 (555) 987-6543", fill="#2c3e50", font=text_font)
    draw.text((30, 250), "🌐 www.techsolutions.com", fill="#2c3e50", font=text_font)
    draw.text(
        (30, 275),
        "📍 456 Innovation Drive, Austin, TX 78701",
        fill="#2c3e50",
        font=text_font,
    )

    # Right side - services
    draw.text((350, 130), "Services:", fill="#2c3e50", font=name_font)
    draw.text((350, 160), "• Cloud Architecture", fill="#34495e", font=text_font)
    draw.text((350, 180), "• AI/ML Solutions", fill="#34495e", font=text_font)
    draw.text((350, 200), "• Digital Transformation", fill="#34495e", font=text_font)
    draw.text((350, 220), "• 24/7 Technical Support", fill="#34495e", font=text_font)

    # Bottom accent
    draw.rectangle([(0, height - 20), (width, height)], fill="#4a90e2")

    return image


def create_business_card_2():
    """Create a creative agency card (vertical layout, dark theme)."""
    # Create image (vertical orientation)
    width, height = 350, 550
    image = Image.new("RGB", (width, height), "#1a1a1a")
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
        name_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
        text_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        name_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    # Background gradient effect
    draw.rectangle([(0, 0), (width, height)], fill="#1a1a1a")

    # Top accent strip
    draw.rectangle([(0, 0), (width, 40)], fill="#e74c3c")

    # Company logo/icon (circle)
    draw.ellipse([(140, 60), (210, 130)], fill="#e74c3c", outline="#c0392b", width=3)
    draw.text((160, 85), "CA", fill="white", font=title_font)

    # Company name (centered)
    draw.text((70, 150), "Creative Agency", fill="white", font=title_font)
    draw.text((110, 175), "Design Studio", fill="#bdc3c7", font=text_font)

    # Divider line
    draw.line([(50, 220), (300, 220)], fill="#e74c3c", width=2)

    # Contact person (centered)
    draw.text((110, 250), "Marcus Chen", fill="white", font=name_font)
    draw.text((90, 275), "Creative Director", fill="#95a5a6", font=text_font)

    # Contact details (centered)
    draw.text((60, 320), "marcus@creativeagency.co", fill="#ecf0f1", font=text_font)
    draw.text((120, 345), "(555) 234-5678", fill="#ecf0f1", font=text_font)
    draw.text((80, 370), "creativeagency.co", fill="#ecf0f1", font=text_font)

    # Address
    draw.text((80, 395), "789 Design District", fill="#ecf0f1", font=text_font)
    draw.text((100, 415), "Brooklyn, NY 11201", fill="#ecf0f1", font=text_font)

    # Services
    draw.text((130, 460), "Specialties:", fill="#e74c3c", font=name_font)
    draw.text((90, 485), "• Brand Identity", fill="#bdc3c7", font=text_font)
    draw.text((90, 505), "• Web Design", fill="#bdc3c7", font=text_font)
    draw.text((90, 525), "• Print Design", fill="#bdc3c7", font=text_font)

    return image


def create_business_card_3():
    """Create a minimal consultant card (square format, minimal design)."""
    # Create image (square format)
    width, height = 450, 450
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        name_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
        )
        text_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except (OSError, IOError):
        name_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Clean white background with subtle border
    draw.rectangle([(0, 0), (width, height)], fill="white", outline="#e0e0e0", width=1)

    # Minimal geometric accent (top left corner)
    draw.rectangle([(0, 0), (80, 80)], fill="#2ecc71")

    # Large name (centered top)
    draw.text((120, 120), "Dr. Elena Rodriguez", fill="#2c3e50", font=name_font)

    # Title (centered)
    draw.text(
        (110, 160), "Business Strategy Consultant", fill="#7f8c8d", font=text_font
    )

    # Subtle divider
    draw.line([(100, 200), (350, 200)], fill="#bdc3c7", width=1)

    # Contact info (clean layout)
    draw.text(
        (100, 230), "elena.rodriguez@strategy-pro.com", fill="#34495e", font=text_font
    )
    draw.text((100, 255), "+1 (555) 456-7890", fill="#34495e", font=text_font)
    draw.text((100, 280), "www.strategy-pro.com", fill="#34495e", font=text_font)

    # Address
    draw.text(
        (100, 320), "321 Business Plaza, Suite 1200", fill="#7f8c8d", font=small_font
    )
    draw.text((100, 340), "San Francisco, CA 94105", fill="#7f8c8d", font=small_font)

    # Services (bottom right)
    draw.text((250, 380), "Strategic Planning", fill="#2ecc71", font=small_font)
    draw.text((250, 395), "Operations Optimization", fill="#2ecc71", font=small_font)
    draw.text((250, 410), "Market Analysis", fill="#2ecc71", font=small_font)

    return image


def create_business_cards():
    """Create three different business card designs and save them."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Create and save each card
    cards = [
        (
            create_business_card_1(),
            "business_card_1.png",
            "Corporate TechSolutions card",
        ),
        (create_business_card_2(), "business_card_2.png", "Creative Agency card"),
        (create_business_card_3(), "business_card_3.png", "Consultant card"),
    ]

    for card_image, filename, description in cards:
        image_path = data_dir / filename
        card_image.save(image_path)
        print(f"✓ Created {description}: {filename}")

    return [card[1] for card in cards]  # Return filenames


# Keep legacy function for backward compatibility
def create_business_card_image():
    """Legacy function - redirects to first card design."""
    return create_business_card_1()


def create_sample_yaml_files():
    """Create sample YAML files for real estate properties."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Luxury Condo YAML
    luxury_condo = """# Luxury Downtown Condo Listing
property_id: "CONDO-2025-001"
listing_agent: "Sarah Mitchell"
agency: "Elite Properties SF"
listing_date: "2025-07-15"

property_details:
  address: "1200 Mission Street, Unit 4502"
  city: "San Francisco"
  state: "California"
  zip_code: "94103"
  property_type: "Condominium"
  year_built: 2019
  
size_and_layout:
  bedrooms: 3
  bathrooms: 2.5
  square_feet: 1850
  floor_number: 45
  balcony: true
  view: "City and Bay View"
  
pricing:
  list_price: 2150000
  price_per_sqft: 1162
  hoa_fees_monthly: 850
  property_taxes_annual: 25800
  
features:
  - "Floor-to-ceiling windows"
  - "Hardwood floors throughout"
  - "Gourmet kitchen with quartz countertops"
  - "In-unit washer/dryer"
  - "Private parking space"
  - "24/7 concierge service"
  - "Rooftop deck access"
  - "Fitness center"

building_amenities:
  doorman: true
  elevator: true
  gym: true
  pool: false
  parking_spaces: 1
  pet_friendly: true
  
market_info:
  days_on_market: 12
  price_reduced: false
  multiple_offers: true
  estimated_closing: "2025-08-30"
"""
    
    # Suburban House YAML  
    suburban_house = """# Family Home in Palo Alto
property_id: "HOUSE-2025-078"
listing_agent: "Michael Chen"
agency: "Peninsula Real Estate Group"
listing_date: "2025-07-10"

property_details:
  address: "567 Elm Street"
  city: "Palo Alto"
  state: "California"
  zip_code: "94301"
  property_type: "Single Family Home"
  year_built: 1987
  last_renovated: 2021
  
size_and_layout:
  bedrooms: 4
  bathrooms: 3
  square_feet: 2650
  lot_size_sqft: 7200
  garage_spaces: 2
  stories: 2
  
pricing:
  list_price: 3750000
  price_per_sqft: 1415
  property_taxes_annual: 45000
  estimated_monthly_utilities: 280
  
features:
  - "Updated chef's kitchen"
  - "Master suite with walk-in closet"
  - "Backyard with mature landscaping"
  - "Home office/study"
  - "Fireplace in living room"
  - "Central air conditioning"
  - "Solar panels"
  - "Two-car garage"

outdoor_space:
  backyard: true
  front_yard: true
  patio: true
  deck: false
  garden: true
  sprinkler_system: true
  
schools:
  elementary: "Duveneck Elementary (9/10)"
  middle: "Jane Lathrop Stanford Middle (9/10)"
  high: "Palo Alto High School (9/10)"
  
market_info:
  days_on_market: 8
  price_reduced: false
  open_house_scheduled: true
  estimated_closing: "2025-08-15"
"""
    
    condo_path = data_dir / "luxury_condo.yaml"
    house_path = data_dir / "suburban_house.yaml"
    
    with open(condo_path, 'w', encoding='utf-8') as f:
        f.write(luxury_condo)
    
    with open(house_path, 'w', encoding='utf-8') as f:
        f.write(suburban_house)
    
    print(f"✓ Created luxury condo listing: {condo_path.name}")
    print(f"✓ Created suburban house listing: {house_path.name}")
    
    return [condo_path.name, house_path.name]


def create_sample_json_files():
    """Create sample JSON files for real estate properties."""
    import json
    
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Commercial Office Building JSON
    office_building = {
        "listing_id": "COMM-2025-0445",
        "property_type": "Commercial Office Building",
        "listing_agent": {
            "name": "David Thompson",
            "company": "Bay Area Commercial Properties",
            "phone": "(415) 555-0199",
            "email": "dthompson@bacprop.com"
        },
        "property_address": {
            "street": "888 Howard Street",
            "city": "San Francisco", 
            "state": "CA",
            "zip": "94103",
            "neighborhood": "SOMA"
        },
        "building_details": {
            "year_built": 2015,
            "total_floors": 12,
            "total_square_feet": 145000,
            "available_square_feet": 8500,
            "parking_spaces": 75,
            "elevator_count": 3,
            "building_class": "A"
        },
        "available_units": [
            {
                "floor": 8,
                "suite": "800A",
                "square_feet": 3200,
                "rent_per_sqft_annual": 85,
                "lease_type": "Modified Gross",
                "available_date": "2025-09-01"
            },
            {
                "floor": 11,
                "suite": "1100B", 
                "square_feet": 5300,
                "rent_per_sqft_annual": 90,
                "lease_type": "Triple Net",
                "available_date": "2025-08-15"
            }
        ],
        "pricing": {
            "asking_rent_psf_annual": 87.50,
            "estimated_operating_expenses": 12.50,
            "property_taxes_psf": 8.75,
            "cad_charges_psf": 3.25
        },
        "amenities": [
            "24/7 security",
            "Fiber internet ready",
            "Conference facilities",
            "Rooftop terrace",
            "Bike storage",
            "Electric vehicle charging",
            "On-site cafeteria"
        ],
        "building_systems": {
            "hvac": "Central air conditioning",
            "electrical": "220V available",
            "internet": "Fiber optic backbone",
            "security": "Card access system",
            "fire_safety": "Sprinkler system throughout"
        },
        "market_metrics": {
            "days_on_market": 45,
            "vacancy_rate_building": 0.12,
            "vacancy_rate_submarket": 0.18,
            "recent_comparable_rent_psf": 82.00,
            "tenant_retention_rate": 0.89
        },
        "zoning_and_legal": {
            "zoning": "C-3-O (Downtown Office)",
            "permits_required": ["Business license", "Signage permit"],
            "restrictions": ["No manufacturing", "Ground floor retail preferred"],
            "seismic_rating": "Excellent"
        }
    }
    
    # Apartment Complex JSON
    apartment_complex = {
        "property_id": "APT-COMPLEX-2025-156",
        "property_name": "Sunset Gardens Apartments",
        "property_type": "Multi-Family Residential",
        "management_company": {
            "name": "Golden Gate Property Management",
            "contact_person": "Jennifer Rodriguez",
            "phone": "(415) 555-0287",
            "email": "jrodriguez@ggpm.com",
            "license_number": "CA-RE-001847293"
        },
        "location": {
            "address": "1425-1475 19th Avenue",
            "city": "San Francisco",
            "state": "California",
            "zip_code": "94122",
            "neighborhood": "Sunset District",
            "walk_score": 78,
            "transit_score": 65
        },
        "property_overview": {
            "year_built": 1995,
            "last_renovation": 2020,
            "total_units": 48,
            "stories": 4,
            "lot_size_sqft": 18500,
            "building_sqft": 52000,
            "parking_spaces": 38
        },
        "unit_mix": [
            {
                "unit_type": "Studio",
                "count": 8,
                "avg_sqft": 450,
                "rent_range": [2100, 2400],
                "currently_available": 1
            },
            {
                "unit_type": "1 Bedroom",
                "count": 24,
                "avg_sqft": 650,
                "rent_range": [2800, 3200],
                "currently_available": 3
            },
            {
                "unit_type": "2 Bedroom",
                "count": 16,
                "avg_sqft": 950,
                "rent_range": [3600, 4100],
                "currently_available": 2
            }
        ],
        "financial_performance": {
            "gross_scheduled_income_annual": 1680000,
            "current_occupancy_rate": 0.875,
            "average_rent_per_unit": 2917,
            "operating_expenses_annual": 485000,
            "net_operating_income": 985000,
            "cap_rate": 0.045
        },
        "property_features": [
            "Controlled access entry",
            "Laundry facility on each floor", 
            "Rooftop deck with city views",
            "Bicycle storage",
            "Package receiving service",
            "Courtyard garden",
            "Updated appliances",
            "Hardwood floors in select units"
        ],
        "nearby_amenities": {
            "transportation": [
                "N-Judah Muni line (0.3 miles)",
                "Multiple bus lines",
                "Ocean Beach (0.8 miles)"
            ],
            "shopping": [
                "Irving Street shopping district",
                "Stonestown Galleria (1.2 miles)",
                "Local grocery stores"
            ],
            "recreation": [
                "Golden Gate Park (0.5 miles)",
                "Stern Grove (0.7 miles)",
                "SF Zoo (1.5 miles)"
            ]
        },
        "investment_metrics": {
            "asking_price": 21900000,
            "price_per_unit": 456250,
            "price_per_sqft": 421,
            "estimated_annual_appreciation": 0.038,
            "property_tax_rate": 0.0118
        }
    }
    
    office_path = data_dir / "office_building.json"
    apartment_path = data_dir / "apartment_complex.json"
    
    with open(office_path, 'w', encoding='utf-8') as f:
        json.dump(office_building, f, indent=2)
    
    with open(apartment_path, 'w', encoding='utf-8') as f:
        json.dump(apartment_complex, f, indent=2)
    
    print(f"✓ Created office building listing: {office_path.name}")
    print(f"✓ Created apartment complex listing: {apartment_path.name}")
    
    return [office_path.name, apartment_path.name]


def create_minimal_test_files():
    """Create minimal test files for various MIME types supported by Gemini."""
    import io
    import wave
    
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    created_files = []
    
    # 1. Create a minimal PDF (using reportlab)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        
        pdf_path = data_dir / "test_minimal.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph("Minimal test PDF for PyRTex MIME type testing.", styles['Normal'])]
        doc.build(story)
        created_files.append(("test_minimal.pdf", "application/pdf"))
        print(f"✓ Created minimal PDF: {pdf_path.name}")
    except ImportError:
        print("⚠️  Skipping PDF creation (reportlab not available)")
    
    # 2. Create a minimal PNG image (1x1 pixel)
    try:
        from PIL import Image
        
        img_path = data_dir / "test_minimal.png"
        # Create a tiny 1x1 white pixel image
        img = Image.new('RGB', (1, 1), color='white')
        img.save(img_path)
        created_files.append(("test_minimal.png", "image/png"))
        print(f"✓ Created minimal PNG: {img_path.name}")
    except ImportError:
        print("⚠️  Skipping PNG creation (PIL not available)")
    
    # 3. Create a minimal JPEG image (1x1 pixel)
    try:
        from PIL import Image
        
        jpg_path = data_dir / "test_minimal.jpg"
        # Create a tiny 1x1 red pixel image
        img = Image.new('RGB', (1, 1), color='red')
        img.save(jpg_path, 'JPEG')
        created_files.append(("test_minimal.jpg", "image/jpeg"))
        print(f"✓ Created minimal JPEG: {jpg_path.name}")
    except ImportError:
        print("⚠️  Skipping JPEG creation (PIL not available)")
    
    # 4. Create a minimal WAV audio file (1 second of silence)
    wav_path = data_dir / "test_minimal.wav"
    try:
        # Create 1 second of silence at 8kHz, 16-bit mono (minimal size)
        sample_rate = 8000
        duration = 1  # 1 second
        frames = sample_rate * duration
        
        with wave.open(str(wav_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(sample_rate)
            
            # Write silence (all zeros)
            silence = b'\x00\x00' * frames
            wav_file.writeframes(silence)
        
        created_files.append(("test_minimal.wav", "audio/wav"))
        print(f"✓ Created minimal WAV: {wav_path.name}")
    except Exception as e:
        print(f"⚠️  Skipping WAV creation: {e}")
    
    # 5. Create a real 1-second MP4 video file
    mp4_path = data_dir / "test_minimal.mp4"
    try:
        import cv2
        import numpy as np
        
        # Create a 1-second video at 30fps with 100x100 resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(mp4_path), fourcc, 30.0, (100, 100))
        
        # Create 30 frames (1 second at 30fps)
        for frame_num in range(30):
            # Create a frame with changing colors
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Make the frame change color slightly each frame
            color_value = int(255 * frame_num / 30)
            frame[:, :] = [color_value, 100, 255 - color_value]  # BGR format
            video_writer.write(frame)
        
        video_writer.release()
        created_files.append(("test_minimal.mp4", "video/mp4"))
        print(f"✓ Created real 1-second MP4 video: {mp4_path.name}")
    except ImportError:
        print("⚠️  OpenCV not available, creating minimal MP4 placeholder...")
        try:
            # Fallback: Create a minimal valid MP4 structure if OpenCV is not available
            with open(mp4_path, 'wb') as f:
                # Write a more complete MP4 header structure
                # ftyp box
                f.write(b'\x00\x00\x00\x20')  # box size
                f.write(b'ftyp')  # box type
                f.write(b'mp42')  # major brand
                f.write(b'\x00\x00\x00\x00')  # minor version
                f.write(b'mp42isom')  # compatible brands
                
                # mdat box with minimal data
                f.write(b'\x00\x00\x00\x10')  # box size
                f.write(b'mdat')  # box type
                f.write(b'\x00' * 8)  # minimal data
            
            created_files.append(("test_minimal.mp4", "video/mp4"))
            print(f"✓ Created minimal MP4 placeholder: {mp4_path.name}")
        except Exception as e2:
            print(f"⚠️  Skipping MP4 creation: {e2}")
    except Exception as e:
        print(f"⚠️  Skipping MP4 creation: {e}")
    
    # 6. Create a WebP image (1x1 pixel)
    try:
        from PIL import Image
        
        webp_path = data_dir / "test_minimal.webp"
        # Create a tiny 1x1 blue pixel image
        img = Image.new('RGB', (1, 1), color='blue')
        img.save(webp_path, 'WEBP')
        created_files.append(("test_minimal.webp", "image/webp"))
        print(f"✓ Created minimal WebP: {webp_path.name}")
    except ImportError:
        print("⚠️  Skipping WebP creation (PIL WebP support not available)")
    except Exception as e:
        print(f"⚠️  Skipping WebP creation: {e}")
    
    return created_files


def main():
    """Generate all sample files."""
    print("Generating sample files for PyRTex examples...")

    # Ensure data directory exists
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Wipe the data directory
    for file in data_dir.glob("*"):
        try:
            file.unlink()
            print(f"Deleted existing file: {file.name}")
        except Exception as e:
            print(f"Error deleting {file.name}: {e}")

    # Generate files
    pdf_path = create_sample_pdf()
    catalog_image = create_product_catalog_image()
    card_filenames = create_business_cards()
    yaml_filenames = create_sample_yaml_files()
    json_filenames = create_sample_json_files()
    test_files = create_minimal_test_files()

    print("\nGenerated sample files:")
    print(f"  📄 PDF Invoice: {pdf_path}")
    print(f"  🖼️  Product Catalog: {catalog_image}")
    print("  💳 Business Cards:")
    for filename in card_filenames:
        print(f"      • {filename}")
    print("  ⚙️  YAML Real Estate Listings:")
    for filename in yaml_filenames:
        print(f"      • {filename}")
    print("  📊 JSON Real Estate Data:")
    for filename in json_filenames:
        print(f"      • {filename}")
    print("  🧪 Test Files (All MIME Types):")
    for filename, mime_type in test_files:
        print(f"      • {filename} ({mime_type})")
    print("\nFiles are ready for use in PyRTex examples!")


if __name__ == "__main__":
    main()
