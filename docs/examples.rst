Examples
========

This section provides detailed examples of using PyRTex for various text processing tasks.

Text Analysis
-------------

Extract key information from text documents::

    from pydantic import BaseModel
    from pyrtex import Job

    class TextAnalysis(BaseModel):
        summary: str
        key_points: list[str]
        word_count: int

    job = Job[TextAnalysis](
        model="gemini-2.0-flash-lite-001",
        output_schema=TextAnalysis,
        prompt_template="""
        Analyze this text: {{ content }}
        
        Provide:
        1. Brief summary
        2. Key points
        3. Word count
        """
    )

Product Parsing
---------------

Structure product information from catalogs::

    class Product(BaseModel):
        name: str
        price: float
        category: str
        
    class ProductCatalog(BaseModel):
        products: list[Product]
        total_count: int

    job = Job[ProductCatalog](
        model="gemini-2.0-flash-lite-001",
        output_schema=ProductCatalog,
        prompt_template="Parse products from: {{ catalog_text }}"
    )

Batch Processing
----------------

Process multiple documents efficiently::

    # Add multiple requests
    for doc_id, content in documents.items():
        job.add_request(doc_id, {"content": content})
    
    # Submit for processing
    job.submit()
    job.wait()
    
    # Collect results
    results = {}
    for result in job.results():
        results[result.request_key] = result.output

See the ``examples/`` directory for complete runnable examples.
