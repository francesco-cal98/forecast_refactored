"""Sample SPARQL responses and test data for regulatory tests."""
from __future__ import annotations

# Sample SPARQL JSON response with multiple documents
SPARQL_RESPONSE_SAMPLE = {
    "head": {"vars": ["work", "celex", "title", "abstract", "date", "rtype", "eurovoc", "lang"]},
    "results": {
        "bindings": [
            {
                "work": {"type": "uri", "value": "http://publications.europa.eu/resource/cellar/abc123"},
                "celex": {
                    "type": "uri",
                    "value": "http://publications.europa.eu/resource/celex/32019R1020",
                },
                "title": {"type": "literal", "value": "Regulation on Market Surveillance"},
                "abstract": {"type": "literal", "value": "This regulation establishes rules..."},
                "date": {"type": "typed-literal", "value": "2019-06-20"},
                "rtype": {"type": "uri", "value": "http://publications.europa.eu/resource/type/REG"},
                "eurovoc": {"type": "uri", "value": "http://eurovoc.europa.eu/4512"},
                "lang": {"type": "literal", "value": "ENG"},
            },
            {
                "work": {"type": "uri", "value": "http://publications.europa.eu/resource/cellar/def456"},
                "celex": {
                    "type": "uri",
                    "value": "http://publications.europa.eu/resource/celex/32020L0262",
                },
                "title": {"type": "literal", "value": "  Directive on Tax Reporting  "},
                "abstract": {"type": "literal", "value": None},
                "date": {"type": "typed-literal", "value": "2020-02-18"},
                "rtype": {"type": "uri", "value": "http://publications.europa.eu/resource/type/DIR"},
                "eurovoc": {"type": "uri", "value": "http://eurovoc.europa.eu/1234"},
                "lang": {"type": "literal", "value": "ENG"},
            },
            {
                # Document with same work but different eurovoc (should be grouped)
                "work": {"type": "uri", "value": "http://publications.europa.eu/resource/cellar/abc123"},
                "celex": {
                    "type": "uri",
                    "value": "http://publications.europa.eu/resource/celex/32019R1020",
                },
                "title": {"type": "literal", "value": "Regulation on Market Surveillance"},
                "abstract": {"type": "literal", "value": "This regulation establishes rules..."},
                "date": {"type": "typed-literal", "value": "2019-06-20"},
                "rtype": {"type": "uri", "value": "http://publications.europa.eu/resource/type/REG"},
                "eurovoc": {"type": "uri", "value": "http://eurovoc.europa.eu/9876"},
                "lang": {"type": "literal", "value": "ENG"},
            },
        ]
    },
}

# Empty SPARQL response
SPARQL_RESPONSE_EMPTY = {"head": {"vars": ["work", "celex", "title", "abstract", "date", "rtype", "eurovoc", "lang"]}, "results": {"bindings": []}}

# Response with NULL fields
SPARQL_RESPONSE_WITH_NULLS = {
    "head": {"vars": ["work", "celex", "title", "abstract", "date", "rtype", "eurovoc", "lang"]},
    "results": {
        "bindings": [
            {
                "work": {"type": "uri", "value": "http://publications.europa.eu/resource/cellar/xyz789"},
                # No celex field
                "title": {"type": "literal", "value": ""},  # Empty title
                # No abstract
                "date": {"type": "typed-literal", "value": "2021-03-15"},
                # No rtype
                # No eurovoc
                "lang": {"type": "literal", "value": "ENG"},
            }
        ]
    },
}

# Response with multiple languages (same work)
SPARQL_RESPONSE_MULTILANG = {
    "head": {"vars": ["work", "celex", "title", "abstract", "date", "rtype", "eurovoc", "lang"]},
    "results": {
        "bindings": [
            {
                "work": {"type": "uri", "value": "http://publications.europa.eu/resource/cellar/multi123"},
                "celex": {
                    "type": "uri",
                    "value": "http://publications.europa.eu/resource/celex/32021R0001",
                },
                "title": {"type": "literal", "value": "English Title"},
                "abstract": {"type": "literal", "value": "English abstract"},
                "date": {"type": "typed-literal", "value": "2021-01-01"},
                "rtype": {"type": "uri", "value": "http://publications.europa.eu/resource/type/REG"},
                "eurovoc": {"type": "uri", "value": "http://eurovoc.europa.eu/1111"},
                "lang": {"type": "literal", "value": "ENG"},
            },
            {
                "work": {"type": "uri", "value": "http://publications.europa.eu/resource/cellar/multi123"},
                "celex": {
                    "type": "uri",
                    "value": "http://publications.europa.eu/resource/celex/32021R0001",
                },
                "title": {"type": "literal", "value": "Titre français"},
                "abstract": {"type": "literal", "value": "Résumé français"},
                "date": {"type": "typed-literal", "value": "2021-01-01"},
                "rtype": {"type": "uri", "value": "http://publications.europa.eu/resource/type/REG"},
                "eurovoc": {"type": "uri", "value": "http://eurovoc.europa.eu/1111"},
                "lang": {"type": "literal", "value": "FRA"},
            },
        ]
    },
}

# Sample EUR-Lex HTML content
EURLEX_HTML_SAMPLE = """
<!DOCTYPE html>
<html>
<head><title>EUR-Lex - 32019R1020 - EN</title></head>
<body>
<div id="text">
<div class="eli-main-title">
<p class="doc-ti">Regulation (EU) 2019/1020</p>
</div>
<div class="ti-section-1">
<p class="sti-art">Article 1</p>
<p class="normal">This Regulation establishes rules for market surveillance and compliance of products.</p>
</div>
<div class="ti-section-1">
<p class="sti-art">Article 2</p>
<p class="normal">For the purposes of this Regulation, the following definitions apply:</p>
<p class="normal">(a) 'product' means any substance, preparation or good...</p>
</div>
</div>
</body>
</html>
"""

# Minimal HTML with no content div
EURLEX_HTML_MINIMAL = """
<!DOCTYPE html>
<html>
<head><title>EUR-Lex</title></head>
<body>
<p>Some text but no main content container</p>
</body>
</html>
"""

# HTML with French content
EURLEX_HTML_FRENCH = """
<!DOCTYPE html>
<html>
<head><title>EUR-Lex - 32019R1020 - FR</title></head>
<body>
<div id="texte">
<p>Le présent règlement établit des règles de surveillance du marché.</p>
<p>Article 1: Objet et champ d'application</p>
</div>
</body>
</html>
"""

# Sample expected normalized DataFrame structure
EXPECTED_NORMALIZED_COLUMNS = [
    "work_uri",
    "celex",
    "title",
    "abstract",
    "date_document",
    "resource_type",
    "eurovoc",
    "link_html",
    "link_pdf",
    "language",
]

# Sample test CELEX codes
TEST_CELEX_CODES = [
    ("http://publications.europa.eu/resource/celex/32019R1020", "32019R1020"),
    ("http://publications.europa.eu/resource/celex/32020L0262", "32020L0262"),
    ("http://publications.europa.eu/resource/celex/52021PC0001", "52021PC0001"),
    ("http://other.url.com/not-celex", None),
    (None, None),
    ("", None),
]

# Sample language mappings
TEST_LANGUAGE_MAPPINGS = [
    ("ENG", "EN"),
    ("FRA", "FR"),
    ("DEU", "DE"),
    ("ITA", "IT"),
    ("SPA", "ES"),
    ("POL", "PL"),
    ("BUL", "BG"),
    ("UNKNOWN", "EN"),  # fallback
    (None, "EN"),  # fallback
]

# Sample EuroVoc URIs for testing
TEST_EUROVOC_URIS = [
    "http://eurovoc.europa.eu/4512",
    "http://eurovoc.europa.eu/1234",
    "http://eurovoc.europa.eu/5678",
]

# Sample query terms
TEST_QUERY_TERMS = [
    "artificial intelligence",
    "climate change",
    "data protection",
]
