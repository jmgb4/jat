import pytest
from app.utils import enforce_experience_bullets


def test_enforce_experience_bullets_converts_plain_job_lines():
    text = """**Summary**
Short summary paragraph.

**Technical Expertise**
*Category:* item1, item2

**Senior Analyst**
Example Corp | 2020 - 2024
Built automation for vulnerability triage.
Reduced remediation time significantly.

**Education**
Some Degree | University
"""
    out = enforce_experience_bullets(text)
    assert "* Built automation for vulnerability triage." in out
    assert "* Reduced remediation time significantly." in out
    assert "Example Corp | 2020 - 2024" in out
    assert "**Education**" in out
    # Education content must NOT become a bullet
    assert "* Some Degree" not in out


def test_heading_with_colon_inside_bold_is_treated_as_section():
    """Headings like **Technical Expertise:** must not be treated as role titles."""
    text = """**Technical Expertise:**
*Security:* Nessus, Qualys
*Automation:* Python, Cribl

**Senior Analyst**
Acme Corp | 2022 - 2024
Built thing one.

**Education:**
B.S. Computer Science | State University
"""
    out = enforce_experience_bullets(text)
    # Technical Expertise items must NOT be bulleted (they are italic-formatted skills)
    assert "* Security" not in out
    assert "* Automation" not in out
    assert "*Security:* Nessus, Qualys" in out
    assert "*Automation:* Python, Cribl" in out
    # Job achievement IS bulleted
    assert "* Built thing one." in out
    # Education content must NOT be bulleted
    assert "* B.S. Computer Science" not in out
    assert "B.S. Computer Science | State University" in out


def test_education_with_colon_inside_bold_is_not_bulleted():
    """**Education:** heading must be detected as a section, not a role title."""
    text = """**Senior Analyst**
Acme Corp | 2022 - 2024
Built the product.

**Education:**
Bachelor of Science in Computer Science | Some University
"""
    out = enforce_experience_bullets(text)
    assert "* Built the product." in out
    assert "**Education:**" in out
    assert "* Bachelor of Science" not in out
    assert "Bachelor of Science in Computer Science | Some University" in out


def test_multiple_jobs_are_not_merged():
    """Each job block must be independently detected; job 2 and 3 must have their own headers."""
    text = """**Senior Analyst**
Company A | 2022 - 2024
Achievement A1.
Achievement A2.

**Junior Analyst**
Company B | 2020 - 2022
Achievement B1.
Achievement B2.

**Education**
Some Degree
"""
    out = enforce_experience_bullets(text)
    assert "* Achievement A1." in out
    assert "* Achievement A2." in out
    assert "**Junior Analyst**" in out
    assert "Company B | 2020 - 2022" in out
    assert "* Achievement B1." in out
    assert "* Achievement B2." in out
    assert "**Education**" in out
    # Education line must not be bulleted
    assert "* Some Degree" not in out


def test_italic_technical_expertise_items_preserved():
    """*Category:* items in Technical Expertise section must never be converted to bullets."""
    text = """**Technical Expertise**
*Vulnerability Lifecycle & Compliance:* FedRAMP, PCI DSS, NIST 800-53
*Automation & Tooling:* Python, Cribl, Redis
*Security Operations:* Bug bounty triage, incident response

**Software Engineer**
WidgetCo | 2019 - 2021
Shipped a large feature.
"""
    out = enforce_experience_bullets(text)
    # Italic lines must remain unchanged
    assert "*Vulnerability Lifecycle & Compliance:* FedRAMP, PCI DSS, NIST 800-53" in out
    assert "*Automation & Tooling:* Python, Cribl, Redis" in out
    assert "*Security Operations:* Bug bounty triage, incident response" in out
    # Achievement must be bulleted
    assert "* Shipped a large feature." in out


def test_existing_bullets_are_preserved():
    """If job lines already have bullet markers they must be normalized, not doubled."""
    text = """**Software Engineer**
WidgetCo | 2021 - 2023
* Already a bullet.
- Dash bullet.
• Unicode bullet.
"""
    out = enforce_experience_bullets(text)
    assert out.count("* Already a bullet.") == 1
    assert "* Dash bullet." in out
    assert "* Unicode bullet." in out


def test_no_bullets_on_empty_or_whitespace_text():
    assert enforce_experience_bullets("") == ""
    assert enforce_experience_bullets("   ") == ""
