#!/usr/bin/env python3
"""
TruckPro Product Scraper MCP Server

This MCP server provides tools to scrape TruckPro product pages, enrich with vendor data,
and generate accurate product descriptions using AI agents.
"""

import os
import re
import json
import asyncio
from io import BytesIO
from typing import Dict, Any, List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
from serpapi import GoogleSearch

from crewai import Agent, Task, Crew, LLM
from langchain_openai import AzureChatOpenAI

# MCP imports
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# ============================
# ENV & CONFIG
# ============================
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise ValueError("❌ Missing Azure OpenAI credentials in .env")

# ============================
# BROWSER (SELENIUM) FOR JS PAGES
# ============================
def fetch_js_rendered(url: str):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        return driver.page_source, 200
    except Exception as e:
        return None, 500
    finally:
        driver.quit()

# ============================
# PARSE TRUCKPRO PRODUCT PAGE
# ============================
def parse_truckpro_product(url: str) -> Optional[Dict[str, Any]]:
    html, status = fetch_js_rendered(url)
    if status != 200 or not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    product_data: Dict[str, Any] = {}

    # JSON-LD extraction
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get("@type") == "Product":
                product_data["title"] = data.get("name")
                product_data["sku"] = data.get("sku")
                product_data["description"] = data.get("description")
        except Exception:
            pass

    # Fallbacks
    if not product_data.get("title"):
        h1 = soup.find("h1")
        if h1:
            product_data["title"] = h1.get_text(strip=True)

    if not product_data.get("sku"):
        sku_tag = soup.find(lambda tag: tag.name in ["span", "div"] and "SKU" in tag.get_text())
        if sku_tag:
            product_data["sku"] = sku_tag.get_text(strip=True)

    if not product_data.get("description"):
        desc_tag = soup.select_one(".product-description, .description, #product-description, .product-long-description")
        if desc_tag:
            product_data["description"] = desc_tag.get_text(" ", strip=True)

    # Specifications
    specs: Dict[str, str] = {}
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                key = cols[0].get_text(" ", strip=True)
                value = cols[1].get_text(" ", strip=True)
                if key and value:
                    specs[key] = value

    # Nearby "Specifications" section
    spec_header = soup.find(lambda tag: tag.name in ["h2", "h3", "strong", "b", "p"] and "Specifications" in tag.get_text())
    if spec_header:
        for sibling in spec_header.find_next_siblings():
            if sibling.name in ["ul", "ol"]:
                for li in sibling.find_all("li"):
                    txt = li.get_text(" ", strip=True)
                    if ":" in txt:
                        k, v = txt.split(":", 1)
                        specs[k.strip()] = v.strip()
            if sibling.name in ["p", "div"]:
                txt = sibling.get_text(" ", strip=True)
                if ":" in txt:
                    k, v = txt.split(":", 1)
                    specs[k.strip()] = v.strip()

    product_data["specifications"] = specs
    product_data["source_url"] = url
    return product_data if product_data else None

# ============================
# VENDOR SEARCH (SERPAPI)
# ============================
def fetch_vendor_data(product_title: str) -> Dict[str, Any]:
    if not SERPAPI_API_KEY or not product_title:
        return {"vendor_desc": "No vendor search performed", "vendor_snippets": [], "vendor_specs": {}, "vendor_sources": []}

    try:
        search = GoogleSearch({"q": product_title, "api_key": SERPAPI_API_KEY, "engine": "google"})
        results = search.get_dict()
    except Exception:
        return {"vendor_desc": "Vendor search failed", "vendor_snippets": [], "vendor_specs": {}, "vendor_sources": []}

    vendor_descs: List[str] = []
    vendor_sources: List[str] = []
    vendor_specs: Dict[str, str] = {}

    for res in results.get("organic_results", [])[:10]:
        snippet = res.get("snippet", "")
        link = res.get("link", "")
        if snippet:
            vendor_descs.append(snippet)
            for line in snippet.split(". "):
                if ":" in line:
                    k, v = line.split(":", 1)
                    vendor_specs[k.strip()] = v.strip()
        if link:
            vendor_sources.append(link)

        # Rich snippet
        if "rich_snippet" in res:
            snippet_data = res["rich_snippet"].get("top", {}).get("extensions", [])
            for ext in snippet_data:
                if ":" in ext:
                    k, v = ext.split(":", 1)
                    vendor_specs[k.strip()] = v.strip()

    # Knowledge graph
    if "knowledge_graph" in results:
        kg = results["knowledge_graph"]
        for k, v in kg.items():
            if isinstance(v, str) and len(v) < 100:
                vendor_specs[k] = v

    return {
        "vendor_desc": " ".join(vendor_descs) if vendor_descs else "No vendor descriptions found",
        "vendor_snippets": vendor_descs,
        "vendor_specs": vendor_specs,
        "vendor_sources": vendor_sources
    }

# ============================
# LLMs AND AGENTS SETUP
# ============================
crewai_llm = LLM(
    model=f"azure/{AZURE_OPENAI_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# Agents
scraper_agent = Agent(
    role="TruckPro & Vendor Scraper",
    goal="Identify missing fields in scraped JSON and suggest additional keys.",
    backstory="Expert web scraper for product data.",
    llm=crewai_llm,
    verbose=True,
)

refiner_agent = Agent(
    role="Data Refiner",
    goal="Merge TruckPro + vendor data, normalize, deduplicate, output clean JSON.",
    backstory="Specialist in structured data cleaning.",
    llm=crewai_llm,
    verbose=True,
)

description_agent = Agent(
    role="Accurate Product Description Writer",
    goal=(
        "Generate a structured product description using ONLY the provided REFINED_JSON and vendor data.\n\n"
        "Format:\n"
        "Product Description (paragraphs only, no bullets).\n\n"
        "**Accuracy Score:** X/100\n\n"
        "❌ Do not include price info.\n"
        "❌ Do not include Height, Width, Length, Weight, Parts Classification, VMRS Category, Unit of Measure info.\n"
        "✅ Only use facts that are verifiable in REFINED_JSON or vendor data.\n"
    ),
    backstory="Technical writer for vendor-ready structured product descriptions.",
    llm=crewai_llm,
    verbose=True,
)

validation_agent = Agent(
    role="Description Validator",
    goal=(
        "Validate the recommended description against REFINED_JSON. "
        "Return JSON with: valid_bullets, invalid_bullets, issues_count.\n\n"
        "Dont give price details.\n"
        "GIVE THE ACCURACY SCORE BY COMPARING THE GENERATED DESCRIPTION WITH THE ACTUAL DESCRIPTION ON THE TRUCKPRO PAGE AND VENDORS WEBSITE.\n"
    ),
    backstory="Expert in verifying product descriptions against structured data.",
    llm=crewai_llm,
    verbose=True,
)

# ============================
# HELPER FUNCTIONS
# ============================
def safe_extract_json(text: str) -> Dict[str, Any]:
    """Extracts first valid JSON object from text."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
    return {}

def export_to_csv(data_dict: Dict[str, Any]) -> str:
    """Export data to CSV format as string"""
    df = pd.DataFrame([data_dict])
    return df.to_csv(index=False)

# ============================
# MCP SERVER SETUP
# ============================
server = Server("truckpro-scraper")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="scrape_truckpro",
            description="Scrape a TruckPro product page and extract product information",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The TruckPro product URL to scrape"
                    }
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="enrich_with_vendor_data",
            description="Enrich product data with vendor information from search results",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_title": {
                        "type": "string",
                        "description": "Product title or name to search for vendor data"
                    }
                },
                "required": ["product_title"]
            }
        ),
        types.Tool(
            name="generate_product_description",
            description="Generate accurate product description using AI agents with TruckPro and vendor data",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The TruckPro product URL to process"
                    },
                    "existing_description": {
                        "type": "string",
                        "description": "Optional existing product description for comparison"
                    }
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="validate_description",
            description="Validate a product description against refined product data",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Product description to validate"
                    },
                    "refined_json": {
                        "type": "string",
                        "description": "JSON string of refined product data"
                    }
                },
                "required": ["description", "refined_json"]
            }
        ),
        types.Tool(
            name="export_product_data",
            description="Export processed product data to CSV format",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_data": {
                        "type": "string",
                        "description": "JSON string of product data to export"
                    }
                },
                "required": ["product_data"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls."""
    
    if name == "scrape_truckpro":
        url = arguments.get("url", "")
        if not url:
            return [types.TextContent(type="text", text="Error: URL is required")]
        
        try:
            product_data = parse_truckpro_product(url)
            if product_data:
                return [types.TextContent(
                    type="text", 
                    text=f"Successfully scraped TruckPro product:\n\n```json\n{json.dumps(product_data, indent=2, ensure_ascii=False)}\n```"
                )]
            else:
                return [types.TextContent(type="text", text="Failed to scrape TruckPro product page")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error scraping TruckPro page: {str(e)}")]
    
    elif name == "enrich_with_vendor_data":
        product_title = arguments.get("product_title", "")
        if not product_title:
            return [types.TextContent(type="text", text="Error: Product title is required")]
        
        try:
            vendor_data = fetch_vendor_data(product_title)
            return [types.TextContent(
                type="text",
                text=f"Vendor data for '{product_title}':\n\n```json\n{json.dumps(vendor_data, indent=2, ensure_ascii=False)}\n```"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error fetching vendor data: {str(e)}")]
    
    elif name == "generate_product_description":
        url = arguments.get("url", "")
        existing_description = arguments.get("existing_description", "")
        
        if not url:
            return [types.TextContent(type="text", text="Error: URL is required")]
        
        try:
            # Scrape TruckPro data
            tp_data = parse_truckpro_product(url)
            if not tp_data:
                return [types.TextContent(type="text", text="Failed to scrape TruckPro product page")]
            
            # Get vendor data
            vendor_data = fetch_vendor_data(tp_data.get("title") or tp_data.get("sku") or "")
            
            # Refine data using CrewAI
            refine_task = Task(
                description=(
                    f"TRUCKPRO_JSON:\n{json.dumps(tp_data, ensure_ascii=False, indent=2)}\n\n"
                    f"VENDOR_DATA:\n{json.dumps(vendor_data, ensure_ascii=False, indent=2)}\n\n"
                    "OUTPUT MUST BE A SINGLE VALID JSON OBJECT NAMED REFINED_JSON MERGING BOTH SOURCES.\n"
                    "Do NOT include any explanations or extra text outside the JSON."
                ),
                agent=refiner_agent,
                expected_output="A single valid JSON object named REFINED_JSON"
            )
            
            crew_refine = Crew(agents=[refiner_agent], tasks=[refine_task], verbose=False)
            refined_output = crew_refine.kickoff()
            refined_json = safe_extract_json(str(refined_output))
            
            if not refined_json:
                return [types.TextContent(type="text", text="Error: Failed to refine product data")]
            
            # Generate description
            desc_prompt = (
                "Generate a clear, professional product description in PARAGRAPH form using ONLY REFINED_JSON.\n\n"
                f"REFINED_JSON:\n{json.dumps(refined_json, ensure_ascii=False, indent=2)}\n"
                "Format:\n<Product Description in 1–2 short paragraphs>\n\n"
                "At the end, include:\nAccuracy Score: X\n"
            )
            desc_task = Task(
                description=desc_prompt,
                agent=description_agent,
                expected_output="Paragraph text followed by 'Accuracy Score: X'"
            )
            crew_desc = Crew(agents=[description_agent], tasks=[desc_task], verbose=False)
            recommended_desc_raw = crew_desc.kickoff()
            recommended_desc = str(recommended_desc_raw).strip()
            
            # Extract accuracy score
            acc_match = re.search(r"Accuracy Score:\s*(\d+)", recommended_desc)
            accuracy_score = int(acc_match.group(1)) if acc_match else None
            clean_desc = re.sub(r"Accuracy Score:\s*\d+", "", recommended_desc).strip()
            
            # Add vendor sources
            vendor_links = vendor_data.get("vendor_sources", [])[:2]
            sources_text = ""
            if vendor_links:
                sources_text = "\n\n**Sources:** " + ", ".join([f"Link {i+1}: {link}" for i, link in enumerate(vendor_links)])
            
            result = {
                "recommended_description": clean_desc,
                "accuracy_score": accuracy_score,
                "sources": sources_text,
                "truckpro_data": tp_data,
                "vendor_data": vendor_data,
                "refined_json": refined_json
            }
            
            response = f"""**Generated Product Description:**

{clean_desc}{sources_text}

**Accuracy Score:** {accuracy_score}/100

**Complete Data:**
```json
{json.dumps(result, indent=2, ensure_ascii=False)}
```"""
            
            return [types.TextContent(type="text", text=response)]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error generating description: {str(e)}")]
    
    elif name == "validate_description":
        description = arguments.get("description", "")
        refined_json_str = arguments.get("refined_json", "")
        
        if not description or not refined_json_str:
            return [types.TextContent(type="text", text="Error: Both description and refined_json are required")]
        
        try:
            refined_json = json.loads(refined_json_str)
            
            validation_task = Task(
                description=(
                    f"REFINED_JSON:\n{json.dumps(refined_json, ensure_ascii=False, indent=2)}\n\n"
                    f"PRODUCT_DESCRIPTION:\n{description}\n\n"
                    "Validate description. Output JSON: valid_bullets, invalid_bullets, issues_count."
                ),
                agent=validation_agent,
                expected_output="Validation JSON"
            )
            crew_validation = Crew(agents=[validation_agent], tasks=[validation_task], verbose=False)
            validation_result_raw = crew_validation.kickoff()
            validation_result = safe_extract_json(str(validation_result_raw))
            
            return [types.TextContent(
                type="text",
                text=f"Validation Result:\n\n```json\n{json.dumps(validation_result, indent=2, ensure_ascii=False)}\n```"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error validating description: {str(e)}")]
    
    elif name == "export_product_data":
        product_data_str = arguments.get("product_data", "")
        
        if not product_data_str:
            return [types.TextContent(type="text", text="Error: Product data is required")]
        
        try:
            product_data = json.loads(product_data_str)
            csv_data = export_to_csv(product_data)
            
            return [types.TextContent(
                type="text",
                text=f"Product data exported to CSV format:\n\n```csv\n{csv_data}\n```"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error exporting data: {str(e)}")]
    
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

class NotificationOptions:
    def __init__(self, tools_changed: bool = True, resources_changed: bool = True):
        self.tools_changed = tools_changed
        self.resources_changed = resources_changed


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="truckpro-scraper",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(
                        tools_changed=True,
                        resources_changed=True,
                    ),
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())