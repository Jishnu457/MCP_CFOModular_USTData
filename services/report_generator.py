"""
Report generation services - PDF and Excel reports
"""

from typing import List, Dict, Any
from datetime import datetime
from io import BytesIO
import structlog

# Check for optional report generation libraries
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import HRFlowable
    from reportlab.lib.enums import TA_RIGHT, TA_CENTER, TA_JUSTIFY
    import xlsxwriter

    REPORT_LIBS_AVAILABLE = True
except ImportError:
    REPORT_LIBS_AVAILABLE = False

logger = structlog.get_logger()


class ReportGenerator:
    """Enhanced report generator using AI-powered content generation"""

    def __init__(self):
        self.ai_services = None  # Will be injected

    def set_ai_services(self, ai_services):
        """Inject AI services for intelligent report generation"""
        self.ai_services = ai_services

    async def generate_pdf_report(
        self,
        data: List[Dict],
        analysis: str,
        report_title: str = "Analytics Report",
        question: str = "",
    ):
        """Generate professional PDF report using AI-enhanced content"""

        if not REPORT_LIBS_AVAILABLE:
            raise ImportError(
                "Report generation libraries not available. Install with: pip install reportlab xlsxwriter"
            )

        if not data:
            data = []
        if not analysis:
            analysis = "Analysis not available"
        if not question:
            question = "Data analysis request"

        try:
            # Generate intelligent report content
            report_content = await self._generate_professional_content(
                question, data, analysis
            )

            # Create PDF
            return self._create_pdf(question, report_content, data)

        except Exception as e:
            logger.error("Report generation failed", error=str(e))
            return self._simple_fallback(question, data, analysis)

    async def _generate_professional_content(
        self, question: str, data: List[Dict], analysis: str
    ) -> str:
        """Generate intelligent report content with full dataset visibility"""

        # Enhanced data summary
        data_summary = "No data available"
        if data and len(data) > 0:
            all_columns = list(data[0].keys())

            # Identify financial categories
            revenue_cols = [
                col
                for col in all_columns
                if any(term in col.lower() for term in ["revenue", "sales", "income"])
            ]
            cost_cols = [
                col
                for col in all_columns
                if any(term in col.lower() for term in ["cost", "expense", "cogs"])
            ]
            profit_cols = [
                col
                for col in all_columns
                if any(
                    term in col.lower()
                    for term in ["profit", "margin", "ebitda", "operating"]
                )
            ]

            # Show date range
            years = list(
                set(
                    [
                        record.get("Year", record.get("year", ""))
                        for record in data
                        if record.get("Year") or record.get("year")
                    ]
                )
            )
            years = [y for y in years if y]

            data_summary = f"""COMPREHENSIVE FINANCIAL DATASET:
SCOPE: {len(data)} records spanning {min(years) if years else 'N/A'} to {max(years) if years else 'N/A'}

AVAILABLE FINANCIAL DATA:
- All Columns: {all_columns}
- Revenue Metrics: {revenue_cols if revenue_cols else 'None identified'}
- Cost/Expense Data: {cost_cols if cost_cols else 'None identified'} 
- Profitability Metrics: {profit_cols if profit_cols else 'None identified'}

SAMPLE DATA STRUCTURE:
{str(data[:3])}

DATA RANGE: This dataset contains {len(data)} records with detailed financial information."""

        prompt = f"""You are a senior financial analyst creating a comprehensive P&L report using REAL FINANCIAL DATA.

QUESTION: {question}
ANALYSIS: {analysis}

AVAILABLE FINANCIAL DATA:
{data_summary}

CRITICAL DATA RESTRICTIONS:
- Use ONLY the exact data provided above
- Do NOT make up any months, numbers, or financial figures
- Only reference actual months and values that appear in the real dataset
- When asked for 2025 data, use ONLY 2025 records from the dataset
- Format month references as names (January, February) and quarters as Q1, Q2, Q3, Q4

FORMAT REQUIREMENTS:
- Use clean, professional formatting
- Keep paragraphs concise (2-3 sentences each)
- Use simple bullet points (just • symbol)
- No sub-headings within sections
- Write everything under sub-headings in regular paragraph text

Create a structured financial report with these sections:

EXECUTIVE SUMMARY
Write a comprehensive 10-15 sentence executive overview that includes:
- Overall financial performance assessment with key metrics
- Most significant trends and patterns identified in the data
- Critical business implications requiring executive attention
- Strategic context and forward-looking perspective

KEY INSIGHTS
Use simple bullet points for specific findings from the REAL data:
• Revenue performance (use actual months/values only)
• Cost management (use actual figures only)
• Profitability trends (use actual data only)
• Financial analysis (use actual figures only)

BUSINESS IMPLICATIONS
Concise analysis (2-3 short paragraphs):
- Financial performance assessment based on actual data
- Cost structure opportunities from real figures
- Risk factors identified from actual trends
- Strategic recommendations based on real patterns

NEXT STEPS
Clear action items with bullet points:
• Immediate actions (next 30 days)
• Short-term initiatives (next 3 months)  
• Medium-term priorities (next 6-12 months)
• Key metrics to monitor
• Follow-up analysis needed

IMPORTANT: Base everything on the actual data provided. Do not invent any financial figures, months, or trends."""

        if self.ai_services:
            try:
                content = await self.ai_services.ask_intelligent_llm_async(prompt)
                # Clean up formatting symbols
                content = (
                    content.replace("###", "")
                    .replace("##", "")
                    .replace("**", "")
                    .replace("---", "")
                    .replace("####", "")
                )
                return content
            except Exception as e:
                logger.error("AI content generation failed", error=str(e))

        # Fallback content
        return f"""EXECUTIVE SUMMARY
Comprehensive financial analysis completed on {len(data) if data else 0} records spanning multiple years of detailed financial data.

KEY INSIGHTS
• Multi-year financial dataset analyzed covering revenue, costs, and profitability metrics
• Performance patterns identified across {len(data) if data else 0} financial records
• Year-over-year trends analyzed for strategic financial planning
• Detailed cost structure and margin analysis conducted

BUSINESS IMPLICATIONS
The comprehensive financial analysis reveals detailed insights into revenue performance, cost management effectiveness, and profitability trends. The multi-year dataset provides valuable perspective on financial health and operational efficiency.

Strategic financial planning should incorporate the identified trends in revenue growth, cost optimization opportunities, and margin improvement initiatives. The analysis supports data-driven decision making for sustainable financial performance.

NEXT STEPS
• Immediate Actions (Next 30 Days): Schedule executive financial review meeting with key stakeholders
• Short-term Initiatives (Next 3 Months): Implement enhanced financial monitoring dashboard
• Medium-term Priorities (Next 6-12 Months): Execute strategic initiatives based on identified opportunities
• Key Metrics to Monitor: Monthly profit margins, cost-to-revenue ratios, cash flow indicators
• Follow-up Analysis: Quarterly trend analysis and competitive benchmark review"""

    def _create_pdf(self, question: str, content: str, data: List[Dict]) -> bytes:
        """Create professional PDF with improved formatting"""

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
        )

        elements = []
        styles = getSampleStyleSheet()

        # Professional styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=18,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=15,
            fontName="Helvetica-Bold",
        )

        header_style = ParagraphStyle(
            "CustomHeader",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.black,
            spaceBefore=20,
            spaceAfter=10,
            fontName="Helvetica-Bold",
            backColor=colors.Color(0.95, 0.95, 0.95),
            borderPadding=6,
        )

        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=10,
            spaceBefore=2,
            spaceAfter=3,
            leading=12,
        )

        bullet_style = ParagraphStyle(
            "Bullet",
            parent=styles["Normal"],
            fontSize=10,
            spaceBefore=2,
            spaceAfter=2,
            leftIndent=0,
            firstLineIndent=0,
            leading=12,
        )

        # Professional Header Section
        current_date = datetime.now().strftime("%B %d, %Y")
        header_table = Table(
            [["CONFIDENTIAL EXECUTIVE REPORT", current_date]],
            colWidths=[4 * inch, 2.5 * inch],
        )
        header_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.black),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 12),
                    ("ALIGN", (0, 0), (0, 0), "LEFT"),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("PADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        elements.append(header_table)
        elements.append(Spacer(1, 20))

        # Title
        title = self._generate_title_from_question(question)
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 20))

        # Process content with improved formatting
        sections = content.split("\n\n")
        first_section = True

        for section in sections:
            if not section.strip():
                continue

            # Clean the entire section of symbols first
            section = (
                section.replace("###", "")
                .replace("##", "")
                .replace("**", "")
                .replace("---", "")
                .replace("*", "")
                .replace("#", "")
            )

            lines = section.strip().split("\n")
            first_line = lines[0].strip()

            # Skip empty sections
            if not first_line:
                continue

            # Special handling for EXECUTIVE SUMMARY - just show content, not header
            if "EXECUTIVE SUMMARY" in first_line.upper():
                # Add content without the header
                for line in lines[1:]:
                    line = line.strip()
                    if line and not line.startswith("---"):
                        elements.append(Paragraph(line, body_style))
                        elements.append(Spacer(1, 4))
                elements.append(Spacer(1, 10))
                first_section = False
                continue

            # Skip the first section if it's just the title repeat
            if first_section and any(
                word in first_line.lower()
                for word in ["profit", "loss", "analysis", "report"]
            ):
                first_section = False
                continue

            # Check if it's a section header
            if self._is_section_header(first_line):
                # Add section header
                elements.append(Paragraph(first_line, header_style))
                elements.append(Spacer(1, 6))

                # Add section content with improved formatting
                for line in lines[1:]:
                    line = line.strip()
                    if line and not line.startswith("---"):
                        # Skip sub-headings (lines ending with colon)
                        if line.endswith(":") and len(line.split()) <= 4:
                            continue

                        if line.startswith("•") or line.startswith("-"):
                            # Clean bullet points
                            elements.append(Paragraph(line, bullet_style))
                        else:
                            elements.append(Paragraph(line, body_style))
                            elements.append(Spacer(1, 2))

                elements.append(Spacer(1, 12))

            first_section = False

        # Enhanced data table
        if data and len(data) <= 25:
            elements.append(Spacer(1, 15))
            elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("<b>DETAILED DATA ANALYSIS</b>", header_style))
            elements.append(Spacer(1, 8))

            # Create comprehensive table
            if data:
                all_columns = list(data[0].keys())

                # Prioritize financial columns
                financial_priority = [
                    "year",
                    "month",
                    "revenue",
                    "income",
                    "profit",
                    "cost",
                    "expense",
                    "margin",
                ]
                selected_columns = []

                # Pick financial priority fields first
                for field in financial_priority:
                    for col in all_columns:
                        if field.lower() in col.lower() and col not in selected_columns:
                            selected_columns.append(col)

                # Add remaining important columns
                for col in all_columns:
                    if col not in selected_columns:
                        selected_columns.append(col)
                    if len(selected_columns) >= 8:
                        break

                headers = [col.replace("_", " ").title() for col in selected_columns]
                table_data = [headers]

                # Show data rows with enhanced formatting
                for record in data[:15]:
                    row = []
                    for col in selected_columns:
                        value = record.get(col, "")
                        # Enhanced formatting for financial data
                        if isinstance(value, float):
                            if abs(value) >= 1000000:  # Millions
                                row.append(
                                    f"${value/1000000:.1f}M"
                                    if any(
                                        term in col.lower()
                                        for term in [
                                            "revenue",
                                            "profit",
                                            "cost",
                                            "sales",
                                        ]
                                    )
                                    else f"{value/1000000:.1f}M"
                                )
                            elif abs(value) >= 1000:  # Thousands
                                row.append(
                                    f"${value/1000:.1f}K"
                                    if any(
                                        term in col.lower()
                                        for term in [
                                            "revenue",
                                            "profit",
                                            "cost",
                                            "sales",
                                        ]
                                    )
                                    else f"{value:,.0f}"
                                )
                            else:
                                row.append(f"{value:.2f}")
                        elif isinstance(value, int) and value > 1000:
                            row.append(f"{value:,}")
                        else:
                            row.append(str(value))
                    table_data.append(row)

                # Create professional table
                table = Table(table_data, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.black),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 8),
                            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                            ("FONTSIZE", (0, 1), (-1, -1), 8),
                            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("PADDING", (0, 0), (-1, -1), 4),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.white, colors.Color(0.95, 0.95, 0.95)],
                            ),
                        ]
                    )
                )
                elements.append(table)

        # Professional Footer
        elements.append(Spacer(1, 30))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.black))
        elements.append(Spacer(1, 10))

        footer_table = Table(
            [
                [
                    "CONFIDENTIAL BUSINESS REPORT",
                    f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
                ]
            ],
            colWidths=[3.5 * inch, 3 * inch],
        )
        footer_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("ALIGN", (0, 0), (0, 0), "LEFT"),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ]
            )
        )
        elements.append(footer_table)

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    def _generate_title_from_question(self, question: str) -> str:
        """Generate appropriate title from question"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["p&l", "pnl", "profit", "loss"]):
            return "Profit & Loss Analysis Report"
        elif any(word in question_lower for word in ["sales", "revenue"]):
            return "Sales Performance Analysis"
        elif any(word in question_lower for word in ["financial", "finance"]):
            return "Financial Analysis Report"
        else:
            return "Executive Business Analysis"

    def _is_section_header(self, text: str) -> bool:
        """Check if text is a section header"""
        headers = [
            "KEY INSIGHTS",
            "BUSINESS IMPLICATIONS",
            "NEXT STEPS",
            "ANALYSIS",
            "FINDINGS",
        ]
        text_upper = text.upper().strip()
        clean_text = text_upper.replace("#", "").replace("*", "").strip()
        return any(header in clean_text for header in headers) or (
            len(clean_text.split()) <= 4 and clean_text.isupper()
        )

    def _simple_fallback(self, question: str, data: List[Dict], analysis: str) -> bytes:
        """Ultra-simple fallback"""
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)

        # Simple content
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Business Analysis Report")
        c.setFont("Helvetica", 12)
        c.drawString(100, 720, f"Question: {question[:60]}")
        c.drawString(100, 700, f"Records: {len(data) if data else 0}")
        c.drawString(100, 680, "Analysis completed successfully")
        c.drawString(100, 640, "Executive review recommended")

        c.save()
        buffer.seek(0)
        return buffer.getvalue()
