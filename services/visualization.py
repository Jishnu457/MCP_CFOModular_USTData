"""
Visualization and chart generation services
"""

import json
from typing import List, Dict, Any, Optional
import structlog
from utils.helpers import Utils

logger = structlog.get_logger()


class VisualizationManager:
    """Consolidated visualization management"""

    def __init__(self, ai_services):
        self.ai_services = ai_services

    def should_generate_visualization(
        self, question: str, sql: str, results: List[Dict[str, Any]]
    ) -> bool:
        """Enhanced visualization detection"""
        if not results or len(results) < 1:
            return False

        # More comprehensive keyword detection
        chart_keywords = [
            "chart",
            "graph",
            "visualize",
            "plot",
            "display",
            "show",
            "trend",
            "distribution",
            "compare",
            "comparison",
            "percentage",
            "over time",
            "by",
            "breakdown",
            "analysis",
            "visual",
            "bar chart",
            "pie chart",
            "line chart",
            "histogram",
        ]

        # Check question for visualization intent
        question_lower = question.lower()
        has_viz_keywords = any(keyword in question_lower for keyword in chart_keywords)

        # Check if data is suitable for visualization
        if len(results) > 100:  # Too many data points
            return False

        columns = list(results[0].keys())
        numeric_cols = []
        categorical_cols = []

        # Better column type detection
        for col in columns:
            sample_values = [row[col] for row in results[:5] if row[col] is not None]
            if sample_values:
                if any(isinstance(val, (int, float)) for val in sample_values):
                    numeric_cols.append(col)
                elif any(isinstance(val, str) for val in sample_values):
                    categorical_cols.append(col)

        # Need at least one numeric and one categorical column, OR aggregated data
        has_suitable_data = (
            len(numeric_cols) >= 1 and len(categorical_cols) >= 1
        ) or len(results) <= 20

        # Always generate chart if explicitly requested
        explicit_chart_request = any(
            word in question_lower for word in ["chart", "graph", "plot", "visualize"]
        )

        return explicit_chart_request or (has_viz_keywords and has_suitable_data)

    async def generate_visualization(
        self, question: str, results: List[Dict], sql: str
    ) -> Optional[Dict]:
        """Enhanced visualization generation with explanatory text"""
        if not results:
            return None

            # Prepare data for chart
        columns = list(results[0].keys())
        print(f" Available columns: {columns}")

        try:
            # Analyze the best chart type
            prompt = f"""
            Analyze this data and recommend the best Chart.js configuration:
            
            Question: {question}
            Data Sample: {json.dumps(results[:3], default=Utils.safe_json_serialize)}
            Total Records: {len(results)}
            
            IMPORTANT RULES:
            1. Choose the best chart type: bar, line, pie, or doughnut
            2. For X-axis labels: Use Business Unit, Client, Year, or Category columns (NOT numeric values)
            3. For Y-axis values: Use Revenue, Profit, Amount, or other numeric columns
            4. Format large numbers: Use 26.7B instead of 26,700,000,000
            5. NEVER put Year/Date in Y-axis as numeric values
            6. If Year column exists, always use it for X-axis and choose 'line' chart for trends
            7. Make the chart readable with proper scientific notation (K, M, B)

            Chart Type Rules:
            - Use 'bar' for comparisons between categories
            - Use 'line' for Year/Quarter/Date on X-axis (time trends) and year-over-year trends
            - Use 'pie' for distributions/percentages
            - Use 'doughnut' for proportions

            Respond with complete Chart.js JSON configuration that:
            - Uses proper column for labels 
            - Formats Y-axis with scientific notation (1.2B, 500M, etc.)
            - Has readable titles and legends
            - Includes proper tooltips with formatted numbers

            
            Respond with JSON only:
            {{
                "chart_type": "recommended_type",
                "x_axis_column": "actual_column_name_from_data",
                "y_axis_column": "actual_column_name_from_data",
                "reasoning": "brief explanation"
            }}
            """

            response = await self.ai_services.ask_intelligent_llm_async(prompt)
            print(f" AI Chart Response: {response}")
            chart_analysis = json.loads(
                response.strip().lstrip("```json").rstrip("```").strip()
            )
            chart_type = chart_analysis.get("chart_type", "bar")

            # label_col = chart_analysis.get("x_axis_column", columns[0])
            # value_col = chart_analysis.get("y_axis_column", columns[-1])

            # print(f" AI Selected - X: {label_col}, Y: {value_col}")

        except Exception as e:
            logger.warning("Chart type analysis failed, using fallback", error=str(e))
            # Fallback logic
            question_lower = question.lower()
            if any(
                word in question_lower for word in ["trend", "over time", "timeline"]
            ):
                chart_type = "line"
            elif any(
                word in question_lower
                for word in ["distribution", "percentage", "proportion"]
            ):
                chart_type = "pie"
            else:
                chart_type = "bar"

        # Find the best label and value columns
        numeric_cols = []
        categorical_cols = []

        for col in columns:
            sample_values = [row[col] for row in results[:5] if row[col] is not None]
            if sample_values:
                if all(isinstance(val, (int, float)) for val in sample_values):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)

        if not numeric_cols:
            return None

        try:
            # Use AI's column selection if available
            if "chart_analysis" in locals():
                label_col = chart_analysis.get("x_axis_column", columns[0])
                value_col = chart_analysis.get("y_axis_column", columns[-1])
                print(f" Using AI Selected - X: {label_col}, Y: {value_col}")
            else:
                # Fallback logic
                label_col = categorical_cols[0] if categorical_cols else columns[0]
                value_col = numeric_cols[0]
                print(f" Using Fallback - X: {label_col}, Y: {value_col}")

        except:
            # Final fallback
            label_col = categorical_cols[0] if categorical_cols else columns[0]
            value_col = numeric_cols[0]
            print(f" Using Final Fallback - X: {label_col}, Y: {value_col}")
        # Choose label column (categorical first, then first column)
        # label_col = categorical_cols[0] if categorical_cols else columns[0]
        # value_col = numeric_cols[0]

        # Limit data points for better visualization
        chart_data = results[:20]  # Limit to 20 points max

        # Extract labels and values with formatting
        labels = []
        values = []

        for row in chart_data:
            label = str(row.get(label_col, "Unknown"))[:30]  # Truncate long labels
            value = row.get(value_col, 0)

            # Convert value to number and format to 2 decimal places
            try:
                if isinstance(value, str):
                    value = float(value) if "." in value else int(value)
                elif value is None:
                    value = 0
                value = Utils.format_number(value, 2)
            except:
                value = 0

            labels.append(label)
            values.append(value)

        # Generate chart explanation with formatted numbers
        total_value = sum(values)
        max_value = max(values) if values else 0
        min_value = min(values) if values else 0
        max_index = values.index(max_value) if values else 0
        min_index = values.index(min_value) if values else 0

        # Create contextual explanation based on chart type with formatted numbers
        if chart_type in ["pie", "doughnut"]:
            max_percentage = (max_value / total_value * 100) if total_value > 0 else 0
            explanation = f"This {chart_type} chart shows the distribution of {value_col.replace('_', ' ').lower()} across different {label_col.replace('_', ' ').lower()}. "
            explanation += f"The largest segment is '{labels[max_index]}' with {max_value:,.2f} ({max_percentage:.1f}% of total). "
            explanation += f"Total across all categories: {total_value:,.2f}."

        elif chart_type == "line":
            explanation = f"This line chart displays the trend of {value_col.replace('_', ' ').lower()} over {label_col.replace('_', ' ').lower()}. "
            if len(values) > 1:
                trend = (
                    "increasing"
                    if values[-1] > values[0]
                    else "decreasing" if values[-1] < values[0] else "stable"
                )
                explanation += f"The overall trend appears to be {trend}. "
            explanation += f"Peak value: {max_value:,.2f} at '{labels[max_index]}', lowest: {min_value:,.2f} at '{labels[min_index]}'."

        else:  # bar chart
            explanation = f"This bar chart compares {value_col.replace('_', ' ').lower()} across different {label_col.replace('_', ' ').lower()}. "
            explanation += f"'{labels[max_index]}' has the highest value at {max_value:,.2f}, while '{labels[min_index]}' has the lowest at {min_value:,.2f}. "
            if len(values) > 2:
                avg_value = sum(values) / len(values)
                explanation += f"Average value: {avg_value:,.2f}."

        # Create Chart.js configuration
        chart_config = {
            "type": chart_type,
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": value_col.replace("_", " ").title(),
                        "data": values,
                        "backgroundColor": [
                            "rgba(75, 192, 192, 0.8)",
                            "rgba(255, 99, 132, 0.8)",
                            "rgba(54, 162, 235, 0.8)",
                            "rgba(255, 206, 86, 0.8)",
                            "rgba(153, 102, 255, 0.8)",
                            "rgba(255, 159, 64, 0.8)",
                            "rgba(199, 199, 199, 0.8)",
                            "rgba(83, 102, 255, 0.8)",
                            "rgba(255, 99, 71, 0.8)",
                            "rgba(50, 205, 50, 0.8)",
                        ][: len(values)],
                        "borderColor": [
                            "rgba(75, 192, 192, 1)",
                            "rgba(255, 99, 132, 1)",
                            "rgba(54, 162, 235, 1)",
                            "rgba(255, 206, 86, 1)",
                            "rgba(153, 102, 255, 1)",
                            "rgba(255, 159, 64, 1)",
                            "rgba(199, 199, 199, 1)",
                            "rgba(83, 102, 255, 1)",
                            "rgba(255, 99, 71, 1)",
                            "rgba(50, 205, 50, 1)",
                        ][: len(values)],
                        "borderWidth": 2,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ').title()}",
                        "font": {"size": 16, "weight": "bold"},
                    },
                    "legend": {"display": chart_type in ["pie", "doughnut"]},
                },
            },
        }

        # Add scales for non-pie charts
        if chart_type not in ["pie", "doughnut"]:
            chart_config["options"]["scales"] = {
                "y": {
                    "beginAtZero": True,
                    "title": {
                        "display": True,
                        "text": value_col.replace("_", " ").title(),
                    },
                },
                "x": {
                    "title": {
                        "display": True,
                        "text": label_col.replace("_", " ").title(),
                    }
                },
            }

        # Return both chart config and explanation
        return {
            "chart_config": chart_config,
            "explanation": explanation,
            "chart_type": chart_type,
            "data_points": len(values),
            "total_value": total_value if chart_type in ["pie", "doughnut"] else None,
        }

    async def add_visualization_to_response(
        self, question: str, sql: str, results: List[Dict], response: Dict
    ):
        """Add visualization with explanation to response if appropriate"""
        try:
            if self.should_generate_visualization(question, sql, results):
                logger.info(
                    "Generating visualization",
                    question=question,
                    result_count=len(results),
                )

                # Generate chart config and explanation
                viz_data = await self.generate_visualization(question, results, sql)

                if viz_data:
                    response["visualization"] = viz_data["chart_config"]
                    response["chart_explanation"] = viz_data["explanation"]
                    response["chart_type"] = viz_data["chart_type"]
                    response["has_visualization"] = True
                    logger.info("Visualization added to response with explanation")

                    # Enhance the main analysis with chart context
                    if "analysis" in response:
                        response[
                            "analysis"
                        ] += f"\n\n**Chart Insights:**\n{viz_data['explanation']}"
                else:
                    logger.warning("Failed to generate chart config")
            else:
                logger.info("No visualization needed", question=question)

        except Exception as e:
            logger.error("Visualization generation failed", error=str(e))
