"""
Utility functions and helpers
"""

import re
import json
from typing import List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
import structlog
import sqlparse

logger = structlog.get_logger()


class Utils:
    """Consolidated utility functions with number formatting"""

    @staticmethod
    def format_number(value, decimal_places=2):
        """Format numbers to specified decimal places"""
        if value is None:
            return None

        try:
            if isinstance(value, (int, float, Decimal)):
                # Round to specified decimal places
                if isinstance(value, Decimal):
                    return float(round(value, decimal_places))
                else:
                    return round(float(value), decimal_places)
            elif isinstance(value, str):
                # Try to convert string to number
                try:
                    num_val = float(value)
                    return round(num_val, decimal_places)
                except (ValueError, TypeError):
                    return value  # Return original if not a number
            else:
                return value  # Return original for non-numeric types
        except (ValueError, TypeError, OverflowError):
            return value  # Return original if conversion fails

    @staticmethod
    def format_results_data(
        results: List[Dict[str, Any]], decimal_places=2
    ) -> List[Dict[str, Any]]:
        """Format all numeric values in query results to specified decimal places"""
        if not results:
            return results

        formatted_results = []
        for row in results:
            formatted_row = {}
            for key, value in row.items():
                formatted_row[key] = Utils.format_number(value, decimal_places)
            formatted_results.append(formatted_row)

        return formatted_results

    @staticmethod
    def safe_json_serialize(obj):
        """Safe JSON serialization for various data types with number formatting"""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return Utils.format_number(obj, 2)  # Format Decimal to 2 decimal places
        elif isinstance(obj, float):
            return Utils.format_number(obj, 2)  # Format float to 2 decimal places
        elif hasattr(obj, "__dict__"):
            return str(obj)
        return obj

    @staticmethod
    def normalize_question(question: str) -> str:
        """Normalize question for better cache hits"""
        question = question.lower().strip()
        question = re.sub(r"\s+", " ", question)
        return question

    @staticmethod
    def remove_sql_comments(sql: str) -> str:
        """Remove SQL comments while preserving string literals"""
        if not sql:
            return sql

        result = []
        i = 0
        in_string = False
        string_char = None

        while i < len(sql):
            char = sql[i]

            # Handle string literals
            if char in ("'", '"') and not in_string:
                in_string = True
                string_char = char
                result.append(char)
            elif char == string_char and in_string:
                in_string = False
                string_char = None
                result.append(char)
            elif in_string:
                result.append(char)
            # Handle comments only when not in string
            elif (
                not in_string and char == "-" and i + 1 < len(sql) and sql[i + 1] == "-"
            ):
                # Skip single-line comment
                while i < len(sql) and sql[i] != "\n":
                    i += 1
                if i < len(sql):  # Add newline back
                    result.append(" ")  # Replace comment with space
            elif (
                not in_string and char == "/" and i + 1 < len(sql) and sql[i + 1] == "*"
            ):
                # Skip multi-line comment
                i += 2
                while i + 1 < len(sql):
                    if sql[i] == "*" and sql[i + 1] == "/":
                        i += 1
                        break
                    i += 1
                result.append(" ")  # Replace comment with space
            else:
                result.append(char)

            i += 1

        return "".join(result)

    @staticmethod
    def parse_select_columns(select_part: str) -> list:
        """Parse SELECT clause to identify non-aggregate columns"""
        columns = []
        current_col = ""
        paren_count = 0

        # Split by comma, respecting parentheses
        for char in select_part:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            elif char == "," and paren_count == 0:
                if current_col.strip():
                    columns.append(current_col.strip())
                current_col = ""
                continue
            current_col += char

        if current_col.strip():
            columns.append(current_col.strip())

        # Filter out aggregate functions
        non_aggregate_columns = []
        aggregate_functions = [
            "COUNT(",
            "SUM(",
            "AVG(",
            "MAX(",
            "MIN(",
            "STDEV(",
            "VAR(",
        ]

        for col in columns:
            col_upper = col.upper()
            is_aggregate = any(func in col_upper for func in aggregate_functions)

            if not is_aggregate:
                # Extract column expression (before AS alias)
                if " AS " in col_upper:
                    col_expr = col[: col_upper.find(" AS ")].strip()
                else:
                    col_expr = col.strip()
                non_aggregate_columns.append(col_expr)

        return non_aggregate_columns

    @staticmethod
    def is_column_in_group_by(select_col: str, group_by_columns: list) -> bool:
        """Check if a SELECT column is present in GROUP BY clause"""
        sel_normalized = (
            select_col.replace("[", "").replace("]", "").replace(" ", "").upper()
        )

        for grp_col in group_by_columns:
            grp_normalized = (
                grp_col.replace("[", "").replace("]", "").replace(" ", "").upper()
            )
            if sel_normalized == grp_normalized or sel_normalized in grp_normalized:
                return True

        return False

    @staticmethod
    def validate_group_by_syntax(sql: str) -> tuple[str, str]:
        """Enhanced GROUP BY validation with automatic fixing"""
        try:
            sql = sql.replace("GROUP BYDATEPART", "GROUP BY DATEPART")
            sql = sql.replace("ORDER BYDATEPART", "ORDER BY DATEPART")
            sql = sql.replace("GROUP BY BY", "GROUP BY")
            sql_upper = sql.upper()

            # Check if this query uses GROUP BY
            if "GROUP BY" not in sql_upper:
                return sql, "No GROUP BY clause"

            # Extract SELECT and GROUP BY clauses
            select_start = sql_upper.find("SELECT")
            from_start = sql_upper.find("FROM")
            group_by_start = sql_upper.find("GROUP BY")

            if select_start == -1 or from_start == -1 or group_by_start == -1:
                return sql, "Invalid SQL structure"

            # Extract SELECT clause
            select_part = sql[select_start + 6 : from_start].strip()

            # Find GROUP BY clause boundaries
            order_by_start = sql_upper.find("ORDER BY", group_by_start)
            having_start = sql_upper.find("HAVING", group_by_start)

            group_by_end = len(sql)
            if order_by_start != -1:
                group_by_end = min(group_by_end, order_by_start)
            if having_start != -1:
                group_by_end = min(group_by_end, having_start)

            group_by_part = sql[group_by_start + 8 : group_by_end].strip()

            # Parse SELECT columns to find non-aggregates
            select_columns = Utils.parse_select_columns(select_part)
            group_by_columns = [
                col.strip() for col in group_by_part.split(",") if col.strip()
            ]

            # Find missing columns
            missing_columns = []
            for sel_col in select_columns:
                if not Utils.is_column_in_group_by(sel_col, group_by_columns):
                    missing_columns.append(sel_col)

            # Auto-fix if needed
            if missing_columns:
                if group_by_part:
                    new_group_by = group_by_part + ", " + ", ".join(missing_columns)
                else:
                    new_group_by = ", ".join(missing_columns)

                # Reconstruct SQL
                fixed_sql = sql[: group_by_start + 8] + new_group_by
                if group_by_end < len(sql):
                    fixed_sql += " " + sql[group_by_end:]

                fixed_sql = fixed_sql.replace("GROUP BYDATEPART", "GROUP BY DATEPART")
                fixed_sql = fixed_sql.replace("ORDER BYDATEPART", "ORDER BY DATEPART")

                return fixed_sql, f"Auto-fixed GROUP BY: Added {missing_columns}"

            return sql, "GROUP BY validation passed"

        except Exception as e:
            logger.warning("GROUP BY validation failed", error=str(e))
            return sql, f"GROUP BY validation error: {str(e)}"

    @staticmethod
    def clean_generated_sql(sql_text: str) -> str:
        """Enhanced SQL cleaning with comment removal, GROUP BY validation, and syntax fixes"""
        if not sql_text:
            return ""

        sql = sql_text.strip()

        # Remove code block markers
        if sql.startswith("```"):
            lines = sql.split("\n")
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines)
            for i, line in enumerate(lines[1:], 1):
                if line.strip().startswith("```"):
                    end_idx = i
                    break
            sql = "\n".join(lines[start_idx:end_idx])

        # Remove all comments from SQL
        sql = Utils.remove_sql_comments(sql)

        # Enhanced syntax fixes
        sql = sql.replace("GROUP BY", "GROUP BY ")
        sql = sql.replace("ORDER BY", "ORDER BY ")
        sql = sql.replace("GROUP  BY", "GROUP BY ")
        sql = sql.replace("ORDER  BY", "ORDER BY ")

        # Fix specific spacing issues
        sql = sql.replace("GROUP BY[", "GROUP BY [")
        sql = sql.replace("ORDER BY[", "ORDER BY [")
        sql = sql.replace("GROUP BYDATEPART", "GROUP BY DATEPART")
        sql = sql.replace("ORDER BYDATEPART", "ORDER BY DATEPART")

        sql = sql.replace(") GROUP BY", " GROUP BY")
        sql = sql.replace("GROUP BY GROUP BY", "GROUP BY")
        sql = sql.replace("WHERE SUM(", "HAVING SUM(")
        sql = sql.replace("WHERE COUNT(", "HAVING COUNT(")
        sql = sql.replace("WHERE AVG(", "HAVING AVG(")

        # Clean up the SQL
        lines = sql.split("\n")
        sql_lines = []
        in_select = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Start of a SELECT statement
            if line.upper().startswith("SELECT"):
                in_select = True
                sql_lines = [line]
            elif in_select:
                # Valid SQL keywords and constructs
                if any(
                    keyword in line.upper()
                    for keyword in [
                        "FROM",
                        "WHERE",
                        "JOIN",
                        "LEFT",
                        "RIGHT",
                        "INNER",
                        "ON",
                        "GROUP",
                        "HAVING",
                        "ORDER",
                        "AND",
                        "OR",
                        "CASE",
                        "WHEN",
                        "THEN",
                        "ELSE",
                        "END",
                    ]
                ):
                    sql_lines.append(line)
                elif line.upper().startswith("SELECT"):
                    # New SELECT statement, stop here
                    break
                elif any(
                    char in line
                    for char in [
                        "[",
                        "]",
                        ".",
                        ",",
                        "(",
                        ")",
                        "=",
                        "<",
                        ">",
                        "!",
                        "'",
                        '"',
                    ]
                ):
                    # Looks like SQL content
                    sql_lines.append(line)
                else:
                    # Doesn't look like SQL, might be end of query
                    if not any(char in line for char in ["[", "]", ".", ","]):
                        break
                    sql_lines.append(line)

        # Join and clean up
        sql = " ".join(sql_lines).strip().rstrip(";").rstrip(",")

        # Remove any remaining inline comments
        sql = Utils.remove_sql_comments(sql)

        # Final cleanup
        sql = sql.replace("GROUP BYDATEPART", "GROUP BY DATEPART")
        sql = sql.replace("ORDER BYDATEPART", "ORDER BY DATEPART")
        sql = sql.replace("GROUP BY BY", "GROUP BY")
        sql = sql.replace("ORDER BY BY", "ORDER BY")

        # Clean up extra spaces but preserve single spaces
        sql = re.sub(r"\s+", " ", sql)

        # Basic validation
        if sql:
            sql_upper = sql.upper()

            # Must start with SELECT and contain FROM
            if not sql_upper.startswith("SELECT") or "FROM" not in sql_upper:
                return ""

            # Check for obvious issues
            if any(
                issue in sql_upper
                for issue in ["FROM FROM", ", FROM", "SELECT FROM", "WHERE FROM"]
            ):
                return ""

            # Check for incomplete statements
            if any(
                sql_upper.endswith(keyword)
                for keyword in [
                    "FROM",
                    "SELECT",
                    "WHERE",
                    "AND",
                    "OR",
                    "JOIN",
                    "ON",
                    "GROUP BY",
                ]
            ):
                return ""

        return sql

    @staticmethod
    def sanitize_sql(sql: str) -> str:
        """Enhanced SQL sanitization with GROUP BY validation"""
        try:
            # First validate GROUP BY syntax
            sql, group_by_msg = Utils.validate_group_by_syntax(sql)
            if "error" in group_by_msg.lower():
                raise ValueError(f"GROUP BY validation failed: {group_by_msg}")

            parsed = sqlparse.parse(sql)[0]

            # Allowed keywords for security
            allowed_keywords = {
                "SELECT",
                "FROM",
                "WHERE",
                "JOIN",
                "INNER",
                "LEFT",
                "RIGHT",
                "OUTER",
                "GROUP",
                "BY",
                "ORDER",
                "HAVING",
                "AND",
                "OR",
                "ON",
                "AS",
                "IN",
                "NOT",
                "IS",
                "NULL",
                "LIKE",
                "BETWEEN",
                "EXISTS",
                "CASE",
                "WHEN",
                "THEN",
                "ELSE",
                "END",
                "COUNT",
                "SUM",
                "AVG",
                "MIN",
                "MAX",
                "TOP",
                "DISTINCT",
                "CAST",
                "CONVERT",
                "DATEPART",
                "DATEADD",
                "DATEDIFF",
                "GETDATE",
                "YEAR",
                "MONTH",
                "DAY",
                "ASC",
                "DESC",
            }

            # Check for dangerous keywords
            dangerous_keywords = {
                "DROP",
                "DELETE",
                "INSERT",
                "UPDATE",
                "ALTER",
                "CREATE",
                "TRUNCATE",
                "EXEC",
                "EXECUTE",
                "SP_",
                "XP_",
                "OPENROWSET",
                "OPENDATASOURCE",
            }

            sql_upper = sql.upper()
            for keyword in dangerous_keywords:
                if keyword in sql_upper:
                    raise ValueError(f"Dangerous SQL keyword detected: {keyword}")

            return str(parsed)

        except Exception as e:
            logger.error("SQL sanitization failed", error=str(e))
            raise ValueError(f"SQL sanitization failed: {str(e)}")

    @staticmethod
    def extract_context_from_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simplified version for KQL storage"""
        context = {}

        if not results:
            return context

        # Simple extraction for KQL storage
        sample_row = results[0]
        all_columns = list(sample_row.keys())

        # Store basic info for KQL
        context["_query_metadata"] = {
            "total_records": len(results),
            "columns_analyzed": all_columns,
            "timestamp": datetime.now().isoformat(),
        }

        return context
